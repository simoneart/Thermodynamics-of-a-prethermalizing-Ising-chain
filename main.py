import numpy as np
from numpy.linalg import eig
from scipy.linalg import null_space

#Number of 1/2-spin particles
N = 4


#single system Pauli matrices
sigmax = np.array([[0,1],[1,0]])
sigmay = np.array([[0,-1j],[1j,0]])
sigmaz = np.array([[1,0],[0,-1]])

#ladder operators
sigmap = 0.5*(sigmax+1j*sigmay)
sigmam = 0.5*(sigmax-1j*sigmay)
    

#embedding in higher dimension 
def sp(j):
    s = np.eye(1)
    for n in range(N):
        if n==j:
            s = np.kron(s,sigmap)
        else:
            s = np.kron(s,np.eye(2))
    return s

def sm(j):
    s = np.eye(1)
    for n in range(N):
        if n==j:
            s = np.kron(s,sigmam)
        else:
            s = np.kron(s,np.eye(2))
    return s

def sx(j):
    s = np.eye(1)
    for n in range(N):
        if n==j:
            s = np.kron(s,sigmax)
        else:
            s = np.kron(s,np.eye(2))
    return s

def sz(j):
    s = np.eye(1)
    for n in range(N):
        if n==j:
            s = np.kron(s,sigmaz)
        else:
            s = np.kron(s,np.eye(2))
    return s

#Quantum Ising chain in a transverse field Hamiltonian
def H_Ising(J,g):
    h = -J/2. * sum([sx(i)@sx(i+1) + g*sz(i) for i in range(N-1)])
    h += -J/2. * (sx(N-1)@sx(0) + g*sz(N-1)) #PBC
    return h


#Ki operators, sign function which counts the number of fermions which sit before site i
#the following function gives the whole set of these operators
def K_hat():
    k = np.zeros((N,2**N,2**N), dtype='complex')
    k[0] = np.eye(2**N)
    for j in range(1,N):
        k[j] = k[j-1]@(np.eye(2**N) - 2*sm(j-1)@sp(j-1))
    
    return k

#Given the number of sites, gives the set of pseudomomenta in the even parity sector.
#This is the sector that gives the true ground state of the system.
#To get all the excited state one needs to consider the odd parity sector as well (see notes).
def K_even():
    k = np.zeros(N)
    c = 0
    for n in range(-int(N/2)+1,int(N/2)+1,1):
        k[c] = 2*np.pi/N*(n-0.5)
        c += 1
    k.sort()
    return k

#Given the number of sites, gives the set of pseudomomenta in the odd parity sector.
def K_odd():
    k = np.zeros(N)
    c = 0
    for n in range(-int(N/2)+1,int(N/2)+1,1):
        k[c] = 2*np.pi/N*n
        c += 1
    k.sort()
    return k

#dispersion relation of the free fermions
def e(k,g):
    return np.sqrt(1+g**2-2*g*np.cos(k))

#Bogoliugov angle (uk = cos(theta), vk = sin(theta))
def theta(k,g):
    return 0.5*np.arctan(np.sin(k)/(g-np.cos(k)))

#annihiation operator of the free fermions 
def gamma(k,g):
    K_op = K_hat()
    b_angle = theta(k,g)
    
    real = np.zeros((2**N,2**N), dtype='complex')
    im = np.zeros((2**N,2**N), dtype='complex')
    
    for j in range(1,N+1):
        real += np.exp(-1j*k*j)*K_op[j-1]@sp(j-1)
        im += np.exp(-1j*k*j)*K_op[j-1]@sm(j-1)
    
    gamma = np.cos(b_angle)/np.sqrt(N)*real -1j*np.sin(b_angle)/np.sqrt(N)*im
            
    return gamma

if __name__ == "__main__":
    J, g = 1., 0.5
    
    H = H_Ising(J, g)
    
    energy_levels, basis = eig(H)
    
    
    #TESTING WHETHER THE GAMMAs ANNIHILATE THE GROUND STATE AS THEY SHOULD
    
    e0 = min(energy_levels)
    
    gs_counter = []
    
    for i in range(2**N):
        if energy_levels[i] == e0:
            gs_counter.append(i)
    
    if len(gs_counter) > 1:    
        print('Degenerate ground state')
    
    GS = basis[:,gs_counter[0]] 
    
    K1_values = K_odd()
    
    K0_values = K_even()
    
    K0_even_pos = []
    
    for k in K0_values:
        if k >= 0:
            K0_even_pos.append(k)
    
    #ground state energy from the free fermioin hamiltonian, must be equal to e0
    e0_2 = - sum([e(k,g) for k in K0_even_pos])
    
    K_operators = K_hat()
    
    Gamma = [gamma(k,g) for k in K0_values]
    
    for gamma in Gamma:
        test =  np.linalg.norm(gamma@GS) 
        print(test)
        
    
    
    test = null_space(Gamma[0])
    
    Gamma.pop(0)
    
    test_nullspace = np.array([test[:,i] for i in range(len(test[1]))])
    
    GS_2 = test_nullspace
    
    epsilon =  0.001 #10**(-15)
    
    for gamma in Gamma: 
        
        for k in range(np.shape(GS_2)[0]):
            b = []
            
            if np.linalg.norm(gamma@GS_2[k]) < epsilon:
                b.append(GS_2[k])
                
        GS_2 = b
    
    #norm = np.linalg.norm(Gamma[0]@GS[0])
    


#norm = np.linalg.norm(Gamma[0]@GS[0])
'''
