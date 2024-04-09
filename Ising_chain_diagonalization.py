import numpy as np
import itertools 
from scipy.linalg import expm

#Number of 1/2-spin particles
N = 20
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

#K_i operators, sign function which counts the number of fermions which sit before site i
#the following function gives the whole set of these operators
def K_hat():
    k = np.zeros((N,2**N,2**N), dtype='complex')
    k[0] = np.eye(2**N)
    for j in range(1,N):
        k[j] = k[j-1]@(np.eye(2**N) - 2*sm(j-1)@sp(j-1))
    
    return k

#Gives the set of pseudomomenta in the even parity sector.
def K_even():
    k = np.zeros(N)
    c = 0
    for n in range(-int(N/2)+1,int(N/2)+1,1):
        k[c] = 2*np.pi/N*(n-0.5)
        c += 1
    k.sort()
    return k

#Gives the set of pseudomomenta in the odd parity sector.
def K_odd():
    k = np.zeros(N)
    c = 0
    for n in range(-int(N/2)+1,int(N/2)+1,1):
        k[c] = 2*np.pi/N*n
        c += 1
    k.sort()
    return k

#dispersion relation of the free fermions
def e(k,J,g):
    return np.sqrt(J**2*(1+g**2)-2*J**2*g*np.cos(k))

#Bogoliugov angle 
def theta(k,J,g):
    return 0.5*np.arctan(np.sin(k)/(np.cos(k)-g))

#ci fermions
def c(j):
    if j == N: #PBC (implement everywhere)
        j = 1
    return K_hat()[j]@sp(j)

def cd(j):
    if j == N: #PBC (implement everywhere)
        j = 1
    return c(j).conjugate().T

#Fourier Transforms
def ck(k,phi):
    return np.exp(-1j*phi)/np.sqrt(N)*sum([np.exp(-1j*k*(i+1))*c(i) for i in range(N)])

def cdk(k,phi):
    return ck(k,phi).conjugate().T

#free fermions, defined with phi=0, Bogoliubov angle
def gamma(k,J,g):
    
    #gammas for the unpaired values of k (0 and Pi)
    if k == 0:
        return cdk(0, phi = 0) 
        
    if k == np.pi:
        return ck(np.pi, phi = 0)
    
    b_angle = theta(k,J,g)
    
    uk = np.cos(b_angle)
    vk = -1j*np.sin(b_angle)
    
    #these conditions guarantee that the gamma refers to the right eigenvalue
    if k > 0 and k != np.pi and np.tan(b_angle) > (g-np.cos(k))/np.sin(k):
        uk = -1j*np.sin(b_angle)
        vk = np.cos(b_angle)
    
    if k < 0 and np.tan(b_angle) < (g-np.cos(k))/np.sin(k):
        uk = -1j*np.sin(b_angle)
        vk = np.cos(b_angle)
    
    ga = uk.conjugate()*ck(k, phi = 0) + vk.conjugate()*cdk(-k, phi = 0)
    
    return ga


def gammaD(k,J,g):
    return gamma(k,J,g).conjugate().T
#-------------------------PRELIMINARY INITIALIZATION---------------------------

#in order to use the gamma defined through the Bogoliubov angle, we fix phi = 0
phi = 0

#The values of the momenta depend only on the number of site, so we define them
#here.


K1 = K_odd()
K0 = K_even()

Kfull = np.array([k for K in (K0,K1) for k in K])
Kfull.sort()

K0p = np.zeros(int(N/2))
t = 0
for n in range(1,int(N/2)+1):
    K0p[t] = (2*n-1.)*np.pi/N
    t += 1
    
K0p.sort()

K1p = np.zeros(int(N/2)-1)
s = 0
for n in range(1,int(N/2)): #0 and Pi must not be included
    K1p[s] = 2*n*np.pi/N
    s += 1
    
K1p.sort()

Kfullp = np.array([k for K in (K0p,K1p) for k in K])
Kfullp.sort()

#----------------BASIS AND SPECTRUM IN THE EVEN SECTOR-------------------------
'''Given the number of sites, this function gives the basis in the Fock
notation, i.e. all the possible arrays of length N composed of 1s and 0s such 
that the number of 1s is even. The order in the array corresponds to the value
of the momentum of the excitation as generated above (K0).'''
def generate_arrays(N):
    if N % 2 != 0:
        raise ValueError("N must be even")

    arrays = []
    for combination in itertools.product([0, 1], repeat=N):
        if combination.count(1) % 2 == 0:
            arrays.append(combination)

    return arrays

'''Given a basis element as written above and the parameter of the Hamiltonian,
this functions returns its respective energy level.'''
'''REMEMBER: the true ground state of the system is the one of the even sector.'''
def even_energies_excitation(nk,J,g): 
    energy = -sum([e(k,J,g) for k in K0p]) #we start from the GS
    for i in range(N):
        if nk[i] == 1:
            energy += e(abs(K0[i]),J,g)
            
    if energy < 1e-14:
        energy = 0
    
    return energy

#basis in the fermionic notation
nks = np.array(generate_arrays(N))
