import numpy as np
from scipy.linalg import expm

#Number of 1/2-spin particles
N = 6

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

#Ki operators, sign function which counts the number of fermions which sit before site i
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

#Z2 symmetry operator 
def Uz2():
    U = np.eye(2**N)
    for i in range(N):
        U = U@sz(i)
    return U

#ci fermions
def c(j):
    if j == N: #PBC (implement everywhere)
        j = 1
    return K_hat()[j]@sp(j)

def cd(j):
    if j == N: #PBC (implement everywhere)
        j = 1
    return c(j).conjugate().T


#number operator in terms of fermion operators
def N_hat():
    n = sum([cd(i)@c(i) for i in range(N)])
    return n

#Fourier Transforms
def ck(k,phi):
    return np.exp(-1j*phi)/np.sqrt(N)*sum([np.exp(-1j*k*(i+1))*c(i) for i in range(N)])

def cdk(k,phi):
    return ck(k,phi).conjugate().T

#projectors on the even and odd sectors
def Peven(): #P0
    return 0.5*(np.eye(2**N) + expm(1j*np.pi*N_hat()))
    
def Podd(): #P1
    return 0.5*(np.eye(2**N) - expm(1j*np.pi*N_hat()))

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



#---------------Hamiltonians at each step, used for validation-----------------

#Quantum Ising chain in a transverse field Hamiltonian
def H_Ising(J,g):
    h = -J/2. * sum([sx(i)@sx(i+1) + g*sz(i) for i in range(N-1)])
    h += -J/2. * (sx(N-1)@sx(0) + g*sz(N-1)) #PBC
    return h

#Hamiltonian written in terms of these fermion operators
def Hc(J,g):
    h = -0.5*J*sum([cd(i)@c(i+1) + cd(i+1)@c(i)\
        + cd(i)@cd(i+1) + c(i+1)@c(i) for i in range(N-1)])\
        + 0.5*J*Uz2()@(cd(N-1)@c(0) + cd(0)@c(N-1)\
        + cd(N-1)@cd(0) + c(0)@c(N-1))\
        - J*g*(N/2.*np.eye(2**N) - N_hat())
    return h

#single momentum pair hamiltonian
def hk(k,J,g,phi):
    h = -J*(np.cos(k)-g)*(cdk(k,phi)@ck(k,phi) - ck(-k,phi)@cdk(-k,phi)) \
        - J*1j*np.sin(k)*(np.exp(-2j*phi)*cdk(k,phi)@cdk(-k,phi) - np.exp(2j*phi)*ck(-k,phi)@ck(k,phi))
    return h

#Hamiltonian in k-space, notice the unpaired terms
def Hck(K0_set,K1_set,J,g,phi):
    h = Peven()@sum([hk(k,J,g,phi) for k in K0_set]) + Podd()@(sum([hk(k,J,g,phi) for k in K1_set]) \
        -J*(cdk(0,phi)@ck(0,phi) - cdk(np.pi,phi)@ck(np.pi,phi)) \
        + J*g*(cdk(0,phi)@ck(0,phi) + cdk(np.pi,phi)@ck(np.pi,phi) - np.eye(2**N)))
    return h

#Hamiltonian written in terms of the free fermions
def Hgamma(K0_set,K1_set,J,g):
    h = Peven()@(sum([e(k,J,g)*(gammaD(k,J,g)@gamma(k,J,g) + gammaD(-k,J,g)@gamma(-k,J,g) - np.eye(2**N)) for k in K0_set])) \
        + Podd()@(sum([e(k,J,g)*(gammaD(k,J,g)@gamma(k,J,g) + gammaD(-k,J,g)@gamma(-k,J,g) - np.eye(2**N)) for k in K1_set]) \
        + (J-J*g)*gammaD(0,J,g)@gamma(0,J,g) + (J+J*g)*gammaD(np.pi,J,g)@gamma(np.pi,J,g) - J*np.eye(2**N))
    return h

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

