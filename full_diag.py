import numpy as np
from numpy.linalg import eig
from scipy.linalg import null_space
import matplotlib.pyplot as plt
from main import *
from scipy.linalg import expm

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

#c_daga_i fermions
def cd(j):
    if j == N: #PBC (implement everywhere)
        j = 1
    return c(j).conjugate().T


#number operator in terms of fermion operators
def N_hat():
    n = sum([cd(i)@c(i) for i in range(N)])
    return n

#Hamiltonian written in terms of these fermion operators, to compare with the 
#one written in terms of Pauli matrices
def Hc(J,g):
    h = -0.5*J*sum([cd(i)@c(i+1) + cd(i+1)@c(i)\
        + cd(i)@cd(i+1) + c(i+1)@c(i) for i in range(N-1)])\
        + 0.5*J*Uz2()@(cd(N-1)@c(0) + cd(0)@c(N-1)\
        + cd(N-1)@cd(0) + c(0)@c(N-1))\
        - J*g*(N/2.*np.eye(2**N) - N_hat())
    return h

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

#single momentum pair hamiltonian
def hk(k,J,g,phi):
    h = -J*(np.cos(k)-g)*(cdk(k,phi)@ck(k,phi) - ck(-k,phi)@cdk(-k,phi)) \
        - J*1j*np.sin(k)*(np.exp(-2j*phi)*cdk(k,phi)@cdk(-k,phi) - np.exp(2j*phi)*ck(-k,phi)@ck(k,phi))
    return h

#Hamiltonian in k-space, notice the unpaired terms
def Hck(K0_set,K1_set,j,g,phi):
    h = Peven()@sum([hk(k,J,g,phi) for k in K0_set]) + Podd()@(sum([hk(k,J,g,phi) for k in K1_set]) \
        -J*(cdk(0,phi)@ck(0,phi) - cdk(np.pi,phi)@ck(np.pi,phi)) \
        + J*g*(cdk(0,phi)@ck(0,phi) + cdk(np.pi,phi)@ck(np.pi,phi) - np.eye(2**N)))
    return h


#free fermions, defined with phi=0, Bogoliubov angle
def gamma(k,J,g):
    
    if k == 0:
        return cdk(0, phi = 0) #controllare definizione

    #these conditions guarantee that the gamma refers to the right eigenvalue
    if k > 0 and np.tan(b_angle) > (g-np.cos(k))/np.sin(k):
        uk = -1j*np.sin(b_angle)
        vk = np.cos(b_angle)
    
    if k < 0 and np.tan(b_angle) < (g-np.cos(k))/np.sin(k):
        uk = -1j*np.sin(b_angle)
        vk = np.cos(b_angle)
    
    b_angle = theta(k,J,g)
    
    uk = np.cos(b_angle)
    vk = -1j*np.sin(b_angle)
    
    
    if np.tan(b_angle) > (g-np.cos(k))/np.sin(k):
        uk = -1j*np.sin(b_angle)
        vk = np.cos(b_angle)
    
    ga = uk.conjugate()*ck(k, phi = 0) + vk.conjugate()*cdk(-k, phi = 0)
    
    return ga

def gammaD(k,J,g):
    return gamma(k,J,g).conjugate().T

'''
#free fermions, defined with phi=0 standard method
def gamma(k,J,g):
    
    if k == 0:
        return cdk(0, phi = 0) 
        
    if k == np.pi:
        return ck(np.pi, phi = 0)
    
    zk = -J*(np.cos(k)-g)
    yk = J*np.sin(k)
    
    uk = (e(k,J,g)+zk)/(np.sqrt(2*e(k,J,g)*(e(k,J,g)+zk)))
    vk = 1j*yk/(np.sqrt(2*e(k,J,g)*(e(k,J,g)+zk)))
    
    ga = uk.conjugate()*ck(k, phi = 0) + vk.conjugate()*cdk(-k, phi = 0)
    
    return ga

'''


'''
#free fermions, defined with phi=np.pi/4
def gamma(k,J,g):
    
    if k == 0:
        return cdk(0, phi = np.pi/4) 
        
    if k == np.pi:
        return ck(np.pi, phi = np.pi/4)
    
    zk = -J*(np.cos(k)-g)
    xk = -J*np.sin(k)
    
    uk = xk/(np.sqrt(2*(xk**2 + zk**2 - e(k,J,g)*zk)))
    vk = (e(k,J,g)-zk)/(np.sqrt(2*(xk**2 + zk**2 - e(k,J,g)*zk)))
    
    ga = uk.conjugate()*ck(k, phi = np.pi/4) + vk.conjugate()*cdk(-k, phi = np.pi/4)
    
    return ga
'''

def gammaD(k,J,g):
    return gamma(k,J,g).conjugate().T


def Hgamma(K0_set,K1_set,J,g):
    h = Peven()@(sum([e(k,J,g)*(gammaD(k,J,g)@gamma(k,J,g) + gammaD(-k,J,g)@gamma(-k,J,g) - np.eye(2**N)) for k in K0_set])) \
        + Podd()@(sum([e(k,J,g)*(gammaD(k,J,g)@gamma(k,J,g) + gammaD(-k,J,g)@gamma(-k,J,g) - np.eye(2**N)) for k in K1_set]) \
        + (J-J*g)*gammaD(0,J,g)@gamma(0,J,g) + (J+J*g)*gammaD(np.pi,J,g)@gamma(np.pi,J,g) - J*np.eye(2**N))
    return h

J, g = 1., 0.3

phi = 0

HI = H_Ising(J, g)
energy_levels, basis = eig(HI)

Hf = Hc(J, g) #OK! They are the same matrix

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

Hfk = Hck(K0p,K1p,J,g,phi) #OK! They are the same matrix


Hg = Hgamma(K0p,K1p,J,g)
energy_levels2, basis2 = eig(Hg)
energy_levels = np.round(energy_levels, 2)
energy_levels2 = np.round(energy_levels2, 2)

Htest = HI - Hg

e0 = min(energy_levels)

gs_counter = []

for i in range(2**N):
    if energy_levels[i] == e0:
        gs_counter.append(i)

if len(gs_counter) > 1:    
    print('Degenerate ground state')

GS = basis[:,gs_counter[0]] 

Gamma = [gamma(k,J,g) for k in K0]

for gamma in Gamma:
    test =  np.linalg.norm(gamma@GS) 
    print(test)
    
energy_levels.sort()
energy_levels2.sort()

print('Sum of Htest elements: ', sum([sum(Htest[i]) for i in range(2**N)]))



