import numpy as np
from numpy.linalg import eig
from scipy.linalg import null_space
import matplotlib.pyplot as plt
from Ising_Chain import *
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

J, g = 1., 0.5

phi = 0

HI = H_Ising(J, g)
Hf = Hc(J, g) #OK! They are the same matrix

K1 = K_odd()
K0 = K_even()


K0p = np.zeros(int(N/2))
t = 0
for n in range(1,int(N/2)+1):
    K0p[t] = (2*n-1.)*np.pi/N
    t += 1
    
K0p.sort()

K1p = np.zeros(int(N/2)-1)
s = 0
for n in range(1,int(N/2)):
    K1p[s] = 2*n*np.pi/N
    s += 1
    
K1p.sort()

Hfk = Hck(K0p,K1p,J,g,phi)

Htest = HI-Hfk
