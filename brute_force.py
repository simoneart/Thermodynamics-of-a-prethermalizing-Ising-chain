import numpy as np
import pickle 
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from numpy.linalg import eig
import matplotlib.pyplot as plt
from Ising_chain_diagonalization import *

#--------------------------------FUNCTIONS-------------------------------------

#z-axis magnetization operator
def mz(N):
    magz = sum(sz(i) for i in range(N))
    return magz

#time average of an operator A
def time_ave(H, T, A):
    
    def matrix_function(t,H,A):
        I = expm(1j*H*t)@A@expm(-1j*H*t)
        return I
    
    def differential_equation(t, y, H, A):
        matrix_at_t = matrix_function(t, H, A)
        dy_dt = matrix_at_t.flatten()  # Flatten the matrix to a 1D array
        return dy_dt
    
    initial_conditions = np.zeros((2**N,2**N), dtype='complex').flatten()
    
    solution = solve_ivp(differential_equation, (0, T), initial_conditions, args=(H, A), dense_output=True)

    ave = 1./T*solution.sol(T).reshape((2**N,2**N))
    
    return ave

def V(l,A,Aave):
    return l/N * ((A-Aave)/2.)@((A-Aave)/2.)

def state_evo(psi,H,NumSteps,dt): #I use hbar = 1
    psit = np.zeros((NumSteps,2**N), dtype='complex')
    psit[0] = psi #initial state
    for n in range(1,NumSteps):
        psit[n] = expm(-1j*H*dt)@psit[n-1] 
    return psit                            

def observable_EV(O,Psi):
    '''
    Parameters
    ----------
    O : observable of which I want to compute the expectation value
    Psi : state on which it is computed

    Returns
    -------
    expectation value

    '''
    return Psi.conjugate().T@O@Psi


#------------------------------------------------------------------------------

#PRE QUENCH HAMILTONIAN
J, g0 = 1., 8

H0 = H_Ising(J, g0)

energies0, eig_vectors0 = eig(H0)

egs = min(energies0)
GS = np.zeros(2**N)

c = 0
for i in range(2**N):
    if energies0[i] == egs:
        GS = eig_vectors0[:,i]
        c += 1
        
if c > 1:
    print("Degenerate ground state?!?")
    
#this is needed for the degenracy degree
for i in range(2**N):
    if abs(energies0[i]) < 1e-14:
        energies0[i] = 0.

#POST QUENCH HAMILTONIAN
g, l = 3., 0.

H0g = H_Ising(J, g)

#energiesg, eig_vectorsg = eig(H0g)

#rewriting H0(g0)'s GS in the basis of this hamiltonian

#do I need to do that??


Mz = mz(N)

T = 10000
Mbar = time_ave(H0g,T,Mz)
#Mcontrol = np.zeros((2**N,2**N))

#perturbed Hamiltonian
H = H0g + V(l, Mz, Mbar)

energies, eig_vectors = eig(H)

energies = np.real(energies)

#this is needed for the degenracy degree
for i in range(2**N):
    if abs(energies[i]) < 1e-14:
        energies[i] = 0.
        
mod = (N+2)//4 - 1

#number operator of an excitation mode, I choose a certain k it's not relevant
#which one.
n_k = gammaD(K0p[mod], J, g)@gamma(K0p[mod], J, g)

#Time evolution of the GS of H0(g0) with H
NumSteps=3000
dt=0.01
psit = state_evo(GS,H,NumSteps,dt)

#Computing the population of the examined mode in time
popk = np.zeros(NumSteps, dtype='complex')

for t in range(NumSteps):
    popk[t] = psit[t].conjugate().T@n_k@psit[t]
    
times = np.array([k*dt for k in range(NumSteps)])

plt.figure(1)
plt.plot(times, popk, '-.')
plt.ylabel(r'$<n_k>$')
plt.xlabel(r'$t$')
plt.grid()
plt.title('Time evolution of the population of the mode k=%1.3f' %K0p[mod])


