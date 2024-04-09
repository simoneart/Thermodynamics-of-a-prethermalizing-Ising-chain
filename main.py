import numpy as np
from Ising_chain_diagonalization import *
from find_V_elements import *
from perturbation_theory import * 
from testing_plateau import generate_initial_state, mode_pop
import matplotlib.pyplot as plt
import time
from numba import jit


'''
I want to see the maximum number of sites I can consider without the perturbation.
I want the spectrum, the basis and the time evolution of a mode.
'''

start_time = time.time()

@jit
def std_time_evo(Psi, e, t): 
    '''
    Parameters
    ----------
    Psi : coefficients of the expansion of the initial state Psi on the energy eigenbasis
    e : energy spectrum 
    
    RK: they must be correctly ordered

    Returns
    -------
    The time-evolved coefficients at time t of the state Psi expanded on the energy basis 
    '''
    
    if len(e) != len(Psi):
        raise ValueError("Both lists must have the same length")
    
    coeff_t = np.array([np.exp(-1j*e[i]*t)*Psi[i] for i in range(len(e))])
    
    return coeff_t


J, g = 1., 0.5

print("Basis ready")

#Energy spectrum of the Hamiltonian
even_energies = np.array([even_energies_excitation(nk,J,g) for nk in nks])
        
print("Spectrum ready")
        
#generate an initial state that is close to the GS
Psi0 = generate_initial_state(2**(N-1))
num_steps = 100
dt = 0.1

nt = []
taxis = []

print("Initializing time evolution")
#mode k = K0p[1]
for k in range(num_steps): 
    Psit = std_time_evo(Psi0, even_energies, k*dt)
    nt.append(mode_pop(1, '+', Psit, nks))
    taxis.append(k*dt)

plt.figure(1)
plt.plot(taxis, nt, '-.')
plt.ylabel(r'$<n_k>$')
plt.xlabel(r'$t$')
plt.grid()
plt.title('Time evolution of the population of the mode k=%1.3f' %K0p[1])

print('Execution time:')
print("--- %s seconds ---" % (time.time() - start_time))
