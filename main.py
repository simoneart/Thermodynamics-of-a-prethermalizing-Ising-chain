import numpy as np
from scipy.linalg import null_space
from numpy.linalg import eig
from Ising_chain_diagonalization import *
from find_V_elements import *
from perturbation_theory import *
import itertools 
import time
start_time = time.time()

#------------------------------------------------------------------------------
J, g, l = 1., 0.5, 0.01

#EVEN sector basis and spectrum
nks = np.array(generate_arrays(N)) #basis in the Fock representation, the order is important
even_energies = np.array([even_energies_excitation(nk,J,g) for nk in nks])
#this is needed for the degenracy degree
for i in range(2**(N-1)):
    if abs(even_energies[i]) < 1e-14:
        even_energies[i] = 0.

#matrix elements of V in the EVEN sector
diag_elem = np.array([diag_V_elem(nk,J,g,l) for nk in nks])
off_diag_elem = [od_V_elem(nks,nks[i],J,g,l) for i in range(2**(N-1))]


rep = find_repeating_indices(even_energies)
degenerate_indices = list(rep.values()) 
degenerate_energies = list(rep.keys())

#deg_space = degenerate_subspaces(even_energies, nks)

V_ds = [V_subspace(diag_elem, off_diag_elem, ind) for ind in degenerate_indices]

#Diagonalize them (RK: REMEMBER THAT EIG() RETURNS THE EIGENVECTORS AS COLUMNS)
Vdiag_ds = [eig(V) for V in V_ds]
fo_energies_corr = first_order_energy_corrections(degenerate_indices, diag_elem, Vdiag_ds)
fo_even_energies = even_energies + fo_energies_corr

so_energies_corr = second_order_energy_corrections(degenerate_indices, off_diag_elem, even_energies)
so_even_energies = fo_even_energies + so_energies_corr

#the degeneracy is not lifted
rep1 = find_repeating_indices(so_even_energies) 
degenerate_indices1 = list(rep1.values()) #for the third order I need these
degenerate_energies1 = list(rep1.keys())

'''Look into non-hermitianity (N=8 and 12, N=10 it doesn't happen')'''

'''The potential breaks some simmetries of the system, but not all of them.
Maybe this implies that the perturbation expansion will never break all the degeneracies
sinche H + V is degenerate.'''

print("--- %s seconds ---" % (time.time() - start_time))
