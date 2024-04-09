import numpy as np
import itertools 
from scipy.linalg import expm

#Number of 1/2-spin particles
N = 20

#Gives the set of pseudomomenta in the even parity sector.
def K_even():
    k = np.zeros(N)
    c = 0
    for n in range(-int(N/2)+1,int(N/2)+1,1):
        k[c] = 2*np.pi/N*(n-0.5)
        c += 1
    k.sort()
    return k


#dispersion relation of the free fermions
def e(k,J,g):
    return np.sqrt(J**2*(1+g**2)-2*J**2*g*np.cos(k))

#Bogoliugov angle 
def theta(k,J,g):
    return 0.5*np.arctan(np.sin(k)/(np.cos(k)-g))

#-------------------------PRELIMINARY INITIALIZATION---------------------------

#in order to use the gamma defined through the Bogoliubov angle, we fix phi = 0
phi = 0

#The values of the momenta depend only on the number of site, so we define them
#here.

K0 = K_even()


K0p = np.zeros(int(N/2))
t = 0
for n in range(1,int(N/2)+1):
    K0p[t] = (2*n-1.)*np.pi/N
    t += 1
    
K0p.sort()


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
