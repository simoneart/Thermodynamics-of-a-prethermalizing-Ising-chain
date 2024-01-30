import numpy as np
from scipy.linalg import null_space
from numpy.linalg import eig
from Ising_chain_diagonalization import *
import itertools #study

def generate_arrays(N):
    if N % 2 != 0:
        raise ValueError("N must be even")

    arrays = []
    for combination in itertools.product([0, 1], repeat=N):
        if combination.count(1) % 2 == 0:
            arrays.append(combination)

    return arrays


def parity_separator(sector_energies, all_energies, whole_basis): 
    E = [e for e in sector_energies]
    sector_basis = []
    '''Must be appropriate, choose with care: '''
    epsilon = 10**(-13) 
    '''This is not a reliable method.'''
    
    for j in range(2**N):
        for i in range(len(E)):
            if abs(E[i] - all_energies[j]) < epsilon:
                sector_basis.append(whole_basis[j])
                break
    
    if np.shape(sector_basis)[0] != 2**(N-1):
        print("WARNING: the sector basis does not have the right dimension!!")
            
    '''This break guarantees the right order in the arrangement of the basis vectors while
    counting the degeneracies the right amount of times. In fact, in this way I associate one vector 
    to one energy.'''
    #MAKE SURE THIS IS CORRECT!!! THINK ABOUT IT AGAIN!!!
                
    return np.array(sector_basis)

#----------------------------------EVEN SECTOR!!-------------------------------
def even_excitation_energy(nk,J,g): 
    energy = -sum([e(k,J,g) for k in K0p]) #we start from the GS
    for i in range(N):
        if nk[i] == 1:
            energy += e(abs(K0[i]),J,g)
    return energy

'''I NEED TO USE DEGENERATE PERTURBATION THEORY!!!!'''

'''Given the vector written in the Fock basis, 
this function gives the diagonal matrix element of V.'''
def even_correction_term(nk,J,g,l): 
    corr = 0
    for i in range(0,int(N/2)): #establishing the ks that give a contribution
        if nk[i+int(N/2)] == 1 and  nk[-i-1+int(N/2)] == 1:
            corr += l/N*np.sin(2**theta(K0[i+int(N/2)],J,g))**2
    if (nk == 0).all():
        corr = sum([l/N*np.sin(2**theta(k,J,g))**2 for k in K0p])
    return corr


'''Given the right vector written in the Fock basis,
this function return an array containing the sandwiches
between the rest of the basis and the given vector.'''
'''I have checked the non-null terms with N=4. '''
def od_matrix_elem(basis,nks,nk_right,J,g,l): 
    counter = 0
    m_elem = []
    for nk_left in nks: #sum over the basis of the sector
        corr = 0
        if any(n != m for n,m in zip(nk_left,nk_right)): #only off-diagonal terms are non zero
            for i in range(int(N/2),N): #computing a single off-diagonal term, sums over the positive momenta
                for j in range(int(N/2),N):    
                    if i == j: #check that if k=q the contribution is 0 (it should be granted by the other conditions, but it's faster this way)
                        corr += 0
                    else:
                        #first term
                        if nk_right[i] == 1 and nk_right[i-int(N/2)] == 1 and nk_right[j] == 1 and nk_right[j-int(N/2)] == 1 \
                            and nk_left[i] == 0 and nk_left[i-int(N/2)] == 0 and nk_left[j] == 0 and nk_left[j-int(N/2)] == 0:
                            s1 = (-1)**sum([nk_right[k] for k in range(0,i-int(N/2))])#ALWAYES BE CAREFUL WITH THE SIGN!!!
                            s2 = (-1)**sum([nk_right[k] for k in range(0,i)])
                            s3 = (-1)**sum([nk_left[k] for k in range(0,i-int(N/2))])
                            s4 = (-1)**(sum([nk_left[k] for k in range(0,i)]) + 1)
                            corr += s1*s2*s3*s4*np.sin(2*theta(K0[i],J,g))*np.sin(2*theta(K0[j],J,g))
                        #second term
                        if nk_right[i] == 0 and nk_right[i-int(N/2)] == 0 and nk_right[j] == 0 and nk_right[j-int(N/2)] == 0 \
                            and nk_left[i] == 1 and nk_left[i-int(N/2)] == 1 and nk_left[j] == 1 and nk_left[j-int(N/2)] == 1:
                            s1 = (-1)**sum([nk_right[k] for k in range(0,i-int(N/2))])#HERE THE SIGN SHOULD WORK THE SAME WAY
                            s2 = (-1)**sum([nk_right[k] for k in range(0,i)])
                            s3 = (-1)**sum([nk_left[k] for k in range(0,i-int(N/2))])
                            s4 = (-1)**(sum([nk_left[k] for k in range(0,i)]))#except here
                            corr += s1*s2*s3*s4*np.sin(2*theta(K0[i],J,g))*np.sin(2*theta(K0[j],J,g))
                        #third term
                        if nk_right[i] == 0 and nk_right[i-int(N/2)] == 0 and nk_right[j] == 1 and nk_right[j-int(N/2)] == 1 \
                            and nk_left[i] == 1 and nk_left[i-int(N/2)] == 1 and nk_left[j] == 0 and nk_left[j-int(N/2)] == 0:
                            s1 = (-1)**sum([nk_right[k] for k in range(0,i-int(N/2))])#HERE THE SIGN SHOULD WORK THE SAME WAY
                            s2 = (-1)**sum([nk_right[k] for k in range(0,i)])
                            s3 = (-1)**sum([nk_left[k] for k in range(0,i-int(N/2))])
                            s4 = (-1)**(sum([nk_left[k] for k in range(0,i)]) + 1)
                            corr += -s1*s2*s3*s4*np.sin(2*theta(K0[i],J,g))*np.sin(2*theta(K0[j],J,g)) #extra -1
                        #fourth term
                        if nk_right[i] == 1 and nk_right[i-int(N/2)] == 1 and nk_right[j] == 0 and nk_right[j-int(N/2)] == 0 \
                            and nk_left[i] == 0 and nk_left[i-int(N/2)] == 0 and nk_left[j] == 1 and nk_left[j-int(N/2)] == 1:
                            s1 = (-1)**sum([nk_right[k] for k in range(0,i-int(N/2))])#HERE THE SIGN SHOULD WORK THE SAME WAY
                            s2 = (-1)**sum([nk_right[k] for k in range(0,i)])
                            s3 = (-1)**sum([nk_left[k] for k in range(0,i-int(N/2))])
                            s4 = (-1)**(sum([nk_left[k] for k in range(0,i)])) #excpet here
                            corr += -s1*s2*s3*s4*np.sin(2*theta(K0[i],J,g))*np.sin(2*theta(K0[j],J,g)) #extra -1
            
        m_elem.append(-l/N*corr)
            
        counter += 1 #next term of the sum over the basis
    
    return np.array(m_elem)

    
#------------------------------------------------------------------------------
J, g, l = 1., 0.5, 0.1

#complete spectrum and basis
H0 = H_Ising(J, g)
eig_val0, eig_vect0 = eig(H0)
eig_vec0 = np.array([eig_vect0[:,i] for i in range(len(eig_vect0[1]))])

#EVEN sector basis and spectrum
nks = np.array(generate_arrays(N)) #basis in the Fock representation, the order is important
even_energies = np.array([even_excitation_energy(nk,J,g) for nk in nks])
even_corrections = np.array([even_correction_term(nk,J,g,l) for nk in nks])
even_perturbated_energies = even_energies + even_corrections

'''
#this is needed for the degenracy degree
for i in range(2**(N-1)):
    if abs(even_energies[i]) < 10**(-14):
        even_energies[i] = 0.
'''

even_basis = parity_separator(even_energies,eig_val0,eig_vec0)
even_matrix_elemets = [od_matrix_elem(even_basis,nks,nks[i],J,g,l) for i in range(2**(N-1))]
  



