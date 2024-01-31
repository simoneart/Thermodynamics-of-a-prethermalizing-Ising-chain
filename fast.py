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

def find_repeating_indices(lst):
    """
    Find the indices at which each repeating value appears in the list.

    Parameters:
    - lst: List of values.

    Returns:
    - indices_dict: Dictionary where keys are values that appear more than once
                    and values are lists of corresponding indices.
    """
    counts = {}
    for i, value in enumerate(lst):
        if value in counts:
            counts[value].append(i)
        else:
            counts[value] = [i]

    repeating_indices_dict = {key: indices for key, indices in counts.items() if len(indices) > 1}
    return repeating_indices_dict

def degenerate_subspaces(energies, whole_basis):
    '''
    Parameters
    ----------
    energies : degenerate energies of the Hamiltonian
                         
    whole_basis : entire basis of the Hilbert space.

    Returns
    -------
    The eigenvectors relative to the degenerate subspaces, grouped for each 
    energy value.
    deg_space[0] is an array that contains the degenerate energies
    deg_space[1] is an array that contains the sets of eigenvectors spanning the 
    degenerate subspace of their respective energy.
    
    deg_space[1][i] select the set associated to energy deg_space[0][i]

    '''
    #finding the degenerate energies and their respective eigenvectors
    indices = find_repeating_indices(energies) 
    degenerate_indices = list(indices.values())
    degenerate_energies = list(indices.keys())
    
    deg_vectors = []
    
    for i in range(len(degenerate_energies)):
        proxy = []
        for n in degenerate_indices[i]:
            proxy.append(whole_basis[n])
        deg_vectors.append(np.array(proxy))
    
    return [degenerate_energies, deg_vectors]

def V_subspace(diag_elem, od_elem, repeating_indices):
    '''
    Parameters
    ----------
    diag_elem : diagonal elements of V on the whole basis
    
    od_elem : off-diagonal elements of V on the whole basis
    
    repeating_indices : indices that refers to the basis vectors in a particular
                        degenerate subspace

    Returns
    -------
    The matrix whose elements are <n|V|m> on the particular degenerate subspace 

    '''
    
    matrix = np.zeros((len(repeating_indices),len(repeating_indices)))
    c1 = 0
    
    for i in repeating_indices:
        c2 = 0
        for j in repeating_indices:
            if i == j:
                matrix[c1,c2] = diag_elem[i]
            if i != j:
                matrix[c1,c2] = od_elem[i][j]
            c2 += 1
        c1 += 1
        
    return matrix
    

def parity_separator(sector_energies, all_energies, whole_basis): 
    E = [e for e in sector_energies]
    sector_basis = []
    '''Must be appropriate, choose with care: '''
    epsilon = 1e-13
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



'''Given the vector written in the Fock basis, 
this function gives the diagonal matrix element of V.'''
def diag_V_elem(nk,J,g,l): 
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

def od_V_elem(basis,nks,nk_right,J,g,l): 
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
even_basis = parity_separator(even_energies,eig_val0,eig_vec0)

#this is needed for the degenracy degree
for i in range(2**(N-1)):
    if abs(even_energies[i]) < 1e-14:
        even_energies[i] = 0.

#matrix elements of V in the EVEN sector
diag_elem = np.array([diag_V_elem(nk,J,g,l) for nk in nks])
off_diag_elem = [od_V_elem(even_basis,nks,nks[i],J,g,l) for i in range(2**(N-1))]

#preliminaries for degenerate perturbation theory

#degenerate energies and their respective eigenvectors 
deg_space = degenerate_subspaces(even_energies, even_basis)


#Find the matrix <n|V|m> in each subspace 
indices = find_repeating_indices(even_energies) 
degenerate_indices = list(indices.values()) #I need the indices

V_deg_sub = [V_subspace(diag_elem, off_diag_elem, ind) for ind in degenerate_indices]

#Diagonalize them (RK: REMEMBER THAT EIG() RETURNS THE EIGENVECTORS AS COLUMNS)
Vdiag_deg_sub = [eig(V) for V in V_deg_sub]


                          
