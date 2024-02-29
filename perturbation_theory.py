import numpy as np
from Ising_chain_diagonalization import *

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
                         
    whole_basis : entire basis of the Hilbert space (Fock representation).

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

def first_order_energy_corrections(repeating_indices, matrix_elements): #deg_subspace
    '''
    Parameters
    ----------
    repeating_indices : list of the degenerate indices grouped with respect to the degenerate energy
    matrix_elements : diagonal elements of the perturbation, corrections outside the degenerate subspaces
        
    deg_subpace : array containing the diagolized perturbation in the degenerate subspaces,
                 the first index groups the degenerate subspace, the second one refers to whether you are
                 looking at the eigenvalues or the eigenvectors.
                 REMOVED, APPARENTLY NOT NEEDED

    Returns
    -------
    first order energy corrections.

    '''
    
    #there is a theorem that tells me that the good basis is the the ''fermionic one'' 
    #thanks to the existence of Ik which commutes with both H0 and V. This imply that
    #I can use non-degenerate perturbation theory for the first order corrections of the energies
    #Ik must have non-repeating eigenvalues in the degenerate subspace!! TO CHECK
    #up to N=10 the ttwo methods give the exact same corrections, this suggests the above is true
    
    #anyway, if eigenvalues of Ik do repeat, the problem is only in finding the corrections to the basis
    corr = np.zeros(2**(N-1))
    
    for i in range(2**(N-1)):
        corr[i] =  matrix_elements[i]
    '''
    
    #non-degenerate part
    for i in range(2**(N-1)):
        if all(i != j for row in repeating_indices for j in row):
            corr[i] =  matrix_elements[i]
            
    for i in range(len(repeating_indices[:])): #cycle over the different subspaces
        count = 0
        for ind in repeating_indices[i]: #cycle over the indices of a certain subspace
            corr[ind] = deg_subspace[i][0][count]
            count += 1 #I need to sum the right eigenvalue to each index
    '''
        
    return corr

def first_order_basis_corrections(repeating_indices, matrix_elements, energies): 
    '''
    

    Parameters
    ----------
    repeating_indices : list of the degenerate indices grouped with respect to the degenerate energy
    matrix_elements : off-diagonal matirx elements of the perturbation
    energies : eigenvalues of the unperturbed Hamiltonian

    Returns
    -------
    first order coefficients for the basis correction arranged in a matrix. The row index refers to the
    vector that is to be corrected, the column index refers to the vector in the expansion.

    '''
    
    Mcoeff = np.zeros((2**(N-1),2**(N-1)))
    
    #non-degenerate part
    for i in range(2**(N-1)):
        for j in range(2**(N-1)):
            if all(i != k and j != k for row in repeating_indices for k in row) and i != j:
                Mcoeff[i,j] =  matrix_elements[j][i]/(energies[i] - energies[j])
    return Mcoeff

def second_order_energy_corrections(repeating_indices, matrix_elements, energies): 
    '''
    

    Parameters
    ----------
    repeating_indices : list of the degenerate indices grouped with respect to the degenerate energy
    matrix_elements : off-diagonal matirx elements of the perturbation
    energies : eigenvalues of the unperturbed Hamiltonian

    Returns
    -------
    second order corrections to the energy

    '''
    
    corr = np.zeros(2**(N-1))
    

    #non-degenerate part
    for i in range(2**(N-1)):
        if all(i != j for row in repeating_indices for j in row):
            corr[i] =  sum([abs(matrix_elements[k][i])**2/(energies[i] - energies[k]) for k in range(2**(N-1)) if k != i])
            
    for i in range(len(repeating_indices[:])): #cycle over the different subspaces
        count = 0
        for ind in repeating_indices[i]: #cycle over the indices of a certain subspace
            corr[ind] = sum([abs(matrix_elements[k][ind])**2/(energies[ind] - energies[k]) for k in range(2**(N-1)) if all(k != l for l in repeating_indices[i])])
         #these are the right corrections for the same reason that allows me to use non-degenerate perturbation theory at first order. 
            count += 1 #I need to sum the right eigenvalue to each index
    return corr
