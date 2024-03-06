import numpy as np
from Ising_chain_diagonalization import *
from find_V_elements import *
from perturbation_theory import * 
import matplotlib.pyplot as plt
import time
start_time = time.time()

def GS_expansion(nks, g0, g): #H0(g0)'s GS expanded on H0(g)'s basis.
    '''
    Parameters
    ----------
    nks : basis written in the Fock representation, notation valid for both Hamiltonians
                         
    g0 : external field of the pre-quench Hamiltonian
    
    g : external field of the post-quench Hamiltonian

    Returns
    -------
    The list of the coeffiecients of the expansion of the ground state of the pre-quench Hamiltonian
    on the basis of the post-quench one.
    '''
    
    coeff = np.zeros(np.shape(nks)[0], dtype='complex')
    c = 0
    #fake_nks = np.zeros((np.shape(nks)[0],np.shape(nks)[1]))
    
    for nk in nks:
        if all(nk[i+int(N/2)] == nk[-i-1+int(N/2)] for i in range(0,int(N/2))): #non-null coefficients
            if any(n != 0 for n in nk): 
                coeff[c] = -1 #- sign due to operators swapping if couples are present
            else:
                coeff[c] = 1 #the ground state is the only term with a positive factor
            for i in range(0,int(N/2)): 
                k = K0p[i]
                j = 2*i
                if nk[i+int(N/2)] == 0:
                    coeff[c] *= np.cos(theta(k, J, g0) - theta(k, J, g)) 
                    #fake_nks[c,j] = 0
                    #fake_nks[c,j+1] = 0
                elif nk[i+int(N/2)] == 1:
                    coeff[c] *= -1j*np.sin(theta(k, J, g0) - theta(k, J, g)) 
                    #fake_nks[c,j] = 1
                    #fake_nks[c,j+1] = 1
        c += 1       
    
    return coeff

def time_evo(Psi, Mcoeff, e, t): 
    '''
    Parameters
    ----------
    Psi : coefficients of the expansion of the initial state Psi on the CORRECTED energy eigenbasis
    Mcoeff : matrix of the coefficients of the expansion of the corrected basis on the unperturbed one
    e : energy spectrum corrected to first order
    
    RK: they must be correctly ordered

    Returns
    -------
    The time-evolved coefficients at time t of the state Psi expanded on the UNPERTURBED basis 
    '''
    
    if len(e) != len(Psi):
        raise ValueError("Both lists must have the same length")
    
    coeff_t = np.array([np.exp(-1j*e[i]*t)*Psi[i] + sum([np.exp(-1j*e[j]*t)*Psi[j]*Mcoeff[j,i] for j in range(len(e))]) for i in range(len(e))])
    
    return coeff_t

#works if we keep the post-quench basis at the zeroth order (no corrections)
def mode_pop(ind,sign,Psi,nks):
    '''
    Parameters
    ----------
    ind : index of the momentum of the desired excitation k in K0p (k = K0p[ind])
    sign : '+' or '-', determines the sign of the momentum of the excitation
    Psi : coefficients of the expansion of the chosen vector on the unperturbed basis
    nks : basis in the Fock representation
    
    Returns
    -------
    The population of the excitation mode k in the state Psi
    '''
    
    #determining the correct indices to use in the basis element nk to refer to the desired k
    if sign == '+':
        i = ind + int(N/2)
        
    if sign == '-':
        i = -ind -1 + int(N/2)
        
    #checking the surviving terms in the expansion after the application of gamma_k
    #they are those basis elements with the desired momentum k
    c = 0
    surv_ind = [] #list of the indices of surviving terms, they refer to the elements of the basis
    for nk in nks:
        if nk[i] == 1 and Psi[c] != 0: #if the coefficients is 0 there is no need to save it, the term must be there in the first place
            surv_ind.append(c)
        c += 1
    '''
    
    new_coeffs = np.zeros(2**(N-1), dtype='complex')
    #applying gammak to Psi
    for k in surv_ind:
        for l in range(2**(N-1)): 
            if  nks[l,i] == 0 and all(nks[l,j] == nks[k,j] for j in range(N) if j != i):
                print(l)
                new_coeffs[l] = coeff_psi[k] 
    '''
                
    pop = 0
    for k in surv_ind:
        pop += abs(Psi[k])**2
    
    return pop


def generate_initial_state(N):
    # Generate N random complex numbers
    complex_numbers = np.random.rand(N) + 1j * np.random.rand(N)
    
    # Ensure the squared modulus of the first element is almost one
    complex_numbers[0] *= 0.99 / np.abs(complex_numbers[0])
    
    # Normalize the array to ensure the sum of squared moduli is 1
    normalization_factor = 1 / np.sqrt(np.sum(np.abs(complex_numbers)**2))
    complex_numbers *= normalization_factor
    
    return complex_numbers
