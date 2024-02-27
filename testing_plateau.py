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
    fake_nks = np.zeros((np.shape(nks)[0],np.shape(nks)[1]))
    
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
                    coeff[c] *= np.cos(theta(k, J, g0) - theta(k, J, g)) #vedere qua l'ordine corretto, al momento non importante
                    fake_nks[c,j] = 0
                    fake_nks[c,j+1] = 0
                elif nk[i+int(N/2)] == 1:
                    coeff[c] *= -1j*np.sin(theta(k, J, g0) - theta(k, J, g)) #vedere qua l'ordine corretto, al momento non importante
                    fake_nks[c,j] = 1
                    fake_nks[c,j+1] = 1
        c += 1       
    
    return coeff

def time_evo(coeff, e, t):
    '''
    

    Parameters
    ----------
    coeff : coefficients of the expansion on the energy eigenbasis
    e : energy spectrum
    
    RK: they must be correctly ordered

    Returns
    -------
    The time-evolved coefficients at time t.

    '''
    
    if len(e) != len(coeff):
        raise ValueError("Both lists must have the same length")
    
    coeff_t = np.array([np.exp(-1j*e[i]*t)*coeff[i] for i in range(len(e))])
    
    return coeff_t

def mode_pop(ind,sign,coeff_psi,nks):
    '''
    

    Parameters
    ----------
    ind : index of the momentum of the desired excitation k in K0p (k = K0p[ind])
    sign : '+' or '-', determines the sign of the momentum of the excitation
    coeff_psi : coefficients of the expansion of the chosen vector on the basis
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
        if nk[i] == 1 and coeff_psi[c] != 0: #if the coefficients is 0 there is no need to save it, the term must be there in the first place
            surv_ind.append(c)
        c += 1
    
    #this is actually not needed for now, 
    #as I need the modulus square of the vector and thus I don't need to know how the coefficcients are moved
    #it will be helpful later on, when I will need the whole basis
    '''
    new_coeffs = np.zeros(2**(N-1), dtype='complex')
    #applying gammak to Psi
    for k in surv_ind:
        #nks[k] is the surviving basis vector, I need to save
        for l in range(2**(N-1)):
            if nks[l][i] == 0 and all(nks[l][j] == nks[k][j] for j in range(N) and j != i): #this should give me only one match
            #test it
                new_coeffs[l] = coeff_psi[k] #the previous coefficients become the coeff for the element without k
                '''
    pop = 0
    for k in surv_ind:
        pop += abs(coeff_psi[k])**2
    
    return pop

J, g, l = 1., 0.5, 0.01 #post-quench
g0 = 0.2 #pre-quench

nks = np.array(generate_arrays(N))

#Energy spectrum of the post-quench Hamiltonian corrected to the first order
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

fo_energies_corr = first_order_energy_corrections(degenerate_indices, diag_elem) #Vdiag_ds
fo_even_energies = even_energies + fo_energies_corr

#Expansion of the ground state of H0(g0) in the basis of H0(g) (no correction to 
#the post-quench basis)
Psi0 = GS_expansion(nks, g0, g)
tfin = 1000

#TEST (CHECKED -> THE EXPANSION OF THE GS IS CORRECT as well as the computation of the populations!)
n0exp = np.sin(theta(K0p[2], J, g) - theta(K0p[2], J, g0))**2
n0 = mode_pop(2, '+', Psi0, nks)
print('Difference between the theoretical value and computed of the population at t=0: %f' %abs(n0-n0exp))

nt = []
taxis = []

#mode k = K0p[0]
for t in range(tfin):
    Psit = time_evo(Psi0, fo_even_energies, t)
    nt.append(mode_pop(0, '+', Psit, nks))
    taxis.append(t)

plt.figure(1)
plt.plot(taxis, nt, '-.')
plt.grid()
plt.title('Time evolution of the population of the mode k=%1.3f' %K0p[1])    

print("--- %s seconds ---" % (time.time() - start_time))
