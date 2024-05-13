import numpy as np
import pickle
import matplotlib.pyplot as plt

#Number of 1/2-spin particles
N = 20

#Number of hard-core bosons (totally paired sector)
N0 = N//2

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

#Bogoliubov angle 
def theta(k,J,g):
    return 0.5*np.arctan(np.sin(k)/(np.cos(k)-g))

#--------------------------------QUASI MOMENTA---------------------------------

K0 = K_even()

K0p = np.zeros(int(N/2))
t = 0
for n in range(1,int(N/2)+1):
    K0p[t] = (2*n-1.)*np.pi/N
    t += 1
    
K0p.sort()

#--------------BASIS AND SPECTRUM IN THE TOTALLY PAIRED SECTOR-----------------
'''Here, I use the properties of the totally paired sector to define the basis
vectors in the Fock representation using only N0=N/2 elements. In this case, I 
can consider only the positive ks, knowing that the negative excitation is present
iff the positive one is (no unpaired excitations). 
'''

def generate_binary_arrays(N):
    if N <= 0:
        return [[]]
    else:
        smaller_arrays = generate_binary_arrays(N - 1)
        return [[0] + arr for arr in smaller_arrays] + \
    [[1] + arr for arr in smaller_arrays]

'''Given a basis element as written above and the parameter of the Hamiltonian,
this functions returns its respective energy level.'''
'''I modify this accordingly to the notation used for the TPS.'''

def TPS_energies_excitation(nk,J,g): 
    
    energy = -sum([e(k,J,g) for k in K0p]) #we start from the GS
    
    for i in range(N0):
        if nk[i] == 1:
            energy += 2*e(K0p[i],J,g)
            
    if abs(energy) < 1e-14:
        energy = 0
    
    return energy

#basis in the fermionic notation restricted to the totally paired sector
nks = np.array(generate_binary_arrays(N0))

#----------------------MATRIX ELEMENTS OF THE POTENTIAL V----------------------
def diag_V_elem(nk,J,g,l): 
    corr = 0
    for i in range(N0): #only the paired momenta give contributions, so this is 
                        #easily modified
        if nk[i] == 1:
            corr += l/N * np.sin(2**theta(K0p[i],J,g))**2
            
    if (nk == 0).all():
        corr = sum([l/N * np.sin(2**theta(k,J,g))**2 for k in K0p])
        
    return corr

def od_V_elem(nks,nk_right,J,g,l): 
    counter = 0
    m_elem = []
    for nk_left in nks: #sum over the basis of the sector
        corr = 0
        if any(n != m for n,m in zip(nk_left,nk_right)): 
            for i in range(N0): 
                for j in range(N0):    
                    if i == j: 
                        corr += 0
                        #the symmetry in the vectors in this basis greatly simplifies
                        #these expressions.
                        #first term
                    elif nk_right[i] == 1 and nk_right[j] == 1  \
                        and nk_left[i] == 0 and nk_left[j] == 0:
                        corr += -1.*np.sin(2*theta(K0p[i],J,g))\
                            *np.sin(2*theta(K0p[j],J,g))
                        #second term
                    elif nk_right[i] == 0 and nk_right[j] == 0  \
                        and nk_left[i] == 1 and nk_left[j] == 1:
                        corr += np.sin(2*theta(K0p[i],J,g))\
                            *np.sin(2*theta(K0p[j],J,g))
                        #third term
                    elif nk_right[i] == 0  and nk_right[j] == 1  \
                        and nk_left[i] == 1 and nk_left[j] == 0:
                        corr += np.sin(2*theta(K0[i],J,g))\
                            *np.sin(2*theta(K0[j],J,g)) 
                    #fourth term
                    elif nk_right[i] == 1 and nk_right[j] == 0 \
                        and nk_left[i] == 0 and nk_left[j] == 1 :
                        corr += -1.*np.sin(2*theta(K0[i],J,g))\
                            *np.sin(2*theta(K0[j],J,g)) 

        m_elem.append(-l/N*corr)
            
        counter += 1 #next term of the sum over the basis
    
    return np.array(m_elem)


#-----------------------------PERTURBATION THEORY------------------------------

'''
I can use non-degenerate perturbation theory!
'''

def Mcoeff_fo(matrix_elements, energies): 
    '''
    Parameters
    ----------
    matrix_elements : off-diagonal matirx elements of the perturbation
    energies : eigenvalues of the unperturbed Hamiltonian

    Returns
    -------
    first order coefficients for the basis correction arranged in a matrix. The row index refers to the
    vector that is to be corrected, the column index refers to the vector in the expansion.
    '''
    
    Mcoeff = np.zeros((2**N0,2**N0))
    
    for i in range(2**N0):
        for j in range(2**N0):
            if i != j:
                Mcoeff[i,j] =  matrix_elements[j][i]/(energies[i] - energies[j])
    
    return Mcoeff

J, g, l = 1., 3., 0.1 #if this and Silva's don't agree when l=0 then there is a macroscopic probelm

E = np.array([TPS_energies_excitation(nk,J,g) for nk in nks])

fo_E = E + np.array([diag_V_elem(nk,J,g,l) for nk in nks])

'''
with open('E_BF_6.pkl', 'rb') as file:
    true_E = pickle.load(file)

RK: from the comparison with true_E obtained by TFIC_brute_force, I notice that
if I choose l=1 as in the paper the perturbative expansion is not accurate. 
I also notice that, as exepected from perturbation theory, the corrections
are an overestimation of the true energy levels. (Comment referring to N=6)
'''

off_diag_elem = [od_V_elem(nks,nks[i],J,g,l) for i in range(2**N0)]
    
fo_basis_coeff = Mcoeff_fo(off_diag_elem, E)

#---------------------------MOVING BETWEEN BASIS-------------------------------
'''
The goal here is to write the pre-quench ground state in terms of the basis of 
the post-quench hamiltonian written using perturbation theory. In this way its 
coefficients can be plugged in time_evo (a function defined below)
'''

'''
Based on: Quantum Quench in the Transverse Field Ising chain I:
Time evolution of order parameter correlators. It is a valid expansion only for 
a low density of excitations. 
'''

def GS_expansion1(nks, g0, g): #H0(g0)'s GS expanded on H0(g)'s basis (the non-corrected one) 
    '''
    Parameters
    ----------
    nks : basis written in the Fock representation in the TPS (notation valid for both Hamiltonians)
                         
    g0 : external field of the pre-quench Hamiltonian
    
    g : external field of the post-quench Hamiltonian

    Returns
    -------
    The list of the coeffiecients of the expansion of the ground state of the pre-quench Hamiltonian
    on the basis of the post-quench one in the low density of excitations limit
    '''
    
    coeff = np.zeros(np.shape(nks)[0], dtype='complex')
    c = 0
    
    for nk in nks:
        if any(n != 0 for n in nk): 
            coeff[c] = -1 #- sign due to operators swapping if couples are present
        else:
            coeff[c] = 1 #the ground state is the only term with a positive factor
        for i in range(0,int(N/2)): 
            k = K0p[i]
            deltaTheta = theta(k, J, g0) - theta(k, J, g)
            if nk[i] == 0:
                coeff[c] *= np.cos(deltaTheta) 
            elif nk[i] == 1:
                coeff[c] *= -1j*np.sin(deltaTheta) 
                    
        c += 1       
    
    return coeff

g0 = 8.

gs0 = GS_expansion1(nks, g0, g)

def GS_expansion2(nks,coeff1,Mcoeff):
    '''
    Parameters
    ----------
    nks :  basis written in the Fock representation in the TPS
    coeff1 : coefficients of the expansion of the GS of H0(g0) on the basis of
            H0(g)
    Mcoeff : matrix with coefficients of the first-order perturbative correction
            to write the basis of H(g) as a linear combination of the basis of
            H0(g)

    Returns
    -------
    coeff2 : coefficients of the expansion of the GS of H0(g0) on the first-order
            corrected basis.
    '''
    
    coeff2 = np.array([coeff1[i] - \
            sum([coeff1[j]*Mcoeff[j,i] for j in range(2**N0)]) for i in range(2**N0)]) 
                 
        #controllare l'ordine di i e j in M, vabbé che è simmetrica...
    
    return coeff2

Psi0 = GS_expansion2(nks, gs0, fo_basis_coeff)


#-------------------EVOLUTION OF THE POPULATION OF A MODE----------------------

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
    
    coeff_t = np.array([np.exp(-1j*e[i]*t)*Psi[i] + \
            sum([np.exp(-1j*e[j]*t)*Psi[j]*Mcoeff[j,i] \
                 for j in range(len(e))]) for i in range(len(e))])
        
    #norm = 1. / np.sqrt(np.sum(np.abs(coeff_t)**2))
    
    #coeff_t *= norm
    
    return coeff_t

def mode_pop(ind,Psi,nks):
    '''
    Parameters
    ----------
    ind : index of the momentum of the desired excitation k in K0p (k = K0p[ind])
    Psi : coefficients of the expansion of the chosen vector on the unperturbed basis
    nks : basis in the Fock representation
    
    Returns
    -------
    The population of the excitation mode k in the state Psi
    '''
        
    #checking the surviving terms in the expansion after the application of gamma_k
    #they are those basis elements with the desired momentum k
    c = 0
    surv_ind = [] #list of the indices of surviving terms, they refer to the elements of the basis
    for nk in nks:
        if nk[ind] == 1 and Psi[c] != 0: #if the coefficients is 0 there is no need to save it, 
                                         #the term must be there in the first place
            surv_ind.append(c)
        c += 1
                
    pop = 0
    for k in surv_ind:
        pop += abs(Psi[k])**2
    
    return pop

num_steps = 500
dt = 0.1
   
nt = []
taxis = []
    
#mode k = K0p[1]
for k in range(num_steps): 
    Psit = time_evo(Psi0, fo_basis_coeff, fo_E, k*dt)
    nt.append(mode_pop(2, Psit, nks))
    taxis.append(k*dt)
    
plt.figure(1)
plt.plot(taxis, nt, '-.')
plt.ylabel(r'$<n_k>$')
plt.xlabel(r'$t$')
plt.grid()
plt.title('Time evolution of the population of the mode k=%1.3f' %K0p[2])
