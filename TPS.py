import numpy as np

#Number of 1/2-spin particles
N = 10

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

#Bogoliugov angle 
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

#basis in the fermionic notation
nks = np.array(generate_binary_arrays(N0))

spec = np.array([TPS_energies_excitation(nk, J=1., g=8.) for nk in nks])

#----------------------MATRIX ELEMENTS OF THE POTENTIAL V----------------------

def diag_V_elem(nk,J,g,l): 
    corr = 0
    for i in range(N0): #only the paired momenta give contributions, so this is 
                        #easily modified
        if nk[i] == 1:
            corr += l/N * np.sin(2**theta(K0[i],J,g))**2
    if (nk == 0).all():
        corr = sum([l/N * np.sin(2**theta(k,J,g))**2 for k in K0p])
    return corr

'''This is still to validate'''
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
