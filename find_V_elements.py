import numpy as np
from Ising_chain_diagonalization import *
import itertools 

def generate_arrays(N):
    if N % 2 != 0:
        raise ValueError("N must be even")

    arrays = []
    for combination in itertools.product([0, 1], repeat=N):
        if combination.count(1) % 2 == 0:
            arrays.append(combination)

    return arrays

def even_energies_excitation(nk,J,g): 
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
            corr += l/N * np.sin(2**theta(K0[i+int(N/2)],J,g))**2
    if (nk == 0).all():
        corr = sum([l/N * np.sin(2**theta(k,J,g))**2 for k in K0p])
    return corr

'''Given the right vector written in the Fock basis,
this function return an array containing the sandwiches
between the rest of the basis and the given vector.'''

'''I have checked the non-null terms with N=4. '''

def od_V_elem(nks,nk_right,J,g,l): 
    counter = 0
    m_elem = []
    for nk_left in nks: #sum over the basis of the sector
        corr = 0
        if any(n != m for n,m in zip(nk_left,nk_right)): #only off-diagonal terms 
            for i in range(int(N/2),N): #computing a single off-diagonal term, sums over the positive momenta
                for j in range(int(N/2),N):    
                    if i == j: #check that if k=q the contribution is 0 (it should be granted by the other conditions, but it's faster this way)
                        corr += 0
                        #first term
                    elif nk_right[i] == 1 and nk_right[i-int(N/2)] == 1 and nk_right[j] == 1 and nk_right[j-int(N/2)] == 1 \
                        and nk_left[i] == 0 and nk_left[i-int(N/2)] == 0 and nk_left[j] == 0 and nk_left[j-int(N/2)] == 0:
                        s1 = (-1)**sum([nk_right[k] for k in range(0,i-int(N/2))])#ALWAYES BE CAREFUL WITH THE SIGN!!!
                        s2 = (-1)**sum([nk_right[k] for k in range(0,i)])
                        s3 = (-1)**sum([nk_left[k] for k in range(0,i-int(N/2))])
                        s4 = (-1)**(sum([nk_left[k] for k in range(0,i)]) + 1)
                        corr += s1*s2*s3*s4*np.sin(2*theta(K0[i],J,g))*np.sin(2*theta(K0[j],J,g))
                        #second term
                    elif nk_right[i] == 0 and nk_right[i-int(N/2)] == 0 and nk_right[j] == 0 and nk_right[j-int(N/2)] == 0 \
                        and nk_left[i] == 1 and nk_left[i-int(N/2)] == 1 and nk_left[j] == 1 and nk_left[j-int(N/2)] == 1:
                        s1 = (-1)**sum([nk_right[k] for k in range(0,i-int(N/2))])#HERE THE SIGN SHOULD WORK THE SAME WAY
                        s2 = (-1)**sum([nk_right[k] for k in range(0,i)])
                        s3 = (-1)**sum([nk_left[k] for k in range(0,i-int(N/2))])
                        s4 = (-1)**(sum([nk_left[k] for k in range(0,i)]))#except here
                        corr += s1*s2*s3*s4*np.sin(2*theta(K0[i],J,g))*np.sin(2*theta(K0[j],J,g))
                        #third term
                    elif nk_right[i] == 0 and nk_right[i-int(N/2)] == 0 and nk_right[j] == 1 and nk_right[j-int(N/2)] == 1 \
                        and nk_left[i] == 1 and nk_left[i-int(N/2)] == 1 and nk_left[j] == 0 and nk_left[j-int(N/2)] == 0:
                        s1 = (-1)**sum([nk_right[k] for k in range(0,i-int(N/2))])#HERE THE SIGN SHOULD WORK THE SAME WAY
                        s2 = (-1)**sum([nk_right[k] for k in range(0,i)])
                        s3 = (-1)**sum([nk_left[k] for k in range(0,i-int(N/2))])
                        s4 = (-1)**(sum([nk_left[k] for k in range(0,i)]) + 1)
                        corr += -s1*s2*s3*s4*np.sin(2*theta(K0[i],J,g))*np.sin(2*theta(K0[j],J,g)) #extra -1
                    #fourth term
                    elif nk_right[i] == 1 and nk_right[i-int(N/2)] == 1 and nk_right[j] == 0 and nk_right[j-int(N/2)] == 0 \
                        and nk_left[i] == 0 and nk_left[i-int(N/2)] == 0 and nk_left[j] == 1 and nk_left[j-int(N/2)] == 1:
                        s1 = (-1)**sum([nk_right[k] for k in range(0,i-int(N/2))])#HERE THE SIGN SHOULD WORK THE SAME WAY
                        s2 = (-1)**sum([nk_right[k] for k in range(0,i)])
                        s3 = (-1)**sum([nk_left[k] for k in range(0,i-int(N/2))])
                        s4 = (-1)**(sum([nk_left[k] for k in range(0,i)])) #excpet here
                        corr += -s1*s2*s3*s4*np.sin(2*theta(K0[i],J,g))*np.sin(2*theta(K0[j],J,g)) #extra -1
            
        m_elem.append(-l/N*corr)
            
        counter += 1 #next term of the sum over the basis
    
    return np.array(m_elem)
