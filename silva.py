import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

'''
Diagonalization via Williamson's theorem: a symmetric, positive-definite 2nx2n
matrix can be always brought into diagonal form via a simplectic transformation
and the corresponding spectrum is positive and doubly degenerate.
'''

#Number of 1/2-spin particles
N = 80

#parameters of the Hamiltonians pre- and post- quench
J = 1.
l = 0.01
g = 0.3
g0 = 0.2

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

#lower block entries of Csi
def Csi_entries_B2(k1,k2):
    if k1 == k2:
        ent = e(k1, J, g)
        return ent
    else: 
        ent = 2*l/N*np.sin(2*theta(k1, J, g))*np.sin(2*theta(k2, J, g))
        return ent

'''Block-diagonalization as in eq.(D25)'''
def D25(K):
    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(K)
    
    # Sort eigenvalues and eigenvectors
    #idx = np.argsort(eigenvalues)
    #sorted_eigenvalues = eigenvalues[idx]
    #sorted_eigenvectors = eigenvectors[:, idx]
    
    # Forming the orthogonal transformation matrix R
    R = eigenvectors @ np.diag(np.sqrt(np.abs(eigenvalues))) @ eigenvectors.T
    
    # Calculate block-diagonal matrix
    block_diagonal_matrix = R.T @ K @ R
    
    # Extracting B from the off-diagonal blocks of R^T K R
    A = block_diagonal_matrix[:N//2, N//2:]
    B = block_diagonal_matrix[N//2:, :N//2]
    
    #farsi sputare quella positiva
    return [abs(A), R]

'''
Matrices encoding the information on the initial datum
'''

def W0_entries(k1,k2):
    if k1 == k2:
        w = np.sin(theta(k1, J, g)-theta(k1, J, g0))**2
        return w
    else:
        w = 0.25*np.sin(theta(k1, J, g)-theta(k1, J, g0))* \
            np.sin(theta(k2, J, g)-theta(k2, J, g0))
        return w
    
def W1_entries(k1,k2):
    if k1 == k2:
        w = 0.
        return w
    else:
        w = -0.25*np.sin(theta(k1, J, g)-theta(k1, J, g0))* \
        np.sin(theta(k2, J, g)-theta(k2, J, g0))
        return w
    
def curlyW():
    idn = np.eye(int(N/2))
    W0 = np.array([[W0_entries(k1,k2) for k1 in K0p] for k2 in K0p])
    W1 = np.array([[W1_entries(k1,k2) for k1 in K0p] for k2 in K0p])
    W = np.block([[W1, idn+W0],[W0, W1]])
    return W

'''Gli indici saranno ordinati in modo corretto ??'''
def pops(k_index,times,A,B,Z1daga,Z0,E):
    nk = np.zeros(len(times))
    for t in range(len(times)):
        for i in range(len(K0p)):
            for j in range(len(K0p)):
                nk[t] += 2*np.real(A[k_index,i].conjugate()*B[k_index,j]*Z1daga[i,j]*np.exp(1j*(E[i]+E[j])*times[t])) +\
                (A[k_index,i].conjugate()*A[k_index,j] + B[k_index,i]*B[k_index,j].conjugate())\
                    *Z0[i,j]*np.exp(1j*(E[i]-E[j])*times[t])
    return nk

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

'''
I work in the totally paired sector {000...} thus the Hamiltonian is restricted
to the N/2 momenta in K0p (i.e. in eq.(28) the sum runs over k in K0p)
'''

'''
A function that gives the matrix for the Bogoliubov transformation and the (positive) energy
spectrum of the one-body bosonic problem
'''
def Bogoliubov_Matrix():
    '''
    Now I define the matrix of eq.(D23). 
    The upper left block is diagonal with the unperturbed eigenvalues ek.
    The bottom right block is given by the function Csi_entries_B2. 
    '''
    
    Csi_B1 = np.diag([e(k,J,g) for k in K0p])
    Csi_B2 = np.array([[Csi_entries_B2(k1,k2) for k1 in K0p] for k2 in K0p])
    Csi = np.block([[Csi_B1,np.zeros_like(Csi_B1)],[np.zeros_like(Csi_B1),Csi_B2]])
    
    '''
    Now I can define the K matrix
    '''
    
    A = sqrtm(np.linalg.inv(Csi))
    idn = np.eye(int(N/2))
    Omega = np.block([[np.zeros_like(idn), idn],[-idn, np.zeros_like(idn)]])
    K = A@Omega@A
    
    Einv, R = D25(K)
    E = np.linalg.inv(Einv)
    
    eigenvalues, eigenvectors = np.linalg.eig(E)
    
    # Sort eigenvalues and eigenvectors
    #idx = np.argsort(eigenvalues)
    #en_boson = eigenvalues[idx]
    
    Esqrt = sqrtm(E)
    
    D = np.block([[Esqrt, np.zeros_like(idn)],[np.zeros_like(idn), Esqrt]])
    
    U = 1./np.sqrt(2)*np.block([[idn, idn],
                                [-1j*idn, 1j*idn]])
    
    Udaga = 1./np.sqrt(2)*np.block([[idn, 1j*idn],
                                [idn, -1j*idn]])
    
    M = Udaga @ A @ R @ D @ U
    
    return [eigenvalues, M]




boson_spectrum, M = Bogoliubov_Matrix()
A = M[:N//2,:N//2]
B = M[:N//2,N//2:]

Minv = np.linalg.inv(M)

curlyZ = Minv @ curlyW() @ Minv.T

Z1daga = curlyZ[N//2:,N//2:]
Z0 = curlyZ[N//2:,:N//2]

dt = 0.1
times = np.array([k*dt for k in range(10000)])

nkt = pops(20,times,A,B,Z1daga,Z0,boson_spectrum)

plt.figure(1)
plt.plot(times, nkt, '-.')
plt.ylabel(r'$<n_k(t)>$')
plt.xlabel(r'$t$')
plt.grid()
plt.title('Time evolution of the population of the mode k=%1.3f' %K0p[20])
