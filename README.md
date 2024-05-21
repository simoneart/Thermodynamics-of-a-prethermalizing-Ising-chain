# Thermodynamics of prethermalizing systems

## Very brief intro

This work is based on [Prethermalization from a low-density Holstein-Primakoff expansion](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Prethermalization+from+a+low-density+Holstein-Primakoff+expansion&btnG=) where a transverse-field Ising chain (TFIC) is subject to a quench that changes the strength of the external field and adds a long-range interaction among the spins of the chain. This kind of perturbation gives rise to a prethermal behaviour due to the fact that it keeps intact a number of the original symmetries of the system. \\
The goal of the project is to study the thermodynamics of such a system investigating the contribution of initial coherences to thermodynamic quantitites such as work and entropy production, relying on the fact that the prethermal metastable states are Generalized Gibbs ensembles and thus retain some memory of the initial datum. \\
This is still w.i.p.

## The codes in this repository 

-  In [Ising_chain_diagonlaiztion.py](https://github.com/simoneart/Thermodynamics-of-prethermalizing-systems/blob/main/Ising_chain_diagonalization.py) are implemented the functions for the diagonalization of the TFIC in the **even parity sector** without perturbations and some basic quantities are initialized given the number of sites. In particular the momenta of the quasiparticles and the basis in the Fock representation are produced and the eigenenergies of a certain basis element can be found;
-  In [find_V_elements.py](https://github.com/simoneart/Thermodynamics-of-prethermalizing-systems/blob/main/find_V_elements.py) the functions for finding the matrix elements in the energy eigenbasis of the perturbation potential V are reported;
-  The program [perturbation_theory.py](https://github.com/simoneart/Thermodynamics-of-prethermalizing-systems/blob/main/perturbation_theory.py) is used to implement the perturbative expansion to first order of the energy spectrum and basis;
-  [main.py](https://github.com/simoneart/Thermodynamics-of-prethermalizing-systems/blob/main/main.py) tests how many sites this technique can implement **without** the perturbation. It turns out that the limit is the memory of the machine since it has to store the $2^{(N-1)}$ basis vectors of length $N$. Furthermore, it computes the population of one mode over time (which remains constant as expected even from a vector that is not in the eigenbasis);
-  [testing_plateau.py](https://github.com/simoneart/Thermodynamics-of-prethermalizing-systems/blob/main/testing_plateau.py) is where all the previous functions are used to test whether the prethermal behaviour is present using as initial state a state that is closed to the ground state (still not related to the pre-quench hamiltonian, wip). Up to 12 sites we see nothing;
-  [silva.py](https://github.com/simoneart/Thermodynamics-of-prethermalizing-systems/blob/main/silva.py) is a code in which I follow the procedure described in Appendix D of [Prethermalization from a low-density Holstein-Primakoff expansion](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Prethermalization+from+a+low-density+Holstein-Primakoff+expansion&btnG=) to find the populations over time;
-  [TPS.py](https://github.com/simoneart/Thermodynamics-of-prethermalizing-systems/blob/main/TPS.py) I essentially repeat the diagonalization and perturbative procedure as in the previous codes, but restricting myself to **the totally-paired sector** in order to study an effective Hilbert space of dimension $N_0 = N/2$. Consistent with the exact diagonalization method (see below) but not with the [silva.py](https://github.com/simoneart/Thermodynamics-of-prethermalizing-systems/blob/main/silva.py) method;
-  [brute_force.py](https://github.com/simoneart/Thermodynamics-of-prethermalizing-systems/blob/main/brute_force.py) is the exact diagonalization of the post-quench Hamiltonian, used for validation for small values of sites (it needs  [Ising_chain_diagonlaiztion.py](https://github.com/simoneart/Thermodynamics-of-prethermalizing-systems/blob/main/Ising_chain_diagonalization.py) to run, in particular to fix the numer of spins).

The codes that are most relevant at this point are: [TPS.py](https://github.com/simoneart/Thermodynamics-of-prethermalizing-systems/blob/main/TPS.py), [silva.py](https://github.com/simoneart/Thermodynamics-of-prethermalizing-systems/blob/main/silva.py) and [brute_force.py](https://github.com/simoneart/Thermodynamics-of-prethermalizing-systems/blob/main/brute_force.py). The others can be considered outdated.



