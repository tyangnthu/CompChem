15/30
Use the knowledge you have learned in week5, week6, and week7 to obtain
1. Run geometry optimizations for phenylnitrene ([N]C1=CC=CC=C1) using the BP86 functional, def2-SVP basis set, and d3 dispersion correction on the broken-symmetry singlet, closed-shell singlet, and triplet spin states. 
* Use !kdiis if you encounter convergence issue.
* To converge to the open-shell singlet, think about orbital restriction or initial guess (moread).
2. Perform single point calculations on top of the corresponding optimized BP86 geometires using the double hybrid density funcitional (DHDF) wB2PLYP functional and def2-SVP basis set.
* Use ! rijcosx autoaux to accelerate DHDF calculations
3. Perform spin purification on the broken-symmetry geometry to obtain the purified energy.
4. Calculate the absorption spectrum for the triplet and broken-symmetry singlet phenylnitrenen using the time-dependent DFT (TD-DFT) with linear response approximation with the wB2PLYP functional and def2-SVP basis set.
* Use tda true in %tddft
* Use sufficient number of roots
* Use ! rijcosx autoaux to accelerate DHDF calculations
5. Plot out the donor and acceptor orbitals of the transition with the highest oscillator strength for the triplet and bs singlet states.
6. Perform TD-DFT calculation for the triplet phenylnitrene with the BP86 functional and def2-SVP basis set.
7. Analyze the output for state with the highest transition probability ABOVE 300 nm (longer wavelength). 
8. Perform excited state optimization on the state with the highest transition probability ABOVE 300 nm (longer wavelength) from the BP86 TD-DFT calculation using the same method.
9. Analyze the output for state with the highest transition probability ABOVE 300 nm (longer wavelength).

Ans:
1. C-N distance for open-shell singlet:, C-N distance for closed-shell singlet:, and C-N distance for triplet:
2. How much energy correction is the spin purification on the broken-symmetry singlet state?
3. From wB2PLYP results, which one is the lowest energy spin state? By how much is the energy gap from the second lowest energy spin state? (watch out for spin purification)
4. Plot the absorption spectrum of the triplet phenylnitrene between 250 to 500 nm using a linewidth of 1500 cm ** -1. 
5. Save the images of the donor and acceptor orbitals as phenylnitrene_0_3_donor.png, phenylnitrene_0_3_acceptor.png, phenylnitrene_0_1bs_donor.png, and phenylnitrene_0_1bs_acceptor.png
6. For state with the highest transition probability ABOVE 300 nm, what is the excited determinant that has the highest CI weight? (e.g. 20a -> 23a)
7. Find out the state with the same dominant CI determinant in the second TD-DFT calculation.
8. What is the number of Stoke shift (in nm)?
