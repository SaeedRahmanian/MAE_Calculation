# MAE Calculation:
This code is designed to find Magneto Crystalline Energy (MAE) using VASP. To do this one need the following inputs in the run directory:
1. POSCAR: it should contain elements symbols
2. POTCAR: for each element separately and name like: POSCAR_<element chemical symbols>. For example POTCAR_Mn or POTCAR_Co or POTCAR_Sn
3. INCAR_univ: You can change all variables other than: NBANDS, MAGMOM, LDAUL, LDAUU, LDAUU which all will be assigned by MAY.py script.
4. Bash script for running VASP named: job_ncl.sh for noncollinear and job_std.sh for collinear runs.

Important parameters in MAE.py need to be changed by user (labeled by #usr).
  
For example "ls" command in the calculation directory with inputs for W3MnB4 MAE calculation:

INCAR_univ job_ncl.sh job_std.sh MAE.py MAE.sbatch POSCAR POTCAR_B POTCAR_Mn POTCAR_W

## Outputs
All outputs will go to the ./outputs folder in run directory.
1. E_vs_K_U*.png: plot showing convergence as a function of number of kpoints per atom
2. EvsTheta_curves_U*.png: Energy as a function of ‘θ’ for different values of ‘φ’
3. MAE_curves_U*.png: MAE curve in plane containing max and min of spin rotation energy
4. MAE_E_U*.npy: MAE curve energies
5. MAE_alpha U*.npy: MAE angles for the ‘MAE E U*.npy’ energies
6. TOTEN_*.npy: energies for each ‘θ’ at selected ‘φ’
7. theta_*.npy: ‘θ’ of each energy in ‘TOTEN *.npy’
8. INCAR: contains both collinear and noncollinear INCARs created by script
9. OutResults.txt: Contain run information and magnetic properties calculated both in VASP output units and SI units.
  
## Other output folders
1. ./z: contain selected KPOINTS and ENCUT VASP non-collinear output file for the spin rotation runs.
2. ./tempelates: Contains files created by the scripts for the beginning of the calculation
