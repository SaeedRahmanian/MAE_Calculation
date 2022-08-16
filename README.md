# MAE Calculation:
This code is designed to find Magneto Crystalline Energy (MAE) using VASP. To do this one need the following inputs in the run directory:
1. POSCAR: it should contain elements symbols
2. POTCAR: for each element separately and name like: POSCAR_<element chemical symbols>. For example POTCAR_Mn or POTCAR_Co or POTCAR_Sn
3. INCAR_univ: You can change all variables other than: NBANDS, MAGMOM, LDAUL, LDAUU, LDAUU which all will be assigned by MAY.py script.
4. Bash script for running VASP named: job_ncl.sh for noncollinear and job_std.sh for collinear runs.

Important parameters in MAE.py need to be changed by user (labeled by #usr).
  
Example <ls> in calculation directory with inputs for W3MnB4:\\
INCAR univ job_ncl.sh job_std.sh MAE.py MAE.sbatch POSCAR POTCAR_B POTCAR_Mn POTCAR_W
