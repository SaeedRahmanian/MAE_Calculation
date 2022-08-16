# MAE Calculation:
This code is designed to find Magneto Crystalline Energy (MAE) using VASP. To do this one need the following inputs in the run directory:
1. POSCAR: it should contain elements symbols
2. POTCAR: for each element separately and name like: POSCAR +\elements symbol".
For example POTCAR Mn or POTCAR Co or POTCAR Sn
3. Put INCAR univ in the running folder
4. Bash script for running VASP named: job ncl.sh and job std.sh for noncollinear and collinear separately
5. Parameters in MAE.py to be changed by user (labeled by #usr):
