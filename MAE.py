import fileinput
import sys
import subprocess
import numpy as np
import os
import matplotlib.cm as cm
import re
import time
import traceback
from itertools import islice
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.outputs import Outcar
from pymatgen.io.vasp.outputs import Oszicar
from pymatgen.io.vasp.outputs import Eigenval


##Inputs
#POSCAR, POTCARS, INCAR_univ, job_std.sh, job_ncl.sh
kpoints = [0.07, 0.08,  0.09,  0.1, 0.12, 0.15, 0.2]#[0.07,0.08,0.09,0.1, 0.11, 0.12, 0.15, 0.2]
Ncpu =    [   8,    4,     4,    4,    2,    2,   2]#[  12,  12,  12, 12,   12,   12,   12,  12]
#kpoints = [0.055, 0.06, 0.07,0.08,  0.09,  0.1, 0.11, 0.12, 0.15, 0.2]#[0.07,0.08,0.09,0.1, 0.11, 0.12, 0.15, 0.2]
#Ncpu =    [   16,   16,   12,  12,    12,   12,   12,   12,   12,  12]#[  12,  12,  12, 12,   12,   12,   12,  12]
ENCUT = [600]
lnodes = range(50,61)
if np.array(ENCUT).shape[0]>1:
    print("Unsafe Encut, more tha one ENCUT not fixed yet.")
Nph = 20
Nth = 10
Nparalel = 2  #usr
N_MAE = 20
U = 0.0  #usr
J = 0.0
ulimitadd = 'yes'
MagAtom = 'Fe'  #usr
LDAUTYPE  = 2 #1 (Liechtenstein), 2 (Dudarev), 4 (same to 1 but no LSDA)
Bh = 9.274009994*1e-24
mu0 = 4*np.pi*1e-7
eV = 1.602176634*1e-19
Ang = 1e-10
if os.path.isdir('./outputs')==False:
    os.makedirs('./outputs')
#subprocess.call('$( mkdir outputs )', shell=True)
outputs = open("./outputs/OutResults.txt","w")
outputs.write("MAE calculation for (U="+str(U)+", J="+str(J)+") for "+Structure.from_file("./POSCAR").formula+ '\n')
outputs.write('################################################################################'+ '\n')
outputs.write('Job information: '+ '\n')
outputs.write('KPOINTS:        '+str(kpoints)+ '\n')
outputs.write('cpu for each K: '+str(Ncpu)+ '\n')
outputs.write('ENCUTS: '+str(ENCUT)+ '\n')
outputs.write('U-J = '+str(U-J)+ '\n')
outputs.write('Magnetic atom specified by usr: '+ MagAtom+ '\n')
outputs.write('Grid points: theta = '+str(Nth)+', phi = '+str(Nph)+', MAE_curve = '+str(N_MAE+1)+ '\n')
outputs.write('################################################################################')
print('################################################################################')
print('Job information: ')
print('KPOINTS:        ', kpoints)
print('cpu for each K: ', Ncpu)
print('ENCUTS: ', ENCUT)
print('U-J = ', U-J)
print('Magnetic atom specified by usr: ', MagAtom)
print('################################################################################')


def magnetization(fileOutCar):
    Mag = 'Nan'
    with open(fileOutCar) as file:#open("./z/OUTCAR") as file:
        for line in file:
            if "General timing and accounting informations for this job:" in line:  
                with open("./z/OSZICAR") as file:
                    for line in file:
                        if 'mag=' in line:
                            head, sep, tail = line.partition('mag=')
                            Mag = [float(j) for j in re.findall(r"[-+]?\d*\.?\d+|\d+", tail)]
                M0 = np.sqrt((Mag[0]**2)+(Mag[1]**2)+(Mag[2]**2))
                #print('total magnetization:', np.sqrt((Mag[0]**2)+(Mag[1]**2)+(Mag[2]**2)))
    return M0


def readPOSCAR():
    struct = Structure.from_file("./POSCAR")
    myV = struct.volume
    myAtoms_symbols = struct.symbol_set
    mycomp = struct.composition
    mycomp_atoms_num = mycomp.num_atoms
    myesp_num = []
    for atoms in myAtoms_symbols:
        myesp_num.append(mycomp.get_atomic_fraction(atoms)*mycomp_atoms_num)
    return myV, myAtoms_symbols, myesp_num, mycomp_atoms_num



def Ordered_atom_symbol(myAtoms_symbols,  myesp_num):
    with open("./POSCAR") as file:
        for line in islice(file, 2, None):
            if  any(r in line for r in myAtoms_symbols):
                elemline = 1
                #print("POSCAR element line: ", line)
                elems = (re.sub("\s\s+" , " ", line)).split()
            if any(r in line for r in [' '+str(int(_))+' ' for _ in  myesp_num]):
                #print("POSCAR element number line: ", line)
                elems_num = [int(j) for j in re.findall(r"[-+]?\d*\.?\d+|\d+", line)]
    return elems, elems_num


def NBANDS():
    elems_list = readPOSCAR()[1]
    ZVAL_list = []
    for el in elems_list:
        with open("./POTCAR_"+el) as file:
            for line in file:
                if 'ZVAL' in line and 'mass and valenz' in line:
                    head, sep, tail = line.partition('ZVAL')
                    ZVAL_list.append([float(j) for j in re.findall(r"[-+]?\d*\.?\d+|\d+", tail)])
                    #print("ZVAL for: ", el, ZVAL_list[-1])
    comp_atoms_num = readPOSCAR()[2]
    #print("In making NBANDS: ", " Atoms composition: ", comp_atoms_num, ", ZVAL:", np.array(ZVAL_list).flatten())
    NBANDS = int(1.5*sum([(m*n)+m for m,n in zip(comp_atoms_num, np.array(ZVAL_list).flatten())])/2.0)
    return NBANDS, (2*NBANDS) 



def NAGMOM_std(myAtoms_symbol, myesp_num, myMagAtom):
    MAGMOM_str = '  '
    for atoms, atomn in zip(myAtoms_symbol,myesp_num):
        if atoms == MagAtom:
            MAGMOM_str = MAGMOM_str + str(int(atomn))+'*5.0  '
        else:
            MAGMOM_str = MAGMOM_str + ' '+str(int(atomn))+'*2.0  '
    print(">>> std MAGMOM: ", MAGMOM_str)
    return MAGMOM_str


def NAGMOM_ncl(myAtoms_symbol, myesp_num, myMagAtom):
    MAGMOM_str =  '  '
    #print("myAtoms_symbol in NAGMOM_ncl: ", myAtoms_symbol)
    #print("myAtoms_symbol in NAGMOM_ncl: ", myesp_num)
    for atoms, atomn in zip(myAtoms_symbol,myesp_num):
        if atoms == MagAtom:
            for i in range(int(atomn+0.001)):
                MAGMOM_str = MAGMOM_str + '  0 0 5  '
        else:
            for i in range(int(atomn+0.001)):
                MAGMOM_str = MAGMOM_str + '  0 0 2  '
    print(">>> ncl MAGMOM: ", MAGMOM_str)
    return MAGMOM_str



def replaceAll(file,searchExp,replaceExp):
    for line in fileinput.input(file, inplace=1):
        if searchExp in line:
            line = line.replace(line, replaceExp)
        sys.stdout.write(line)



def Addnode(file,searchExp,replaceExp):
    for line in fileinput.input(file, inplace=1):
        if searchExp in line:
            line = line.replace(line, line + replaceExp)
        sys.stdout.write(line)

        

def RotatingAngles(mytheta, myphi):
    x= np.sin(mytheta)*np.cos(myphi)
    y= np.sin(mytheta)*np.sin(myphi)
    z= np.cos(mytheta)
    return x,y,z
        
        
        
        
def LUJ(myU, myJ,myMagAtom, myspecies):
    LLL = ' '
    UUU = ' '
    JJJ = ' '
    #print("myMagAtom: ", myMagAtom)
    #print("myspecies: ", myspecies)
    for atoms in  myspecies:
        if atoms == myMagAtom:
            LLL = LLL + ' 2 '
            UUU = UUU + ' '+str(myU*1.0)+ ' '
            JJJ = JJJ + ' '+str(myJ)+ ' '
        else:
            LLL = LLL + ' -1 '
            UUU = UUU + ' 0 '
            JJJ = JJJ + ' 0 '
    return LLL, UUU, JJJ
        

def Universal_INCAR(myU,myJ, myMagAtom):
    esp_num = readPOSCAR()[2]
    Atoms_symbol, esp_num = Ordered_atom_symbol(readPOSCAR()[1], esp_num)
    command='cp INCAR_univ INCAR;'
    subprocess.call('$('+command+')', shell=True)
    replaceAll("./INCAR", "LDAUTYPE", 'LDAUTYPE = '+ str(LDAUTYPE)+'  \n')
    replaceAll("./INCAR", "*MMM_std*", 'MAGMOM = '+NAGMOM_std(Atoms_symbol, esp_num, myMagAtom)+'  #_std_ \n')
    replaceAll("./INCAR", "*MMM_ncl*", 'MAGMOM = '+NAGMOM_ncl(Atoms_symbol, esp_num, myMagAtom)+'  #_ncl_ \n')
    replaceAll("./INCAR", "*LLL*", 'LDAUL = ' + LUJ(myU,myJ,myMagAtom, Atoms_symbol)[0]+'\n')
    replaceAll("./INCAR", "*UUU*", 'LDAUU = ' + LUJ(myU,myJ,myMagAtom, Atoms_symbol)[1]+'\n')
    replaceAll("./INCAR", "*JJJ*", 'LDAUJ = ' + LUJ(myU,myJ,myMagAtom, Atoms_symbol)[2]+'\n')
    replaceAll("./INCAR", "*NNN_std*", 'NBANDS = ' + str(NBANDS()[0])+'   #_std_ \n')
    replaceAll("./INCAR", "*NNN_ncl*", 'NBANDS = ' + str(NBANDS()[1])+'  #_ncl_ \n')
    return True

def tempelates_files():
    esp_num = readPOSCAR()[2]
    if os.path.isdir('./tempelates') == False:
        os.makedirs('./tempelates')
    command = 'cat '
    for elems in Ordered_atom_symbol(readPOSCAR()[1], esp_num)[0]:
        command+=' POTCAR_'+elems+' '
    command+=' > POTCAR; '
    command+='cp POTCAR  tempelates/;'
    command+='cp POSCAR tempelates/;'
    command+='cp INCAR tempelates/INCAR;'
    command+='cp INCAR outputs/INCAR;'
    command+='cp job_std.sh tempelates/;'
    command+='cp job_ncl.sh tempelates/;'
    print("This command creates tempelates files: \n>>>>  ", command)
    subprocess.call('$('+command+')', shell=True)
    print('"./tempelates" folder was successfully created!')
    return True

def FunMAE(x, a, b, c):
    return a*((np.sin(x))**2)+b*((np.sin(x))**4)+c

def last_200_OUTCAR_line(myfile, myfolder):
    a_file = open(myfile, "r")
    lines = a_file. readlines()
    last_lines = lines[-200:]
    Save_OUTCAR = open(myfolder+"/OUTCAR_lastlines","w")
    for line in last_lines:
        Save_OUTCAR.write(line)
    Save_OUTCAR.close()
    return True


def find_job_ID(myfolder): #Finding ID for the last run in the myfolder
    with open(myfolder+"/log.output") as file:
        for line in file:
            if "SLURM_JOB_ID =" in line:
                myJobID = re.findall(r"[-+]?\d*\.?\d+|\d+", line)
                break
    return myJobID[0]

def find_job_fail(myfolder, myJobID):
    myState = 0
    with open(myfolder+'/JobState_'+str(Job_ID)) as file:
        for line in file:
            if str(myJobID)+' ' in line and 'FAILED' in line:
                myState = 1
                break
    return myState


def Zero_to_one(i):
    if abs(i) < 1e-10:
        return 1.0
    else:
        return i

def NormedCross(myvec1, myvec2):
    return(np.cross(myvec1, myvec2)/Zero_to_one(np.linalg.norm(np.cross(myvec1, myvec2))))

def job_fail_Signal9_11(myfolder, myOutfile):
    failed_ = 0
    if os.path.isfile(myfolder+"/"+myOutfile)==True:
        with open(myfolder+"/"+myOutfile) as file:
            for line in file:
                if "APPLICATION TERMINATED WITH THE EXIT STRING:" in line:
                    print("Job failed with message:  "+line)
                    failed_ = 1
                    break
                if "num prob" in line:# and "RtMAE" in myfolder:
                    print("\"num prob\" found in log file")
                    with open(myfolder+"/OUTCAR") as file_OUTCAR:
                        for line_OUTCAR in file_OUTCAR:
                            if "General timing and accounting informations for this job:" in line_OUTCAR:
                                print("Job is possibly done but has \"num prob\" error  ")
                                #subprocess.call('$(rm '+myfolder+'/*_cmpl; rm '+myfolder+'/OUTCAR ')', shell=True)
                                failed_ = 1
                                break
        with open(myfolder+"/"+myOutfile) as file:
            for line in file:
                if "Job resubmitted again for -->"+folder in line:
                    print("Job was already resubmitted for:  "+myfolder)
                    failed_ = 0
                    break
    if failed_ == 1:
        with open(folder+"/"+myOutfile, "a") as myfile:
            myfile.write("\nJob resubmitted again for -->"+folder)
    return failed_
                



INCAR_created = Universal_INCAR(U,J,MagAtom)
print("INCAR created: ", INCAR_created, "\n---------------------------------------------------------------")

#Universal_INCAR(U,J, MagAtom)

tempelates_created = tempelates_files()
print("./tempelates created: ", tempelates_created, "\n---------------------------------------------------------------")



ln=0
StillRun = []
for En in ENCUT:
    for Nc, K in zip(Ncpu,kpoints):
        searchExp = 'KSPACING'
        replaceExp = 'KSPACING =  '+str(np.round(K,4)) + '\n'
        folder = 'K_'+str(K)+'_EN_'+str(En)
        #print(folder+'/OUTCAR_std')
        if os.path.isfile(folder+'/OUTCAR_std') == True:
            JobStat = 0
            StillRun.append(JobStat)
            print("job for std K=",np.round(K,4), ", ENCUT=", np.round(En)," has already completed!")
            continue
        JobStat = 1
        StillRun.append(JobStat)
        if os.path.isdir(folder) == False:
            os.makedirs(folder)
        command='cp tempelates/POTCAR '+folder+'/POTCAR ;'
        command+='cp tempelates/POSCAR '+folder+'/POSCAR ; '
        command+='cp tempelates/job_ncl.sh '+folder+'/job_ncl.sh ; '
        command+='cp tempelates/job_std.sh '+folder+'/job_std.sh ; '
        command+='cp tempelates/INCAR '+folder+'/INCAR_ncl ; '
        command+='cp tempelates/INCAR '+folder+'/INCAR_std ; '
        subprocess.call('$('+command+')', shell=True)
        print("Folder "+folder+" created!")
        replaceAll("./"+folder+"/INCAR_std",searchExp,replaceExp)
        replaceAll("./"+folder+"/INCAR_std", 'ENCUT', 'ENCUT = '+str(En)+'\n')
        replaceAll("./"+folder+"/INCAR_ncl", 'ENCUT', 'ENCUT = '+str(En)+'\n')
        replaceAll("./"+folder+"/INCAR_ncl", '_std_', '')
        replaceAll("./"+folder+"/INCAR_std", '_ncl_', '')
        if K < 0.15:
            Addnode("./"+folder+"/job_ncl.sh", '#SBATCH -e log.error', '\n'+'ulimit -s unlimited'+'\n')
            Addnode("./"+folder+"/job_std.sh", '#SBATCH -e log.error', '\n'+'ulimit -s unlimited'+'\n')
        #if K < 0.04:
        #    Addnode("./"+folder+"/job_std.sh", 'SBATCH -n', '#SBATCH -w lnode'+str(np.round(lnodes[ln],3))+'\n')
        #    Addnode("./"+folder+"/job_ncl.sh", 'SBATCH -n', '#SBATCH -w lnode'+str(np.round(lnodes[ln],3))+'\n')
        #    ln+=1
        replaceAll("./"+folder+"/job_std.sh", 'SBATCH -J', '#SBATCH -J '+'stdK_'+str(np.round(K,4))+'\n')
        replaceAll("./"+folder+"/job_ncl.sh", 'SBATCH -J', '#SBATCH -J '+'nclK_'+str(np.round(K,4))+'\n')
        replaceAll("./"+folder+"/job_std.sh", 'SBATCH -n', '#SBATCH -n '+str(Nc)+'\n')
        replaceAll("./"+folder+"/job_ncl.sh", 'SBATCH -n', '#SBATCH -n '+str(Nc)+'\n')
        command='cp ./'+folder+'/INCAR_std '+folder+'/INCAR ; '
        command+='cd ./'+folder+' ;'
        command+='sbatch --reservation=srahmanian_22  job_std.sh; '
        command+= 'cd ../'
        subprocess.call('$('+command+')', shell=True)
        job_fail_Signal9_11(folder, 'log_std')
        print('Job for K=',np.round(K,4),' std submited!')


checkT = 20
t=0
while max(StillRun)>0:
    StillRun = []#np.zeros(np.array(phi).shape[0]*np.array(theta).shape[0])
    print("Job still runing, will check in 300 seconds.")
    time.sleep(checkT)
    for En in ENCUT:
        for K in kpoints:
            folder = 'K_'+str(K)+'_EN_'+str(En)
            FailState = job_fail_Signal9_11(folder, 'log_std')
            #if os.path.isfile(folder+"/log.output")==True:
            #    Job_ID = find_job_ID(folder)
            #    subprocess.call('$(cd ./'+folder+'; sacct -j '+str(Job_ID)+' > JobState_'+str(Job_ID)+'; cd ../)', shell=True)
            #    print('Command:  cd ./'+folder+'; sacct -j '+str(Job_ID)+' > JobState_'+str(Job_ID)+'; cd ../')
            #    FailState = find_job_fail(folder, Job_ID)
            if FailState==1:
                print('Job for '+folder+' was failed with error! Job resubmitted but check the log files for details')
                command='cd ./'+folder+' ;'
                command+='sbatch  --reservation=srahmanian_22  job_std.sh; '
                command+= 'cd ../'
                subprocess.call('$('+command+')', shell=True)
                continue
            print('Job for '+folder+' was NOT failed!')
            if os.path.isfile(folder+'/OUTCAR') == True:
                print(folder+'/OUTCAR exists!')
                with open("./"+folder+"/OUTCAR") as file:
                    JobStat = 1
                    for line in file:
                        if "General timing and accounting informations for this job:" in line:
                            command='cp ./'+folder+'/OUTCAR ./'+folder+'/OUTCAR_std ;'
                            command+='cp ./'+folder+'/OSZICAR ./'+folder+'/OSZICAR_std '
                            subprocess.call('$('+command+')', shell=True)
                            JobStat = 0
                    StillRun.append(JobStat)
            if os.path.isfile(folder+'/OUTCAR') == False and os.path.isfile(folder+'/OUTCAR_std') == False:
                print(folder+'/OUTCAR doesn\'t exists!')
                StillRun.append(1)
    t = t + checkT
    if t%300==0:
        print("StillRun:", StillRun)
subprocess.call('$( rm K_0*EN*/OUTCAR  )', shell=True)
print("***************** std job completed************************")


StillRun = []
for En in ENCUT:
    for K in kpoints:
        folder = 'K_'+str(K)+'_EN_'+str(En)
        if os.path.isfile(folder+'/OUTCAR_ncl') == True:
            JobStat = 0
            StillRun.append(JobStat)
            print("job for ncl K=",np.round(K,4), ", ENCUT=", np.round(En)," has already completed!")
            continue
        JobStat = 1
        StillRun.append(JobStat)
        if os.path.isdir(folder) == False:
            print("Folder ", folder, " doesn't exist!")
            continue
        command='cp ./'+folder+'/INCAR_ncl '+folder+'/INCAR ; '
        command+='cp ./'+folder+'/CONTCAR '+folder+'/POSCAR ; '
        command+='cp ./'+folder+'/IBZKPT '+folder+'/KPOINTS ; '
        command+='cd ./'+folder+' ;'
        command+='sbatch  --reservation=srahmanian_22  job_ncl.sh; '
        command+= 'cd ../'
        subprocess.call('$('+command+')', shell=True)
        job_fail_Signal9_11(folder, 'log_ncl')
        print('Job for K=',np.round(K,4),' ncl submited!')

checkT = 20
t=0
while max(StillRun)>0:
    StillRun = []#np.zeros(np.array(phi).shape[0]*np.array(theta).shape[0])
    print("Job still runing, will check in 300 seconds.")
    time.sleep(checkT)
    for En in ENCUT:
        for K in kpoints:
            folder = 'K_'+str(K)+'_EN_'+str(En)
            FailState = job_fail_Signal9_11(folder, 'log_ncl')
            #if os.path.isfile(folder+"/log.output")==True:
            #    Job_ID = find_job_ID(folder)
            #    subprocess.call('$(cd ./'+folder+'; sacct -j '+str(Job_ID)+' > JobState_'+str(Job_ID)+'; cd ../)', shell=True)
            #    FailState = find_job_fail(folder, Job_ID)
            if FailState==1:
                print('Job for '+folder+' was failed with error! Job resubmitted but check the log files for details')
                command='cd ./'+folder+' ;'
                command+='sbatch  --reservation=srahmanian_22  job_ncl.sh; '
                command+= 'cd ../'
                subprocess.call('$('+command+')', shell=True)
                continue
            print('Job for '+folder+' was NOT failed!')
            if os.path.isfile(folder+'/OUTCAR') == True:
                print(folder+'/OUTCAR exists!')
                with open("./"+folder+"/OUTCAR") as file:
                    JobStat = 1
                    for line in file:
                        if "General timing and accounting informations for this job:" in line:
                            command='cp ./'+folder+'/OUTCAR ./'+folder+'/OUTCAR_ncl ;'
                            command+='cp ./'+folder+'/OSZICAR ./'+folder+'/OSZICAR_ncl '
                            subprocess.call('$('+command+')', shell=True)
                            JobStat = 0
                    StillRun.append(JobStat)
            if os.path.isfile(folder+'/OUTCAR') == False and os.path.isfile(folder+'/OUTCAR_ncl') == False:
                print(folder+'/OUTCAR doesn\'t exists!')
                StillRun.append(1)
    t = t + checkT
    if t%300==0:
        print("StillRun:", StillRun)
print("***************** ncl job completed************************")



for En in ENCUT:
    E0 = []
    Kp_read = []
    nkpt_read = []
    for K in kpoints:
        folder = 'K_'+str(K)+'_EN_'+str(En)
        nkpt_read.append(Eigenval("./"+folder+"/EIGENVAL").nkpt)
        Kp_read.append(K)
        E0.append(float((Oszicar("./"+folder+"/OSZICAR").all_energies[-1])[-2]))


fig = plt.figure()
fig.suptitle('Enthalpy/atom as a function of number of K-points for U='+str(U)+', J='+str(J))
#plt.rcParams['axes.grid'] = True
plt.grid()
plt.xlabel(r'(number of Kpoints)/atom', size=15)
plt.ylabel(r'(E-E$_{Kmin}$) $(MJ/m^3)$', size=15)
#plt.plot(np.array(alpha_read)*180.0/np.pi, (np.array(E_alpha)-E_ref)*((10.0**-6.0)/readPOSCAR()[0])*(eV/(Ang**3))*(1.0/readPOSCAR()[3]), '-o')
#plt.plot(np.array(nkpt_read)/readPOSCAR()[3], (np.array(E0)-E0[0])*1000/readPOSCAR()[3], '-o',label = "U= "+str(U)+", J="+str(J))
plt.plot(np.array(nkpt_read)/readPOSCAR()[3], (np.array(E0)-E0[0])*((10.0**-6.0)/readPOSCAR()[0])*(eV/(Ang**3))*(1.0/readPOSCAR()[3]), '-o',label = "U= "+str(U)+", J="+str(J))
plt.legend(prop={'size': 15}, loc='upper right')
plt.savefig("./outputs/E_vs_K_U"+str(U)+"_J"+str(J)+".png")
np.save("./outputs/E_of_Kpoints"+str(U)+"_J"+str(J)+".npy", np.array(nkpt_read))
np.save("./outputs/Number_Kpoints"+str(U)+"_J"+str(J)+".npy", np.array(E0))

        
K_select = 0
Encut_select = ENCUT[-1]
E0_SI = (np.array(E0)-E0[0])*((10.0**-6.0)/readPOSCAR()[0])*(eV/(Ang**3))*(1.0/readPOSCAR()[3])
K_select_list = [y for x,y in zip(abs(E0_SI), Kp_read) if x<0.12 and y <0.11 and y>0.05]
K_select_list_E = [x for x,y in zip(abs(E0_SI), Kp_read) if x<0.12 and y <0.11 and y>0.05]
K_select = max(K_select_list)
K_select_MAE_curve = min(K_select_list)
#for i in range(np.array(E0).shape[0]):
#    if i < (np.array(E0).shape[0]-2):
#        if abs(E0_SI[-i-1]-E0_SI[-i-2])/readPOSCAR()[3]<0.12 and abs(E0_SI[-i-1]-E0_SI[-i-3])/readPOSCAR()[3]<0.12:
#            K_select = Kp_read[-i-1]
#            break
#if K_select == 0:
#    if abs(E0_SI[1]-E0_SI[0])/readPOSCAR()[3]<0.12:
#        K_select = Kp_read[1]
#    else:
#        K_select = Kp_read[0]


folder = 'K_'+str(K_select)+'_EN_'+str(Encut_select)
folder_MAE_curve = 'K_'+str(K_select_MAE_curve)+'_EN_'+str(Encut_select)
outputs.write('\n\nSelected Kpoint fro MAE curve:  '+str(K_select_MAE_curve) + '\n')
outputs.write('Selected Kpoints:                   '+str(K_select_list) + '\n')
outputs.write('Selected Kpoint energies in SI:     '+str(K_select_list_E) + '\n')
outputs.write('Selected ENCUT:                     '+str(Encut_select) + '\n')

if os.path.isfile('./K_'+str(K_select)+'_EN_'+str(Encut_select)+'/WAVECAR')==False and os.path.isfile('./z/WAVECAR')==False:
    sys.exit("Selected Kpoints and ENCUT ./z/WAVECAR were removed, start from the begining!!")
if os.path.isfile('./K_'+str(K_select)+'_EN_'+str(Encut_select)+'/WAVECAR')==True and os.path.isfile('./z/WAVECAR')==False:
    subprocess.call('$('+'cp -r '+folder+' z'+'; cp -r '+folder_MAE_curve+' z_MAE_curve'+')', shell=True)
    print('z and z_MAE_curve folders were created! for K=', K_select, K_select_MAE_curve,' and ENCUT=', Encut_select,  ' \n\n\n')
if os.path.isfile('./z/WAVECAR')==True:
    print('z folder already exists!!!')
    #command='rm  ./K_*/WAVECAR;'
    #command+='rm  ./K_*/CHGCAR; '
    #command+='rm  ./K_*/vasprun.xml; '
    #subprocess.call('$('+command+')', shell=True)
    print("large files in K_* was removed!!")


phi = np.linspace(0.0001, 2 * np.pi-0.0001, Nph+1)
theta = np.linspace(0+0.0001, np.pi-0.00001, Nth+1)
THETA, PHI = np.meshgrid(theta, phi)
Nrunning = 0
phi_run = []
theta_run = []
for phin, thetan in zip(PHI.flatten(),THETA.flatten()):
    folder = 'PhTh_'+str(np.round((phin/np.pi)*180,2))+'_'+str(np.round((thetan/np.pi)*180,2))
    Jobe_done = 1
    if os.path.isfile(folder+'/OUTCAR_cmpl') == True:
        with open("./"+folder+"/OUTCAR_cmpl") as file:
            for line in file:
                if "General timing and accounting informations for this job:" in line:
                    print("Job for ",folder," was already completed!")
                    #Jobe_done = 0
                    Jobe_done = job_fail_Signal9_11(folder, 'log_ncl')
    if Jobe_done == 0:
        continue
    print("Nrunning: " , Nrunning)
    if Nparalel>= Nrunning:
        X,Y,Z = RotatingAngles(thetan, phin)
        searchExp = 'SAXIS'
        replaceExp = 'SAXIS =  '+str(np.round(X,3))+ '  '+str(np.round(Y,3))+'  '+str(np.round(Z,3))
        if os.path.isdir(folder) == False:
            os.makedirs(folder)
        command='cp z/POTCAR '+folder+'/POTCAR ;'
        command+='cp z/POSCAR '+folder+'/POSCAR ; '
        command+='cp z/job_ncl.sh '+folder+'/job_ncl.sh ; '
        command+='cp z/CHGCAR '+folder+'/CHGCAR ; '
        #command+='cp z/WAVECAR '+folder+'/WAVECAR ; '
        command+='cd ./'+folder+' ;'
        command+='ln -s  ../z/WAVECAR ./WAVECAR ; '
        command+='cd ../ ;'
        command+='cp z/INCAR '+folder+'/INCAR ; '
        command+='cp z/KPOINTS '+folder+'/KPOINTS ; '
        subprocess.call('$('+command+')', shell=True)
        print("Folder "+folder+" created!")
        replaceAll("./"+folder+"/INCAR",searchExp,replaceExp)
        Addnode("./"+folder+"/INCAR", searchExp,'\n'+'LWAVE = .FALSE.')
        Addnode("./"+folder+"/INCAR", searchExp, '\n'+'LCHARG = .FALSE.')
        replaceAll("./"+folder+"/job_ncl.sh", 'SBATCH -J', '#SBATCH -J '+'nclRt_'+str(np.round((phin/np.pi)*180,1))+'_'+str(np.round((thetan/np.pi)*180,1))+'\n')
        command='cd ./'+folder+' ;'
        command+='sbatch  --reservation=srahmanian_22   job_ncl.sh; '
        command+= 'cd ../'
        #print("Submit command: ", command)
        phi_run.append(phin)
        theta_run.append(thetan)
        subprocess.call('$('+command+')', shell=True)
        job_fail_Signal9_11(folder, 'log_ncl')
        print('Job for Phi=', np.round((phin/np.pi)*180,2), ", theta=" , np.round((thetan/np.pi)*180,2),' submited!')
        Nrunning = Nrunning + 1
    if Nrunning == Nparalel:
        checkT = 60
        #StillRun = [1]
        while Nrunning == Nparalel:#max(StillRun)>0:
            #StillRun = []#np.zeros(np.array(phi).shape[0]*np.array(theta).shape[0])
            #print("Job still runing, will check in ", checkT, " seconds.")
            time.sleep(checkT)
            for phic, thetac in zip(phi_run, theta_run):
                folder = 'PhTh_'+str(np.round((phic/np.pi)*180,2))+'_'+str(np.round((thetac/np.pi)*180,2))
                ############
                FailState = job_fail_Signal9_11(folder, 'log_ncl')
                #if os.path.isfile(folder+"/log.output")==True:
                #    Job_ID = find_job_ID(folder)
                #    subprocess.call('$(cd ./'+folder+'; sacct -j '+str(Job_ID)+' > JobState_'+str(Job_ID)+'; cd ../)', shell=True)
                #    FailState = find_job_fail(folder, Job_ID)
                if FailState==1:
                    print('Job for '+folder+' was failed with error! Job resubmitted but check the log files for details')
                    command='cd ./'+folder+' ;'
                    command+='sbatch  --reservation=srahmanian_22   job_ncl.sh; '
                    command+= 'cd ../'
                    subprocess.call('$('+command+')', shell=True)
                    continue
                print('Job for '+folder+' was NOT failed!')
                ############
                if os.path.isfile(folder+'/OUTCAR') == True:
                    #print(folder+'/OUTCAR exists!')
                    with open("./"+folder+"/OUTCAR") as file:
                        JobStat = 1
                        for line in file:
                            if "General timing and accounting informations for this job:" in line:
                                last_200_OUTCAR_line('./'+folder+'/OUTCAR', './'+folder)
                                Nrunning = Nrunning - 1
                                while  os.path.isfile(folder+'/CHGCAR') == False or os.path.isfile(folder+'/vasprun.xml') == False:
                                    time.sleep(5)
                                    if os.path.isfile(folder+'/OSZICAR_cmpl') == True:
                                        break
                                if os.path.isfile(folder+'/OSZICAR_cmpl') == False:
                                    command='rm  ./'+folder+'/WAVECAR;'
                                    command+='rm  ./'+folder+'/CHGCAR; '
                                    command+='rm  ./'+folder+'/vasprun.xml; '
                                    command+='rm  ./'+folder+'/PROCAR; '
                                    command+='cp ./'+folder+'/OUTCAR ./'+folder+'/OUTCAR_cmpl ;'
                                    command+='cp ./'+folder+'/OSZICAR ./'+folder+'/OSZICAR_cmpl ' 
                                    #print("remove command:   ", command)
                                    subprocess.call('$('+command+')', shell=True)
                                print(">>> job for phi = ", np.round((phic/np.pi)*180,2), ", theta=" , np.round((thetac/np.pi)*180,2) ,"completed!")
                                phi_run.remove(phic)
                                theta_run.remove(thetac)
                                JobStat = 0
                                #StillRun.append(JobStat)
                        if JobStat == 1:
                            print(">>> job for phi = ", np.round((phic/np.pi)*180,2), ", theta=" , np.round((thetac/np.pi)*180,2)," still runnig!")
                if os.path.isfile(folder+'/OUTCAR') == False:
                    if os.path.isfile(folder+'/OSZICAR_cmpl') == True:
                        Nrunning = Nrunning - 1
                        break
                    print(folder+'/OUTCAR doesn\'t exists!')
                    #StillRun.append(1)
            #print("StillRun:", StillRun)

            

            
checkT = 1
t = 0
StillRun = [1]
while max(StillRun)>0 and Nrunning>0:
    StillRun = []#np.zeros(np.array(phi).shape[0]*np.array(theta).shape[0])
    print("Job still runing, will check in ", checkT, " seconds.")
    for phin in phi:
        for thetan in theta:
            folder = 'PhTh_'+str(np.round((phin/np.pi)*180,2))+'_'+str(np.round((thetan/np.pi)*180,2))
            FailState = job_fail_Signal9_11(folder, 'log_ncl')
            #if os.path.isfile(folder+"/log.output")==True:
            #    Job_ID = find_job_ID(folder)
            #    subprocess.call('$(cd ./'+folder+'; sacct -j '+str(Job_ID)+' > JobState_'+str(Job_ID)+'; cd ../)', shell=True)
            #    FailState = find_job_fail(folder, Job_ID)
            if FailState==1:
                print('Job for '+folder+' was failed with error! Job resubmitted but check the log files for details')
                command='cd ./'+folder+' ;'
                command+='sbatch  --reservation=srahmanian_22   job_ncl.sh; '
                command+= 'cd ../'
                StillRun.append(1)
                subprocess.call('$('+command+')', shell=True)
                continue
            print('Job for '+folder+' was NOT failed!')
            if os.path.isfile(folder+'/OUTCAR') == True:
                #print(folder+'/OUTCAR exists!')
                with open("./"+folder+"/OUTCAR") as file:
                    JobStat = 1
                    for line in file:
                        if "General timing and accounting informations for this job:" in line:
                            JobStat = 0
                            last_200_OUTCAR_line('./'+folder+'/OUTCAR', './'+folder)
                            while  os.path.isfile(folder+'/CHGCAR') == False or os.path.isfile(folder+'/vasprun.xml') == False:
                                time.sleep(1)
                                if os.path.isfile(folder+'/OSZICAR_cmpl') == True:
                                        break
                            if os.path.isfile(folder+'/OSZICAR_cmpl') == False:
                                command='rm  ./'+folder+'/WAVECAR;'
                                command+='rm  ./'+folder+'/CHGCAR; '
                                command+='rm  ./'+folder+'/PROCAR; '
                                command+='rm  ./'+folder+'/vasprun.xml; '
                                command+='cp ./'+folder+'/OUTCAR ./'+folder+'/OUTCAR_cmpl ;'
                                command+='cp ./'+folder+'/OSZICAR ./'+folder+'/OSZICAR_cmpl '
                                subprocess.call('$('+command+')', shell=True)
                            print(">>> job for phi = ", np.round((phin/np.pi)*180,2), ", theta=" , np.round((thetan/np.pi)*180,2) ,"completed!")
                    if JobStat == 1:
                        print(">>> job for phi = ", np.round((phin/np.pi)*180,2), ", theta=" , np.round((thetan/np.pi)*180,2)," still runnig!")
                    StillRun.append(JobStat)
            if os.path.isfile(folder+'/OUTCAR') == False:
                if os.path.isfile(folder+'/OSZICAR_cmpl') == True:
                    Nrunning = Nrunning - 1
                    break
                print(folder+'/OUTCAR doesn\'t exists!')
                StillRun.append(1)
    #t = t + checkT
    #if t%300==0:
    print("max of StillRun: ",  max(StillRun))
    time.sleep(checkT)
print("*****************job for all spin rotations completed************************")



command='rm ./PhTh_*/vasprun.xml;'
command+='rm ./PhTh_*/CHGCAR '
subprocess.call('$('+command+')', shell=True)






fig = plt.figure()
fig.suptitle('Enthalpy as a function of $\theta$ for defferent $\phi$ for U='+str(U)+', J='+str(J))
plt.rcParams['axes.grid'] = True
plt.xlabel('$\theta$', size=15)
plt.ylabel('Enthapy $(MJ/m^3)$', size=15)
all_phi = []
all_theta = []
all_E = []
colors = [cm.rainbow(i) for i in np.linspace(0, 1, Nph+1)]
E_ref = float((Oszicar("./z/OSZICAR").all_energies[-1])[-2])
for phin, c in zip(phi, colors):
    TOTEN = []
    theta_read = []
    for thetan in theta:
        folder = 'PhTh_'+str(np.round((phin/np.pi)*180,2))+'_'+str(np.round((thetan/np.pi)*180,2))
        print('Folder is:'+folder)
        if os.path.isdir("./"+folder) == False:
            print(folder, ' doesn\'t exist')
            continue
        elif os.path.isdir("./"+folder) == True:
            print("folder ", folder," was found!")
            with open("./"+folder+"/OUTCAR_cmpl") as file:
                FinCond = 0
                for line in file:
                    if "General timing and accounting informations for this job:" in line:
                        FinCond = 1
                        #print('energies: ', Oszicar("./"+folder+"/OSZICAR").all_energies)
                        TOTEN.append(float((Oszicar("./"+folder+"/OSZICAR").all_energies[-1])[-2]))
                        theta_read.append(thetan)
                        all_theta.append(thetan)
                        all_phi.append(phin)
                        all_E.append(float((Oszicar("./"+folder+"/OSZICAR").all_energies[-1])[-2]))
                        print('Energy for Phi='+str(np.round(phin,2))+' and theta='+str(np.round(thetan,2))+'  '+str(TOTEN[-1]))
                if FinCond == 0:
                    all_E.append(np.log(-2)) #to ignore in plotting
                    print('job for Phi='+str(np.round(phin,2))+' and theta='+str(np.round(thetan,2))+' is not finished!')
    #print('TOTEN: ', TOTEN)
    plt.plot(np.array(theta_read)*180.0/np.pi, (np.array(TOTEN)-E_ref)*((10.0**-6.0)/readPOSCAR()[0])*(eV/(Ang**3))*(1.0/readPOSCAR()[3]), '-o',color=c, label='$\phi$='+str(np.round(phin,3)))
    np.save("./outputs/TOTEN_"+str(np.round(phin,3))+".npy", (np.array(TOTEN)-E_ref)*((10.0**-6.0)/readPOSCAR()[0])*(eV/(Ang**3))*(1.0/readPOSCAR()[3]))
    np.save("./outputs/theta_"+str(np.round(phin,3))+".npy", np.array(theta_read)*180.0/np.pi)
    #plt.plot(theta, TOTEN)
plt.legend(prop={'size': 8})
plt.savefig("./outputs/EvsTheta_curves_U"+str(U)+"_J"+str(J)+".png")




MAE = (max(all_E)-min(all_E))
K1 = (max(all_E)-min(all_E))

for e1, p1,t1 in zip(all_E, all_phi, all_theta):
    if e1==min(all_E):
        phi_min = p1; theta_min = t1
    if e1 == max(all_E):
        phi_max = p1; theta_max = t1
outputs.write('\n\n\nphi_min: '+str(np.round(phi_min*180/np.pi,2))+',   theta_min: '+str(np.round(theta_min*180/np.pi,2))+'\n')
outputs.write('phi_max: '+str(np.round(phi_max*180/np.pi,2))+',   theta_max: '+str(np.round(theta_max*180/np.pi,2))+'\n')
print('phi_min: ', np.round(phi_min*180/np.pi,2), 'theta_min: ', np.round(theta_min*180/np.pi,2))
print('phi_max: ', np.round(phi_max*180/np.pi,2), 'theta_max: ', np.round(theta_max*180/np.pi,2))
min_vec = [np.sin(theta_min) * np.cos(phi_min), np.sin(theta_min) * np.sin(phi_min), np.cos(theta_min)]
max_vec = [np.sin(theta_max) * np.cos(phi_max), np.sin(theta_max) * np.sin(phi_max), np.cos(theta_max)]
print('max_vec: ', np.round(max_vec,2))
print('min_vec: ', np.round(min_vec,2))
outputs.write('max_vec: '+str(list(np.round(max_vec,2)))+'\n')
outputs.write('min_vec: '+str(list(np.round(min_vec,2)))+'\n')

MAE_x = np.array(min_vec[:])
if max(abs(np.array(np.cross(min_vec, max_vec))))>0.0001:
    MAE_z = np.array(NormedCross(min_vec, max_vec))
else:
    if os.path.isfile('./outputs/MAE_Ran_vec.npy') == False:
        Ran_vec = np.random.rand(3)
        Ran_vec = Ran_vec/np.linalg.norm(Ran_vec)
        MAE_z = np.array(np.cross(min_vec, Ran_vec))/np.linalg.norm(np.cross(min_vec, Ran_vec))
        np.save('./outputs/MAE_Ran_vec.npy', MAE_z)
    if os.path.isfile('./outputs/MAE_Ran_vec.npy') == True:
        MAE_z = np.load('./outputs/MAE_Ran_vec.npy')
MAE_y = np.array(np.cross(MAE_z, MAE_x))/np.linalg.norm(np.cross(MAE_z, MAE_x))
outputs.write('MAE_x: '+str(list(np.round(MAE_x,3)))+', norm='+str(np.round(np.linalg.norm(MAE_x)))+'\n')
outputs.write('MAE_z: '+str(list(np.round(MAE_z,3)))+', norm='+str(np.round(np.linalg.norm(MAE_y)))+'\n')
outputs.write('MAE_y: '+str(list(np.round(MAE_y,3)))+', norm='+str(np.round(np.linalg.norm(MAE_z)))+'\n')
print('MAE_x: ', np.round(MAE_x,3), ', norm', np.round(np.linalg.norm(MAE_x)))
print('MAE_z: ', np.round(MAE_z,3), ', norm', np.round(np.linalg.norm(MAE_y)))
print('MAE_y: ', np.round(MAE_y,3), ', norm', np.round(np.linalg.norm(MAE_z)))


fig = plt.figure()
fig.suptitle('MAE curve (Enthalpy) for U='+str(U)+', J='+str(J)+'. Atoms number = '+str(readPOSCAR()[3])+', Volume = '+str(np.round(readPOSCAR()[0],1))+' $A^3$ ', fontsize=10)
plt.rcParams['axes.grid'] = True
plt.xlabel('$\\alpha$', size=15)
plt.ylabel('Enthalpy $(MJ/m^3)$', size=15)
MAE_vs_K = []
for K in kpoints:
    print("Finding MAE curve for K=", K,"\n")
    outputs.write('########################################################\n')
    outputs.write('Kpoint = '+str(np.round(K,4))+'\n')
    outputs.write('########################################################\n')
    folder_MAE_curve = 'K_'+str(K)+'_EN_'+str(Encut_select)
    subprocess.call('$('+'rm -r z_MAE_curve; cp -r '+folder_MAE_curve+' z_MAE_curve'+')', shell=True)
    if os.path.isfile('./'+folder_MAE_curve+'/WAVECAR')==False and os.path.isdir(folder_MAE_curve+'_RtMAE_IsDone') == False:
        print("WAVECAR for K=", K, " doesn't exists!! and this job is not completed")
        continue
    Nrunning = 0
    alpha_run = []
    #Nparalel = int(0.5*Nparalel+1)
    E_ref_MAE = float((Oszicar("./z_MAE_curve/OSZICAR").all_energies[-1])[-2])
    for alpha in  np.linspace(0, 2*np.pi, N_MAE+1):
        folder = 'K_'+str(K)+'_RtMAE_'+str(np.round((alpha/np.pi)*180,2))
        SAX = np.sin(alpha)*MAE_y+np.cos(alpha)*MAE_x
        searchExp = 'SAXIS'
        print('alpha: ', np.round(alpha*180/np.pi,3), ', SAXIS: ', np.round(SAX,3))
        if os.path.isfile(folder+'/OSZICAR_cmpl') == True:
            print("job for alpha = ", str(np.round((alpha/np.pi)*180,2)) ," was already completed!")
            continue
        if Nparalel>= Nrunning:
            print('np.linalg.norm(SAX): ', np.round(np.linalg.norm(SAX), 10))
            print('(MAE_x*SAX).MAE_z: ', np.round(Zero_to_one(np.dot(NormedCross(MAE_x, SAX),MAE_z)),10), ", sing(sin(a)):    ",np.sign(Zero_to_one(np.sin(alpha))))
            print('(SAX*MAE_y).MAE_z: ', np.round(Zero_to_one(np.dot(NormedCross(SAX, MAE_y),MAE_z)),10), ", sing(sin(pi-a)): ",np.sign(Zero_to_one(np.sin(np.pi/2-alpha))))
            while True:
                if np.round(np.linalg.norm(SAX),10) == 1.0 and np.round(Zero_to_one(np.dot(NormedCross(MAE_x, SAX),MAE_z))*np.sign(Zero_to_one(np.sin(alpha))),10)==1.0 and np.round(Zero_to_one(np.dot(NormedCross(SAX, MAE_y),MAE_z))*np.sign(Zero_to_one(np.sin(np.pi/2-alpha))),10)==1.0:
                    break
                elif np.round(np.linalg.norm(SAX),10) != 1.0:
                    print('norm SAXIS: ', np.round(np.linalg.norm(SAX),10))
                    sys.exit("Oops! SAXIS not normalized")
                elif np.round(Zero_to_one(np.dot(NormedCross(MAE_x, SAX),MAE_z))*np.sign(Zero_to_one(np.sin(alpha))),10)!=1.0:
                    print("((X x SAX).Z)*sing(sin(a)): ", np.round(Zero_to_one(np.dot(NormedCross(MAE_x, SAX),MAE_z))*np.sign(Zero_to_one(np.sin(alpha))),10))
                    sys.exit("Oops! np.cross(MAE_x, SAX) is not correct")
                elif np.round(Zero_to_one(np.dot(NormedCross(SAX, MAE_y),MAE_z))*np.sign(Zero_to_one(np.sin(np.pi/2-alpha))),10)!=1.0:
                    print("((SAX x Y).Z)*sing(sin(pi-a)): ", np.round(Zero_to_one(np.dot(NormedCross(SAX, MAE_y),MAE_z))*np.sign(Zero_to_one(np.sin(np.pi/2-alpha))),10))
                    sys.exit("Oops! np.cross(SAX, MAE_y) is not correct")
            replaceExp = 'SAXIS =  '+str(np.round(SAX[0],3))+ '  '+str(np.round(SAX[1],3))+'  '+str(np.round(SAX[2],3))
            if os.path.isdir(folder) == False:
                os.makedirs(folder)
            command='cp z_MAE_curve/POTCAR '+folder+'/POTCAR ;'
            command+='cp z_MAE_curve/POSCAR '+folder+'/POSCAR ; '
            command+='cp z_MAE_curve/job_ncl.sh '+folder+'/job_ncl.sh ; '
            command+='cp z_MAE_curve/CHGCAR '+folder+'/CHGCAR ; '
            #command+='cp z_MAE_curve/WAVECAR '+folder+'/WAVECAR ; '
            command+='cd ./'+folder+' ;'
            command+='ln -s  ../z_MAE_curve/WAVECAR ./WAVECAR ; '
            command+='cd ../ ;'
            command+='cp z_MAE_curve/INCAR '+folder+'/INCAR ; '
            command+='cp z_MAE_curve/KPOINTS '+folder+'/KPOINTS ; '
            subprocess.call('$('+command+')', shell=True)
            print("Folder "+folder+" created!")
            replaceAll("./"+folder+"/INCAR",searchExp,replaceExp)
            Addnode("./"+folder+"/INCAR", searchExp,'\n'+'LWAVE = .FALSE.')
            Addnode("./"+folder+"/INCAR", searchExp, '\n'+'LCHARG = .FALSE.')
            replaceAll("./"+folder+"/job_ncl.sh", 'SBATCH -J', '#SBATCH -J '+'MAERt_'+str(np.round((alpha/np.pi)*180,1))+'\n')
            command='cd ./'+folder+' ;'
            command+='sbatch  --reservation=srahmanian_22   job_ncl.sh; '
            command+= 'cd ../'
            print("Submit command: ", command)
            subprocess.call('$('+command+')', shell=True)
            job_fail_Signal9_11(folder, 'log_ncl')
            print('Job for alpha='+str(np.round((alpha/np.pi)*180,2))+' submited!')
            alpha_run.append(alpha)
        Nrunning = Nrunning + 1
        if Nrunning == Nparalel:
            checkT = 60
            #StillRun = [1]
            while Nrunning == Nparalel:#max(StillRun)>0:
                #StillRun = []#np.zeros(np.array(phi).shape[0]*np.array(theta).shape[0])
                #print("Job still runing, will check in ", checkT, " seconds.")
                time.sleep(checkT)
                for alphac in  alpha_run:
                    folder = 'K_'+str(K)+'_RtMAE_'+str(np.round((alphac/np.pi)*180,2))
                    #########
                    FailState = job_fail_Signal9_11(folder, 'log_ncl')
                    #if os.path.isfile(folder+"/log.output")==True:
                    #    Job_ID = find_job_ID(folder)
                    #    subprocess.call('$(cd ./'+folder+'; sacct -j '+str(Job_ID)+' > JobState_'+str(Job_ID)+'; cd ../)', shell=True)
                    #    FailState = find_job_fail(folder, Job_ID)
                    if FailState==1:
                        print('Job for '+folder+' was failed with error! Job resubmitted but check the log files for details')
                        command='cd ./'+folder+' ;'
                        command+='sbatch  --reservation=srahmanian_22  job_ncl.sh; '
                        command+= 'cd ../'
                        subprocess.call('$('+command+')', shell=True)
                        continue
                    print('Job for '+folder+' was NOT failed!')
                    #########
                    if os.path.isfile(folder+'/OUTCAR') == True:
                        #print(folder+'/OUTCAR exists!')
                        with open("./"+folder+"/OUTCAR") as file:
                            JobStat = 1
                            for line in file:
                                if "General timing and accounting informations for this job:" in line:
                                    Nrunning = Nrunning - 1
                                    last_200_OUTCAR_line('./'+folder+'/OUTCAR', './'+folder)
                                    while  os.path.isfile(folder+'/CHGCAR') == False or os.path.isfile(folder+'/vasprun.xml') == False:
                                        if os.path.isfile(folder+'/OSZICAR_cmpl') == True:
                                            break
                                        time.sleep(1)
                                    if os.path.isfile(folder+'/OSZICAR_cmpl') == False:
                                        command='rm  ./'+folder+'/WAVECAR;'
                                        command+='rm  ./'+folder+'/CHGCAR; '
                                        command+='rm  ./'+folder+'/vasprun.xml; '
                                        command+='rm  ./'+folder+'/PROCAR; '
                                        command+='cp ./'+folder+'/OUTCAR ./'+folder+'/OUTCAR_cmpl ;'
                                        command+='cp ./'+folder+'/OSZICAR ./'+folder+'/OSZICAR_cmpl ' 
                                        subprocess.call('$('+command+')', shell=True)
                                    #print("remove command:   ", command)
                                    print(">>> job for alpha = ", str(np.round((alphac/np.pi)*180,2)) ,"completed!")
                                    alpha_run.remove(alphac)
                                    JobStat = 0
                                    #StillRun.append(JobStat)
                            if JobStat == 1:
                                print(">>> job for alpha = ", str(np.round((alphac/np.pi)*180,2))," still runnig!")
                    if os.path.isfile(folder+'/OUTCAR') == False:
                        if os.path.isfile(folder+'/OSZICAR_cmpl') == True:
                            Nrunning = Nrunning - 1
                            break
                        #subprocess.call('$('+'cp ./'+folder+'/OUTCAR_cmpl ./'+folder+'/OUTCAR'+')', shell=True)
                        print(folder+'/OUTCAR doesn\'t exists!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1')
                        #StillRun.append(1)
                #print("StillRun:", StillRun)
    checkT = 20
    t = 0
    StillRun = [1]
    while max(StillRun)>0:
        StillRun = []#np.zeros(np.array(phi).shape[0]*np.array(theta).shape[0])
        print("Job still runing, will check in ", checkT, " seconds.")
        for alpha in  np.linspace(0, 2*np.pi, N_MAE+1):
            folder = 'K_'+str(K)+'_RtMAE_'+str(np.round((alpha/np.pi)*180,2))
            FailState = job_fail_Signal9_11(folder, 'log_ncl')
            #if os.path.isfile(folder+"/log.output")==True:
            #    Job_ID = find_job_ID(folder)
            #    subprocess.call('$(cd ./'+folder+'; sacct -j '+str(Job_ID)+' > JobState_'+str(Job_ID)+'; cd ../)', shell=True)
            #    FailState = find_job_fail(folder, Job_ID)
            if FailState==1:
                print('Job for '+folder+' was failed with error! Job resubmitted but check the log files for details')
                command='cd ./'+folder+' ;'
                command+='sbatch  --reservation=srahmanian_22   job_ncl.sh; '
                command+= 'cd ../'
                subprocess.call('$('+command+')', shell=True)
                StillRun.append(1)
                continue
            print('Job for '+folder+' was NOT failed!')
            if os.path.isfile(folder+'/OUTCAR') == True:
                #print(folder+'/OUTCAR exists!')
                with open("./"+folder+"/OUTCAR") as file:
                    JobStat = 1
                    for line in file:
                        if "General timing and accounting informations for this job:" in line:
                            JobStat = 0
                            last_200_OUTCAR_line('./'+folder+'/OUTCAR', './'+folder)
                            while  os.path.isfile(folder+'/CHGCAR') == False or os.path.isfile(folder+'/vasprun.xml') == False:
                                if os.path.isfile(folder+'/OSZICAR_cmpl') == True:
                                    break
                                time.sleep(1)
                            if os.path.isfile(folder+'/OSZICAR_cmpl') == False:
                                command='rm  ./'+folder+'/WAVECAR;'
                                command+='rm  ./'+folder+'/CHGCAR; '
                                command+='rm  ./'+folder+'/PROCAR; '
                                command+='rm  ./'+folder+'/vasprun.xml; '
                                command+='cp ./'+folder+'/OUTCAR ./'+folder+'/OUTCAR_cmpl ;'
                                command+='cp ./'+folder+'/OSZICAR ./'+folder+'/OSZICAR_cmpl '
                                subprocess.call('$('+command+')', shell=True)
                            print(">>> job for alpha = ", str(np.round((alpha/np.pi)*180,2)) ,"completed!")
                    if JobStat == 1:
                        print(">>> job for alpha = ", str(np.round((alpha/np.pi)*180,2))," still runnig!")
                    StillRun.append(JobStat)
            if os.path.isfile(folder+'/OUTCAR') == False:
                if os.path.isfile(folder+'/OSZICAR_cmpl') == True:
                    StillRun.append(0)
                    #Nrunning = Nrunning - 1
                    break
                #subprocess.call('$('+'cp ./'+folder+'/OUTCAR_cmpl ./'+folder+'/OUTCAR'+')', shell=True)
                print(folder+'/OUTCAR doesn\'t exists!')
                StillRun.append(1)
        #t = t + checkT
        if np.array(StillRun).shape[0]>0:
            print("max of StillRun: ",  max(StillRun))
        time.sleep(checkT)
    print("*****************job for all MAE rotations completed************************")

    E_alpha = []
    alpha_read = []
    MAE_E = []
    for alpha in  np.linspace(0, 2 * np.pi, N_MAE+1):
        folder = 'K_'+str(K)+'_RtMAE_'+str(np.round((alpha/np.pi)*180,2))
        print('Folder is:'+folder)
        if os.path.isdir("./"+folder) == False:
            print(folder, ' doesn\'t exist')
            continue
        elif os.path.isdir("./"+folder) == True:
            print(folder," was found!")
            with open("./"+folder+"/OUTCAR_cmpl") as file:
                FinCond = 0
                for line in file:
                    if "General timing and accounting informations for this job:" in line:
                        FinCond = 1
                        #print('energies: ', Oszicar("./"+folder+"/OSZICAR").all_energies)
                        E_alpha.append(float((Oszicar("./"+folder+"/OSZICAR").all_energies[-1])[-2]))
                        alpha_read.append(alpha)
                        print('Energy for alpha='+str(np.round((alpha/np.pi)*180,2))+'  '+str(E_alpha[-1]))
                if FinCond == 0:
                    print('job for alpha='+str(np.round((alpha/np.pi)*180,2))+' did not finish successfully!')
        #print('TOTEN: ', TOTEN)
    plt.plot(np.array(alpha_read)*180.0/np.pi, (np.array(E_alpha)-E_ref_MAE)*((10.0**-6.0)/readPOSCAR()[0])*(eV/(Ang**3))*(1.0/readPOSCAR()[3]), '-o', label = "K = "+str(K))
    plt.legend()
    if K == min(kpoints):
        np.save("./outputs/Kmin_MAE_curves_alpha_U"+str(U)+"_J"+str(J)+".npy", np.array(alpha_read)*180.0/np.pi)
        np.save("./outputs/Kmin_MAE_curves_mae_U"+str(U)+"_J"+str(J)+".npy", (np.array(E_alpha)-E_ref_MAE)*((10.0**-6.0)/readPOSCAR()[0])*(eV/(Ang**3))*(1.0))
    np.save("./outputs/K_"+str(K)+"_raw_MAE_E_U"+str(U)+"_J"+str(J)+".npy", np.array(E_alpha))
    np.save("./outputs/K_"+str(K)+"_MAE_E_U"+str(U)+"_J"+str(J)+".npy", (np.array(E_alpha)-E_ref_MAE)*((10.0**-6.0)/readPOSCAR()[0])*(eV/(Ang**3))*(1.0/readPOSCAR()[3]))
    np.save("./outputs/K_"+str(K)+"_MAE_alpha_U"+str(U)+"_J"+str(J)+".npy", np.array(alpha_read)*180.0/np.pi)
    plt.savefig("./outputs/MAE_curves_U"+str(U)+"_J"+str(J)+".png")
    MAE = (max(E_alpha)-min(E_alpha))
    K1 = (max(E_alpha)-min(E_alpha))
    MAE_vs_K.append(MAE*((10.0**-6.0)/readPOSCAR()[0])*(eV/(Ang**3))*(1.0))

    outputs.write('\n---------------------------------------------------------------------------------------\nPOSCAR information: \n')
    outputs.write('Volume = '+str(readPOSCAR()[0])+' \n')
    outputs.write('composition = '+str(readPOSCAR()[1])+' \n')
    outputs.write('number of each element = '+str(readPOSCAR()[2])+' \n')
    outputs.write('total number of atoms = '+str(readPOSCAR()[3])+' \n')
    outputs.write('\n---------------------------------------------------------------------------------------\nVASP unit outputs: \n')
    outputs.write('MAE = '+str(MAE)+' eV\n')
    outputs.write('K1 = '+str(K1)+' eV\n')
    outputs.write('M0 = '+str(magnetization('./'+folder_MAE_curve+'/OUTCAR_ncl'))+' Bohr Magneton\n')
    outputs.write('\n---------------------------------------------------------------------------------------\nSI unit outputs (MAE, K1 and M0 are calculated per atom): \n')
    outputs.write("MAE = "+str(MAE*((10.0**-6.0)/readPOSCAR()[0])*(eV/(Ang**3))*(1.0))+" (MJ/m^3)\n")
    param, param_cov = curve_fit(FunMAE, np.array(alpha_read), (np.array(E_alpha)-min(np.array(E_alpha)))*((10.0**-6.0)/readPOSCAR()[0])*(eV/(Ang**3))*(1.0))
    outputs.write("K1  = "+str(param[0])+" (MJ/m^3)\n")
    outputs.write("K2  = "+str(param[1])+" (MJ/m^3)\n")
    outputs.write("M0  = "+str(magnetization('./'+folder_MAE_curve+'/OUTCAR_ncl')*(1e-6)*(Bh/(Ang**3))*(1/readPOSCAR()[0])*(1))+" MA/m\n")
    outputs.write("(BH)max = "+str((1e-3)*0.25*mu0*((magnetization('./'+folder_MAE_curve+'/OUTCAR_ncl')*(Bh/(Ang**3))*(1/readPOSCAR()[0]))**2))+" KJ/m^3\n")
    outputs.write("mu0xHa = "+str(2*(K1*eV)/(Bh*magnetization('./'+folder_MAE_curve+'/OUTCAR_ncl')))+" T\n")
    outputs.write("Hardenss= "+ str(np.sqrt((K1*((1.0)/readPOSCAR()[0])*(eV/(Ang**3))*(1.0))/(mu0*((magnetization('./'+folder_MAE_curve+'/OUTCAR_ncl')*(Bh/(Ang**3))*(1/readPOSCAR()[0]))**2))))+" \n")

    #print(max((np.array(E_alpha)-E_ref_MAE)*((10.0**-6.0)/readPOSCAR()[0])*(eV/(Ang**3))*(1.0/readPOSCAR()[3]))-min((np.array(E_alpha)-E_ref_MAE)*((10.0**-6.0)/readPOSCAR()[0])*(eV/(Ang**3))*(1.0/readPOSCAR()[3])))
    print("MAE = ", MAE*((10.0**-6.0)/readPOSCAR()[0])*(eV/(Ang**3))*(1.0/readPOSCAR()[3]), " (MJ/m^3)")
    print("K1  = ", K1*((10.0**-6.0)/readPOSCAR()[0])*(eV/(Ang**3))*(1.0/readPOSCAR()[3]), " (MJ/m^3)")
    print("M0  = ", magnetization('./'+folder_MAE_curve+'/OUTCAR_ncl')*(1e-6)*(Bh/(Ang**3))*(1/readPOSCAR()[0])*(1/readPOSCAR()[3]), " MA/m")
    print("(BH)max = ", (1e-3)*0.25*mu0*((magnetization('./'+folder_MAE_curve+'/OUTCAR_ncl')*(Bh/(Ang**3))*(1/readPOSCAR()[0]))**2), " KJ/m^3")
    print("mu0xHa = ", 2*(K1*eV)/(Bh*magnetization('./'+folder_MAE_curve+'/OUTCAR_ncl')), " T")
    print("Hardenss= ", np.sqrt((K1*((1.0)/readPOSCAR()[0])*(eV/(Ang**3))*(1.0/readPOSCAR()[3]))/(mu0*((magnetization('./'+folder_MAE_curve+'/OUTCAR_ncl')*(Bh/(Ang**3))*(1/readPOSCAR()[0]))**2))))
    subprocess.call('$('+'mkdir '+folder_MAE_curve+'_RtMAE_IsDone)', shell=True)

outputs.write("MAE vs K = "+str(MAE_vs_K)+" (MJ/m^3)\n")
outputs.write("KPOINTS = "+str(np.array(nkpt_read)/readPOSCAR()[3])+" (number of Kpoints)/atom\n")
outputs.close()
fig = plt.figure()
plt.rcParams['axes.grid'] = True
fig.suptitle('MAE as a function of number of K-points for U='+str(U)+', J='+str(J))
plt.xlabel(r'(number of Kpoints)', size=15)
plt.ylabel(r'MAE $(MJ/m^3)$', size=15)
np.save("./outputs/MAE_vs_Kpoints_curve_U"+str(U)+"_J"+str(J)+"_x.npy", np.array(nkpt_read))
np.save("./outputs/MAE_vs_Kpoints_curve_U"+str(U)+"_J"+str(J)+"_y.npy", MAE_vs_K)
plt.plot(np.array(nkpt_read), MAE_vs_K, '-o',label = "U= "+str(U)+", J="+str(J))
plt.savefig("./outputs/MAE_vs_Kpoints_curve_U"+str(U)+"_J"+str(J)+".png")
command='rm  ./*/PROCAR; '
command+='rm  ./*/vasprun.xml; '
subprocess.call('$('+command+')', shell=True)
print("\n======================================================================")
print("====================== Calculation completed =========================")
print("======================================================================")

