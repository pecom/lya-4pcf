Lines to change:


lya_signal.slurm / lya_simul.slurm :
	Lines 19 - 30  : if SLURM cluster [If needed]
	Lines 47 : Change to path of the directory set (set anzepath = )
	Line 52 : Change to be some scratch directory with fast I/O (set tmp = )
	Line 64: Set number of threads (set OMP_NUM_THREADS = )
	Line 88: Change to access the relevant python3 version [if needed]

Run testmulti.sh - Will make 5 simulations and 1 true signal on a reduced data set. Mostly to check that everything is working
	There should be output in anze_encore/output/simul[1-5] and anze_encore/output/signal/ in the form of sig.zeta_4pcf.txt files
	There is an error log file in anze_encore/output/simul[1-5]/errlog which prints some debug info and will have any error statement if there are issues. If there are no issues it just prints some stuff in the file. It should say "Finished with computation .... "


If All is well with testmulti.sh then change the first line of singlerun.sh and signalrun.sh from (drop the —test argument):
python makedat.py SIGNAL --test  —> python makedat.py SIGNAL 
python makedat.py SIMUL --test  —> python makedat.py SIMUL 
 
I have never ran on the full data set and have no idea if the script to generate the data/simulation works. If possible it'll be a good idea to re-run testmulti.sh and double check that it runs properly.

Then we can run fullmulti.sh and hope and pray. 
