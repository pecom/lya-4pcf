python makedat.py SIMUL --test
cd data/simul
bash ../../gzipsh.sh
cd ../..
#### RUN ENCORE COMMAND ####
# ./mac_encore -ran 100 -outstr tmp/test
csh lya_simul.slurm
rm -rf output/simul$1
mkdir output/simul$1 
mv output/tmp/*  output/simul$1
