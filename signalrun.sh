python makedat.py SIGNAL --test
cd data/signal
bash ../../gzipsh.sh
cd ../..
#### RUN ENCORE COMMAND ####
# ./mac_encore -ran 100 -outstr tmp/test
rm -rf output/signal
mkdir output/signal
csh lya_signal.slurm

