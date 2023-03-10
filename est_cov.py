import os,sys,pickle
import numpy as np
from scipy.stats import percentileofscore, chi2
from scipy.linalg import sqrtm
from scipy.interpolate import interp1d
from scipy.special import loggamma
from scipy.optimize import minimize
# import emcee # only needed for step 8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rng = np.random.default_rng()

R_min = 20 
R_max = 160 
n_r = 10
LMAX = 4

ang_filt = np.array([False, False, False, False, False, False, False,  True, False,
       False,  True, False, False,  True, False, False,  True, False,
       False,  True, False, False,  True, False,  True, False, False,
        True, False,  True, False,  True, False, False, False,  True,
       False, False,  True, False,  True, False,  True, False,  True,
       False, False,  True, False,  True, False, False,  True, False,
        True, False, False,  True, False,  True, False,  True, False,
        True, False])
radial_filt = np.array([False, False, False, False, False, False, False, False, False,
        True,  True,  True,  True,  True,  True, False,  True,  True,
        True,  True,  True, False,  True,  True,  True,  True, False,
        True,  True,  True, False,  True,  True, False,  True, False,
       False, False, False, False, False, False, False, False,  True,
        True,  True,  True,  True, False,  True,  True,  True,  True,
       False,  True,  True,  True, False,  True,  True, False,  True,
       False, False, False, False, False, False, False, False,  True,
        True,  True,  True, False,  True,  True,  True, False,  True,
        True, False,  True, False, False, False, False, False, False,
       False,  True,  True,  True, False,  True,  True, False,  True,
       False, False, False, False, False, False,  True,  True, False,
        True, False, False, False, False, False,  True, False, False,
       False, False, False])

binner = lambda bins: (0.5+bins)*(R_max-R_min)/n_r+R_min


def load_boss(patch,return_all=False,disc=False, data_dir = './data/delta/'):
    """Load precomputed 4PCFs from BOSS.
    This loads data from the NGC or SGC, here labelled 'N' or 'S'.
    If 'disc' = True, then the disconnected 4PCF is loaded."""
    if disc:
        infile = data_dir+'boss_cmass%s.zeta_discon_4pcf.txt'%patch
    else:
        infile = data_dir+'boss_cmass%s.zeta_4pcf.txt'%patch
    infile = data_dir+'%s.zeta_4pcf.txt'%patch
    bins1,bins2,bins3 = np.asarray(np.genfromtxt(infile,skip_header=3,max_rows=3),dtype=int)
    ell1,ell2,ell3 = np.asarray(np.loadtxt(infile,skiprows=9)[:,:3],dtype=int).T
    fourpcf_boss = np.loadtxt(infile,skiprows=9)[:,3:]
    r1_4pcf = binner(bins1)
    r2_4pcf = binner(bins2)
    r3_4pcf = binner(bins3)
    if return_all:
        return [r1_4pcf,r2_4pcf,r3_4pcf],[bins1,bins2,bins3],[ell1,ell2,ell3],fourpcf_boss
    else:
        return fourpcf_boss

def get_chi2(cov, mn=np.zeros(1288), signal=np.zeros(1288)):
    chi2 = np.inner(signal - mn, np.inner(cov, signal - mn))
    return chi2

def format_boss(prefix, ddir='./data/signal/'):
    signal = load_boss(prefix, return_all=False, data_dir=ddir)
    signal *= 10**4
    filt_signal = signal[ang_filt][:,radial_filt]
    flat_filt_signal = filt_signal.ravel()
    return flat_filt_signal

empty_cov = np.zeros((1288,1288))
all_fsigs = np.zeros((32, 1288))

for i in range(20, 52):
	fsignal = format_boss('sig', f'./output/simul{i}S/')
	all_fsigs[i-20] = fsignal

mean_signal = np.mean(all_fsigs, axis=0)

for af in all_fsigs:
	empty_cov += np.outer(af - mean_signal, af-mean_signal)
empty_cov /= icount

new_cov = np.diag(empty_cov)
cov_inv = np.diag(1/new_cov)

chi2 = get_chi2(cov_inv*icount, signal=mean_signal)
print(f"Mean signal χ2 score: {chi2}")

for i in range(20, 52):
	chi2 = get_chi2(cov_inv, signal=all_fsigs[i-20], mn=mean_signal)
	print(f"Patch {i - 19} χ2 score: {chi2}")

test_sig = format_boss('sig', './output/simul60S/')
chi2 = get_chi2(cov_inv, signal=test_sig)
print(f"Test χ2 score: {chi2}")
