import numpy as np
import csv
import os
import pyccl as ccl
from astropy import units as u
from astropy.coordinates import SkyCoord
from argparse import ArgumentParser

h=.7
cosmo = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=h, sigma8=0.8, n_s=0.96)

rng = np.random.default_rng()

def process_sample(sample):
    alpha = .1
    lya_signal = []
    lya_random = []

    ras, decs, zs, weight, delt = sample.T

    sig_weights = weight * (1+ alpha * delt) # Signal
    rand_weights = -1 * weight
    radii = cosmo.comoving_radial_distance(1/(1+zs))
    c = SkyCoord(ras*u.rad, decs*u.rad, distance=radii*u.Mpc)
    goodndx = (weight > .1) * (np.abs(delt) < 5)
    goodc = c[goodndx]
    goodsw = sig_weights[goodndx]
    goodrw = rand_weights[goodndx]

    glen = len(goodc)
    
    pos = goodc.cartesian.xyz.T
    
    lya_signal = np.zeros((glen, 4))
    lya_random = np.zeros_like(lya_signal)
    
    lya_signal[:,:3] = pos
    lya_signal[:,3] = goodsw
    lya_random[:,:3] = pos
    lya_random[:,3] = goodrw
    
    return lya_signal, lya_random

def process_sample_mock(sample):
    alpha = .1
    lya_signal = []
    lya_random = []
    slen = len(sample)

    ras, decs, zs, weight, delt = sample.T
    permute = rng.permutation(slen)
    ras = ras[permute]
    decs = decs[permute]

    print (ras[:100])
    print(zs[:100])


    sig_weights = weight * (1+ alpha * delt) # Signal
    rand_weights = -1 * weight
    radii = ccl.comoving_radial_distance(cosmo,1/(1+zs))
    c = SkyCoord(ras*u.rad, decs*u.rad, distance=radii*u.Mpc)
    goodndx = (weight > .1) * (np.abs(delt) < 5)
    goodc = c[goodndx]
    goodsw = sig_weights[goodndx]
    goodrw = rand_weights[goodndx]
    
    glen = len(goodc)
    
    pos = goodc.cartesian.xyz.T
    
    lya_signal = np.zeros((glen, 4))
    lya_random = np.zeros_like(lya_signal)
    
    lya_signal[:,:3] = pos
    lya_signal[:,3] = goodsw
    lya_random[:,:3] = pos
    lya_random[:,3] = goodrw
    
    return lya_signal, lya_random




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", choices=["SIGNAL", "SIMUL"])
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    # print(args)
    print ("reading...")
    if args.test:
        fsample = np.load('./test.npy')
    else:
        fsample = np.load('./fullobj.npy')
    print("done...")
    if args.mode == "SIGNAL":
        ls, lr = process_sample(fsample)
        np.savetxt('./data/signal/sig.data.gz', ls, fmt='%1.6f', delimiter=' ')
        np.savetxt('./data/signal/ran.ran.00.gz', lr, fmt='%1.6f', delimiter=' ')
    else:
        ms, mr = process_sample_mock(fsample)
        os.makedirs(f'./data/simul{args.n}', exist_ok=True)
        print(f"Made dir simul{args.n}")
        np.savetxt(f'./data/simul{args.n}/sig.data.gz', ms, fmt='%1.6f', delimiter=' ')
        np.savetxt(f'./data/simul{args.n}/ran.ran.00.gz', mr, fmt='%1.6f', delimiter=' ')
