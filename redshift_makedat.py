import numpy as np
import csv
import os
import pyccl as ccl
from astropy import units as u
from astropy.coordinates import SkyCoord
from argparse import ArgumentParser
import tqdm

# Need to set the scratch database and write directly here
# if we want to remove the gunzip in the scripts


# scratch_dirbase = '/astro/u/anze/work/prakruth/lya-4pcf/scratch'

h=.7
cosmo = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=h, sigma8=0.8, n_s=0.96)


def process_sample(sample, north, test, shuffle=False, seed=None, zmax=4):
    alpha = .1
    lya_signal = []
    lya_random = []

    qid0, ras, decs, zs, weight, delt = sample.T

    if not test:
        if north:
            w=np.where((ras>1)&(ras<5)&(zs<zmax))
        else:
            w=np.where(((ras<=1)|(ras>=5))&(zs<zmax))
    else:
        if north:
            w=np.where((ras>2)&(ras<2.1)&(zs<zmax))
        else:
            w=np.where((ras>2.5)&(ras<2.6)&(zs<zmax))

    qid, ras, decs, zs, weight, delt = sample[w].T
    print (f"Cutting to {len(ras)} pixels. North = {north} Test = {test} Zmax = {zmax}")


    if shuffle:
        print ('shuffling ra/dec')
        rng = np.random.default_rng(seed=seed)
        qtable = np.load('./qidZ_lintable.npy')
        # tqid, tra, tdec = qtable.T

#        assert(np.all(tqid == np.arange(len(tqid))))
        qid = qid.astype(int)-1
        qid_ndx = np.unique(qid)

        fullqid = np.max(qid0).astype(int)

        qid_filt = np.zeros(fullqid + 1).astype(bool)
        qid_filt[qid_ndx] = True
#        print ('total qid:',len(tqid), 'in sample:', len(qid_ndx))

        tqid, tra, tdec, tz, tzbin  = qtable[qid_filt].T
        tqid = tqid.astype(int) ## they just happen to go 1..len()

        print ('total qid:',fullqid, 'in sample:', len(qid_ndx), len(tqid))
        del qid0


        xmap_diff = 1000000
        xmap = np.zeros(fullqid+1,int)+xmap_diff #to make sure we throw index error if fuck up
        xmap[~qid_filt] = -1 # Every QID outside of the ones we care about go to -1==nonsense
#        print("not mapping", len(np.where(xmap == -1)[0]))
        # print(tzbin, np.unique(tzbin))
        unique_bins = np.unique(tzbin)
        unq_ndx = {}
        for ub in unique_bins:
            unq_ndx[ub] = np.where(tzbin==ub)[0]
        for z in unique_bins:
            bin_z = np.where(tzbin == z)[0]
            remap = rng.permutation(bin_z)
            xmap[tqid[bin_z]] = remap
        print('unmodded xmaps', np.where(xmap == xmap_diff)[0])
#         for ub in unique_bins:
#             post_ndx =  np.where(tzbin==ub)[0]
#             print("Unique bin", ub, np.setdiff1d(post_ndx, unq_ndx[ub]), np.setdiff1d(unq_ndx[ub], post_ndx))
        # remap = rng.permutation(len(qid_ndx))

        # xmap[qid_ndx] = qid_ndx[remap]
        ##debug: null reshuflle## remap = np.arange(len(qid_ndx))
        new_ras = np.zeros(len(ras))
        new_decs = np.zeros(len(decs))

        new_ras = tra[xmap[qid]]
        new_decs = tdec[xmap[qid]]

        assert (np.all(new_ras!=0))
        ras = new_ras
        decs = new_decs
        print ("done shuffling")


    print ("calculating distances...")
    sig_weights = weight * (1+ alpha * delt) # Signal
    rand_weights = -1 * weight
    radii = ccl.comoving_radial_distance(cosmo, 1/(1+zs))
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
    fsample = np.load('./qid_obj.npy')

    if args.mode == "SIGNAL":
        # ls, lr = process_sample(fsample, north=True, test=args.test)
        # np.savetxt('./data/signal/sigN.data.gz', ls, fmt='%1.6f', delimiter=' ')
        # np.savetxt('./data/signal/ranN.ran.00.gz', lr, fmt='%1.6f', delimiter=' ')
        ls, lr = process_sample(fsample, north=False, test=args.test, zmax=2.5)
        np.savetxt('./data/signal/sigS.data.gz', ls, fmt='%1.6f', delimiter=' ')
        np.savetxt('./data/signal/ranS.ran.00.gz', lr, fmt='%1.6f', delimiter=' ')

    else:
        # os.makedirs(f'./data/simul{args.n}N', exist_ok=True)
        os.makedirs(f'./data/simul{args.n}S', exist_ok=True)
        print(f"Made dir simul{args.n}")
        # ms, mr = process_sample(fsample,north=True, test=args.test, shuffle=True, seed=args.n)
        # print ('writing N...')
        # np.savetxt(f'./data/simul{args.n}N/sig.data.gz', ms, fmt='%1.6f', delimiter=' ')
        # np.savetxt(f'./data/simul{args.n}N/ran.ran.00.gz', mr, fmt='%1.6f', delimiter=' ')
        ms, mr = process_sample(fsample,north=False, test=args.test, shuffle=True, seed=1000000+args.n, zmax=3)
        print ('writing S...')
        np.savetxt(f'./data/simul{args.n}S/sig.data.gz', ms, fmt='%1.6f', delimiter=' ')
        np.savetxt(f'./data/simul{args.n}S/ran.ran.00.gz', mr, fmt='%1.6f', delimiter=' ')
