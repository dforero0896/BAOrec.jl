#!/usr/bin/env python
import numpy as np
import pandas as pd
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import proplot as pplt
import cosmology_library as CL
from tqdm import tqdm
import glob
import sys
import numpy as np
import sys
sys.path.append("/global/u1/d/dforero/codes/powspec_py/powspec")
from pypowspec import compute_auto_lc
from pyrecon import IterativeFFTReconstruction
from pyrecon.utils import sky_to_cartesian, cartesian_to_sky



def plot_correlations(pk_res, ax, label):
    
    k = pk_res['k']
    pk = pk_res['multipoles']
    ax[0].plot(k, k*(pk[:,0]), label=label)
    ax[1].plot(k, k*pk[:,1], label=label)
    ax[2].plot(k, k*pk[:,2], label=label)
    
    ax.format(xscale='log', xlabel='k [h/Mpc]', ylabel = r'$kP(k)$')
    
    
    return ax


if __name__ == '__main__':

    P0 = 5e3
    Omega_m = 0.31
    Omega_l = 1. - Omega_m
    ztable = np.linspace(0, 10, 10000)
    
    rtable = np.array([CL.comoving_distance(z, Omega_m, Omega_l) for z in ztable])
    fig, ax = pplt.subplots(nrows = 1,ncols = 3, share=0)
    data_fn = "/global/cfs/projectdirs/desi/mocks/UNIT/HOD_Shadab/multiple_snapshot_lightcone/UNIT_lightcone_multibox_ELG_footprint_nz_NGC.dat"
    rand_fn = "/global/cfs/projectdirs/desi/mocks/UNIT/HOD_Shadab/multiple_snapshot_lightcone/UNIT_lightcone_multibox_ELG_footprint_nz_1xdata_5.ran_NGC.dat"
    data = pd.read_csv(data_fn, delim_whitespace = True, engine = 'c', usecols = (0,1,3,4), names = ['ra', 'dec', 'z', 'nz']).values
    data = data[(data[:,2] > 0.8) & (data[:,2] < 1)]

    rand = pd.read_csv(rand_fn, delim_whitespace = True, engine = 'c', usecols = (0,1,3,4), names = ['ra', 'dec', 'z', 'nz']).values
    rand = rand[(rand[:,2] > 0.8) & (rand[:,2] < 1)]
    
    _data = data.copy()
    _rand = data.copy()
    
    data = np.c_[sky_to_cartesian(np.interp(data[:,2], ztable, rtable) , data[:,0], data[:,1]), data[:,3]]
    rand = np.c_[sky_to_cartesian(np.interp(rand[:,2], ztable, rtable) , rand[:,0], rand[:,1]), rand[:,3]]

    fkp_data = 1. / (1 + data[:,3] * P0)
    fkp_rand = 1. / (1 + rand[:,3] * P0)

    padding = 500.
    
    box_min = rand[:,:3].min(axis=0) - padding / 2
    box_max = rand[:,:3].max(axis=0) + padding / 2
    boxsize = max(box_max - box_min)
    boxcenter = 0.5 * (rand[:,:3].max(axis=0) + rand[:,:3].min(axis=0))
    
    recon = IterativeFFTReconstruction(f=0.757, bias=2.2, nmesh=512, nthreads=64, fft_engine = 'fftw', boxsize = boxsize, boxcenter = boxcenter, wrap=False)
    recon.assign_data(data[:,:3], weights = fkp_data)
    recon.assign_randoms(rand[:,:3], weights = fkp_rand)
    
    
    recon.set_density_contrast(smoothing_radius = 15.)
    
    recon.run(niterations = 3)
    
    
    positions_rec_data = recon.read_shifted_positions(data[:,:3]).astype(np.double)
    positions_rec_randoms_sym = recon.read_shifted_positions(rand[:,:3]).astype(np.double) #Sym
    positions_rec_randoms_iso = recon.read_shifted_positions(rand[:,:3], field='disp').astype(np.double) #Iso
    r, ra, dec = cartesian_to_sky(data[:,:3])
    data = np.c_[ra, dec, np.interp(r, rtable, ztable), data[:,3]]
    r, ra, dec = cartesian_to_sky(rand[:,:3])
    rand = np.c_[ra, dec, np.interp(r, rtable, ztable), rand[:,3]]

    pk = compute_auto_lc(data[:,0], data[:,1], data[:,2], np.ones_like(data[:,3]), fkp_data, data[:,3],
                    rand[:,0], rand[:,1], rand[:,2], np.ones_like(rand[:,3]), fkp_rand, rand[:,3],
                    powspec_conf_file = "test_helpers/powspec_lc.conf",
                    output_file = None)
    plot_correlations(pk, ax, label = 'Pre')

    fig.savefig("plots/pyrecon_lightcone.png")

    
    r, ra, dec = cartesian_to_sky(positions_rec_data)
    
    data = np.c_[ra, dec, np.interp(r, rtable, ztable), data[:,3]]
    r, ra, dec = cartesian_to_sky(positions_rec_randoms_iso)
    rand = np.c_[ra, dec, np.interp(r, rtable, ztable), rand[:,3]]

    fkp_data = 1. / (1 + data[:,3] * P0)
    fkp_rand = 1. / (1 + rand[:,3] * P0)

    pk = compute_auto_lc(data[:,0], data[:,1], data[:,2], np.ones_like(data[:,3]), fkp_data, data[:,3],
                    rand[:,0], rand[:,1], rand[:,2], np.ones_like(rand[:,3]), fkp_rand, rand[:,3],
                    powspec_conf_file = "test_helpers/powspec_lc.conf",
                    output_file = None)

    plot_correlations(pk, ax, label = 'Iso')
    

    fig.savefig("plots/pyrecon_lightcone.png")
    
    r, ra, dec = cartesian_to_sky(positions_rec_randoms_sym)
    rand = np.c_[ra, dec, np.interp(r, rtable, ztable), rand[:,3]]
    fkp_data = 1. / (1 + data[:,3] * P0)
    fkp_rand = 1. / (1 + rand[:,3] * P0)

    pk = compute_auto_lc(data[:,0], data[:,1], data[:,2], np.ones_like(data[:,3]), fkp_data, data[:,3],
                    rand[:,0], rand[:,1], rand[:,2], np.ones_like(rand[:,3]), fkp_rand, rand[:,3],
                    powspec_conf_file = "test_helpers/powspec_lc.conf",
                    output_file = None)
    plot_correlations(pk, ax, label = 'Sym')

    ax.legend(loc='top')

    fig.savefig("plots/pyrecon_lightcone.png")