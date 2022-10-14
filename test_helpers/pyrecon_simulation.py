#!/usr/bin/env python
import numpy as np
import pandas as pd
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import proplot as pplt
from tqdm import tqdm
import glob
import sys
from pyrecon import IterativeFFTReconstruction
sys.path.append("/global/u1/d/dforero/codes/powspec_py/powspec")
from pypowspec import compute_auto_box, compute_auto_box_rand



def plot_correlations(pk_res, ax, label):
    
    k = pk_res['k']
    pk = pk_res['multipoles']
    ax[0].plot(k, k*(pk[:,0]), label=label)
    ax[1].plot(k, k*pk[:,1], label=label)
    ax[2].plot(k, k*pk[:,2], label=label)
    
    ax.format(xscale='log', xlabel='k [h/Mpc]', ylabel = r'$kP(k)$')
    
    
    return ax


if __name__ == '__main__':


    fig, ax = pplt.subplots(nrows = 2,ncols = 3, share=0)
    data_fn = "/global/cfs/projectdirs/desi/mocks/UNIT/HOD_Shadab/HOD_boxes/redshift0.9873/UNIT_DESI_Shadab_HOD_snap97_ELG_v0.txt"
    pre = pd.read_csv(data_fn, delim_whitespace = True, engine = 'c', usecols = (0,1,3), names = ['x', 'y', 'z']).values.astype(np.double)
    box_size = 1000.
    rand_multiplier = 10
    ndata = pre.shape[0]

    recon = IterativeFFTReconstruction(f=0.757, bias=2.2, nmesh=512, boxsize=box_size, boxcenter=box_size / 2, los=[0,0,1], wrap=True, nthreads=64, fft_engine = 'fftw')
    recon.assign_data(pre, np.broadcast_to([1.], pre.shape[0]))
    recon.set_density_contrast(smoothing_radius = 15.)
          
    
    recon.run(niterations = 3)
    positions_rec_data = recon.read_shifted_positions(pre).astype(np.double)
    rng = np.random.default_rng(42)
    positions_randoms = rng.random((rand_multiplier * ndata, 3), dtype = np.float32) * box_size
    positions_rec_randoms_sym = recon.read_shifted_positions(positions_randoms).astype(np.double) #Sym
    positions_rec_randoms_iso = recon.read_shifted_positions(positions_randoms, field='disp').astype(np.double) #Iso
    

    
    fig, ax = pplt.subplots(nrows = 1,ncols = 3, share=0)
    pre = (pre + box_size) % box_size
    pk = compute_auto_box(pre[:,0], pre[:,1], pre[:,2], np.ones_like(pre[:,0]), 
                      powspec_conf_file = "test_helpers/powspec_auto.conf",
                      output_file = None)
    plot_correlations(pk, ax, label = 'Pre')

    fig.savefig("plots/pyrecon_simulation.png")

    

    post_d = positions_rec_data
    post_r = positions_rec_randoms_iso
    post_d = (post_d + box_size) % box_size
    post_r = (post_r + box_size) % box_size
    pk = compute_auto_box_rand(post_d[:,0], post_d[:,1], post_d[:,2], np.ones_like(post_d[:,0]),
                      post_r[:,0], post_r[:,1], post_r[:,2], np.ones_like(post_r[:,0]),
                      powspec_conf_file = "test_helpers/powspec_auto.conf",
                      output_file = None)    
    plot_correlations(pk, ax, label = 'Iso')
    

    fig.savefig("plots/pyrecon_simulation.png")
    
    post_r = positions_rec_randoms_sym
    post_r = (post_r + box_size) % box_size
    pk = compute_auto_box_rand(post_d[:,0], post_d[:,1], post_d[:,2], np.ones_like(post_d[:,0]),
                      post_r[:,0], post_r[:,1], post_r[:,2], np.ones_like(post_r[:,0]),
                      powspec_conf_file = "test_helpers/powspec_auto.conf",
                      output_file = None)    
    plot_correlations(pk, ax, label = 'Sym')

    ax.legend(loc='top')

    fig.savefig("plots/pyrecon_simulation.png")