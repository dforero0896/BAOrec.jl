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
import numpy as np
import sys
sys.path.append("/global/u1/d/dforero/codes/powspec_py/powspec")
from pypowspec import compute_auto_lc



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
    fig, ax = pplt.subplots(nrows = 1,ncols = 3, share=0)
    data_fn = "/global/cfs/projectdirs/desi/mocks/UNIT/HOD_Shadab/multiple_snapshot_lightcone/UNIT_lightcone_multibox_ELG_footprint_nz_NGC.dat"
    rand_fn = "/global/cfs/projectdirs/desi/mocks/UNIT/HOD_Shadab/multiple_snapshot_lightcone/UNIT_lightcone_multibox_ELG_footprint_nz_1xdata_5.ran_NGC.dat"
    data = pd.read_csv(data_fn, delim_whitespace = True, engine = 'c', usecols = (0,1,3,4), names = ['ra', 'dec', 'z', 'nz']).values
    data = data[(data[:,2] > 0.) & (data[:,2] < 1)]

    rand = pd.read_csv(rand_fn, delim_whitespace = True, engine = 'c', usecols = (0,1,3,4), names = ['ra', 'dec', 'z', 'nz']).values
    rand = rand[(rand[:,2] > 0.) & (rand[:,2] < 1)]

    fkp_data = 1. / (1 + data[:,3] * P0)
    fkp_rand = 1. / (1 + rand[:,3] * P0)

    pk = compute_auto_lc(data[:,0], data[:,1], data[:,2], np.ones_like(data[:,3]), fkp_data, data[:,3],
                    rand[:,0], rand[:,1], rand[:,2], np.ones_like(rand[:,3]), fkp_rand, rand[:,3],
                    powspec_conf_file = "test_helpers/powspec_lc.conf",
                    output_file = None)
    plot_correlations(pk, ax, label = 'Pre')
   

    data = np.load("data/GPU_UNIT_lightcone_multibox_ELG_footprint_nz_NGC.rec.npy").astype(np.double)
    rand = np.load("data/GPU_UNIT_lightcone_multibox_ELG_footprint_nz_1xdata_5.ran_NGC.rec.iso.npy").astype(np.double)
    print(data[:100])
    fkp_data = 1. / (1 + data[:,3] * P0)
    fkp_rand = 1. / (1 + rand[:,3] * P0)

    pk = compute_auto_lc(data[:,0], data[:,1], data[:,2], np.ones_like(data[:,3]), fkp_data, data[:,3],
                    rand[:,0], rand[:,1], rand[:,2], np.ones_like(rand[:,3]), fkp_rand, rand[:,3],
                    powspec_conf_file = "test_helpers/powspec_lc.conf",
                    output_file = None)

    plot_correlations(pk, ax, label = 'Iso')
    
    
    rand = np.load("data/GPU_UNIT_lightcone_multibox_ELG_footprint_nz_1xdata_5.ran_NGC.rec.sym.npy").astype(np.double)
    fkp_data = 1. / (1 + data[:,3] * P0)
    fkp_rand = 1. / (1 + rand[:,3] * P0)

    pk = compute_auto_lc(data[:,0], data[:,1], data[:,2], np.ones_like(data[:,3]), fkp_data, data[:,3],
                    rand[:,0], rand[:,1], rand[:,2], np.ones_like(rand[:,3]), fkp_rand, rand[:,3],
                    powspec_conf_file = "test_helpers/powspec_lc.conf",
                    output_file = None)
    plot_correlations(pk, ax, label = 'Sym')

    ax.legend(loc='top')

    fig.savefig("plots/lightcone_gpu.png")