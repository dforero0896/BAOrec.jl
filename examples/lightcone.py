#!/usr/bin/env python
import numpy as np
import pandas as pd
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import proplot as pplt
import MAS_library as MASL
import Pk_library as PKL
from tqdm import tqdm
import glob
import sys
import numpy as np
import sys
sys.path.append("/home/astro/dforero/codes/pypowspec/powspec")
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
    data_fn = "/home/astro/dforero/codes/BAOrec/data/Patchy-Mocks-DR12NGC-COMPSAM_V6C_0001.dat"
    rand_fn = "/home/astro/dforero/codes/BAOrec/data/Patchy-Mocks-Randoms-DR12NGC-COMPSAM_V6C_x20.dat"
    data = pd.read_csv(data_fn, delim_whitespace = True, engine = 'c', usecols = (0,1,2,3,4), names = ['ra', 'dec', 'z', 'w', 'nz']).values
    data = data[(data[:,2] > 0.2) & (data[:,2] < 0.5)]

    rand = pd.read_csv(rand_fn, delim_whitespace = True, engine = 'c', usecols = (0,1,2,3,4), names = ['ra', 'dec', 'z', 'w', 'nz']).values
    rand = rand[(rand[:,2] > 0.2) & (rand[:,2] < 0.5)]

    fkp_data = 1. / (1 + data[:,4] * P0)
    fkp_rand = 1. / (1 + rand[:,4] * P0)

    pk = compute_auto_lc(data[:,0], data[:,1], data[:,2], data[:,3], fkp_data, data[:,4],
                    rand[:,0], rand[:,1], rand[:,2], rand[:,3], fkp_rand, rand[:,4],
                    powspec_conf_file = "examples/powspec_lc.conf",
                    output_file = None)
    plot_correlations(pk, ax, label = 'Pre')

    fig.savefig("/home/astro/dforero/codes/BAOrec/examples/lightcone.png")

   

    data = np.load("/home/astro/dforero/codes/BAOrec/data/CPU_Patchy-Mocks-DR12NGC-COMPSAM_V6C_0001.dat.rec.npy").astype(np.double)
    rand = np.load("/home/astro/dforero/codes/BAOrec/data/CPU_Patchy-Mocks-Randoms-DR12NGC-COMPSAM_V6C_x20.dat.rec.iso.npy").astype(np.double)
    
    fkp_data = 1. / (1 + data[:,4] * P0)
    fkp_rand = 1. / (1 + rand[:,4] * P0)

    pk = compute_auto_lc(data[:,0], data[:,1], data[:,2], data[:,3], fkp_data, data[:,4],
                    rand[:,0], rand[:,1], rand[:,2], rand[:,3], fkp_rand, rand[:,4],
                    powspec_conf_file = "examples/powspec_lc.conf",
                    output_file = None)

    plot_correlations(pk, ax, label = 'Iso')
    

    fig.savefig("/home/astro/dforero/codes/BAOrec/examples/lightcone.png")
    
    rand = np.load("/home/astro/dforero/codes/BAOrec/data/CPU_Patchy-Mocks-Randoms-DR12NGC-COMPSAM_V6C_x20.dat.rec.sym.npy").astype(np.double)
    fkp_data = 1. / (1 + data[:,4] * P0)
    fkp_rand = 1. / (1 + rand[:,4] * P0)

    pk = compute_auto_lc(data[:,0], data[:,1], data[:,2], data[:,3], fkp_data, data[:,4],
                    rand[:,0], rand[:,1], rand[:,2], rand[:,3], fkp_rand, rand[:,4],
                    powspec_conf_file = "examples/powspec_lc.conf",
                    output_file = None)
    plot_correlations(pk, ax, label = 'Sym')

    ax.legend(loc='top')

    fig.savefig("/home/astro/dforero/codes/BAOrec/examples/lightcone.png")