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


    fig, ax = pplt.subplots(nrows = 1,ncols = 3, share=0)
    pre = np.load("/home/astro/dforero/codes/BAOrec/data/CATALPTCICz0.466G960S1010008301_zspace.dat.npy").astype(np.double)
    pre = (pre + 2500) % 2500
    pk = compute_auto_box(pre[:,0], pre[:,1], pre[:,2], np.ones_like(pre[:,0]), 
                      powspec_conf_file = "examples/powspec_auto.conf",
                      output_file = None)
    plot_correlations(pk, ax, label = 'Pre')

    fig.savefig("/home/astro/dforero/codes/BAOrec/examples/simulation_mg_gpu.png")

    

    post_d = np.load("/home/astro/dforero/codes/BAOrec/data/MG_GPU_CATALPTCICz0.466G960S1010008301_zspace.dat.rec.npy").astype(np.double)

    post_d_cpu = np.load("/home/astro/dforero/codes/BAOrec/data/MG_CPU_CATALPTCICz0.466G960S1010008301_zspace.dat.rec.npy").astype(np.double)

    print(post_d[:10])
    print(post_d_cpu[:10])
    #exit()




    post_r = np.load("/home/astro/dforero/codes/BAOrec/data/MG_GPU_CATALPTCICz0.466G960S1010008301_zspace.ran.rec.iso.npy").astype(np.double)
    post_d = (post_d + 2500) % 2500
    post_r = (post_r + 2500) % 2500
    pk = compute_auto_box_rand(post_d[:,0], post_d[:,1], post_d[:,2], np.ones_like(post_d[:,0]),
                      post_r[:,0], post_r[:,1], post_r[:,2], np.ones_like(post_r[:,0]),
                      powspec_conf_file = "examples/powspec_auto.conf",
                      output_file = None)    
    plot_correlations(pk, ax, label = 'Iso')
    

    fig.savefig("/home/astro/dforero/codes//BAOrec/examples/simulation_mg_gpu.png")
    
    post_r = np.load("/home/astro/dforero/codes//BAOrec/data/MG_GPU_CATALPTCICz0.466G960S1010008301_zspace.ran.rec.sym.npy").astype(np.double)
    post_r = (post_r + 2500) % 2500
    pk = compute_auto_box_rand(post_d[:,0], post_d[:,1], post_d[:,2], np.ones_like(post_d[:,0]),
                      post_r[:,0], post_r[:,1], post_r[:,2], np.ones_like(post_r[:,0]),
                      powspec_conf_file = "examples/powspec_auto.conf",
                      output_file = None)    
    plot_correlations(pk, ax, label = 'Sym')

    ax.legend(loc='top')

    fig.savefig("/home/astro/dforero/codes//BAOrec/examples/simulation_mg_gpu.png")



    