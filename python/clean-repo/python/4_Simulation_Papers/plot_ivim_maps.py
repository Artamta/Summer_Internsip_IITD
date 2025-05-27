#!/usr/bin/env python3
"""
plot_ivim_maps.py

Reproduce Fig.3-style parameter maps (reference / BE / BE+TV)
for Simulations 1–3, across all b-value combinations (one realization).
Enhanced TV for Sim1 and Sim2, and fixed color scales.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from skimage.restoration import denoise_tv_chambolle
from joblib import Parallel, delayed

# --- User settings ---
OUTDIR = "/Users/ayush/Desktop/project-internsip/Results/4_Simulations_Res"
SIZE   = 64
SNR    = 30
N_JOBS = -1  # use all cores

os.makedirs(OUTDIR, exist_ok=True)

# --- IVIM model ---
def ivim_model(b, D, D_star, f):
    return f * np.exp(-b * D_star) + (1 - f) * np.exp(-b * D)

# --- Phantom builder ---
def make_phantom(vals):
    yy, xx = np.indices((SIZE, SIZE))
    c = (SIZE - 1) / 2
    R = np.sqrt((xx - c)**2 + (yy - c)**2)
    idx = np.floor(R / R.max() * len(vals)).astype(int)
    idx[idx >= len(vals)] = len(vals) - 1
    return vals[idx]

# --- Single-voxel BE fit ---
def fit_voxel(y, bvals):
    try:
        popt, _ = curve_fit(
            ivim_model, bvals, y,
            p0=(1.3e-3, 13e-3, 0.19),
            bounds=([0, 0, 0], [3e-3, 50e-3, 1]),
            maxfev=5000
        )
        return popt
    except:
        return (np.nan, np.nan, np.nan)

# --- Define simulations ---
sims = {
    'Sim1_D':     {'vary':'D',      'values':np.array([0.7,1.0,1.3,1.6,1.9,2.2])*1e-3,
                   'fixed':{'D_star':13e-3,'f':0.19},
                   'cmap':'viridis'},
    'Sim2_Dstar': {'vary':'D_star', 'values':np.array([7,10,13,16,19,22])*1e-3,
                   'fixed':{'D':1.3e-3,   'f':0.19},
                   'cmap':'viridis'},
    'Sim3_f':     {'vary':'f',      'values':np.array([0.03,0.11,0.19,0.27,0.35,0.43]),
                   'fixed':{'D':1.3e-3,   'D_star':13e-3},
                   'cmap':'viridis'},
}

# --- b-value sets ---
b_sets = {
    '4b1': np.array([0,25,200,2000]),
    '4b2': np.array([0,50,150,2000]),
    '6b1': np.array([0,25,100,800,1250,2000]),
    '6b2': np.array([0,50,150,500,1500,2000]),
    '8b1': np.array([0,25,75,100,200,800,1250,2000]),
    '8b2': np.array([0,50,75,150,500,800,1500,2000]),
    '13b':np.array([0,25,50,75,100,150,200,500,800,1000,1250,1500,2000])
}

# --- One realization per sim & b-set ---
for sim_name, cfg in sims.items():
    print(f"\nPlotting {sim_name} maps...")
    fig, axes = plt.subplots(
        nrows=3, ncols=len(b_sets)+1,
        figsize=((len(b_sets)+1)*2, 6),
        constrained_layout=True
    )

    # Reference phantom column
    phantom = make_phantom(cfg['values'])
    ax = axes[0,0]
    im = ax.imshow(phantom, cmap=cfg['cmap'])
    ax.set_title("Reference")
    ax.axis('off')
    fig.colorbar(im, ax=axes[:,0], fraction=0.05, label=sim_name.split('_')[-1])

    # Loop over b-sets
    for col, (bkey, bvals) in enumerate(b_sets.items(), start=1):
        # Build true maps
        D_map  = make_phantom(cfg['values']) if cfg['vary']=='D'      else np.full((SIZE,SIZE), cfg['fixed']['D'])
        Ds_map = make_phantom(cfg['values']) if cfg['vary']=='D_star' else np.full((SIZE,SIZE), cfg['fixed']['D_star'])
        f_map  = make_phantom(cfg['values']) if cfg['vary']=='f'      else np.full((SIZE,SIZE), cfg['fixed']['f'])

        # Simulate noisy data
        S0 = 1.0
        signals = np.stack([ivim_model(b, D_map, Ds_map, f_map) for b in bvals], axis=-1)
        noisy = signals + np.random.normal(0, S0/SNR, signals.shape)

        # BE fit in parallel
        flat = noisy.reshape(-1, len(bvals))
        pops = Parallel(n_jobs=N_JOBS)(
            delayed(fit_voxel)(flat[i], bvals) for i in range(flat.shape[0])
        )
        pops = np.array(pops).reshape(SIZE, SIZE, 3)
        D_be, Ds_be, f_be = pops[:,:,0], pops[:,:,1], pops[:,:,2]

        # Choose stronger TV for Sim1 & Sim2
        tv_w = 0.1 if sim_name in ('Sim1_D','Sim2_Dstar') else 0.02
        D_tv  = denoise_tv_chambolle(D_be,  weight=tv_w)
        Ds_tv = denoise_tv_chambolle(Ds_be, weight=tv_w)
        f_tv  = denoise_tv_chambolle(f_be,  weight=tv_w)

        # Select parameter maps and scales
        if sim_name=='Sim1_D':
            maps = [D_map, D_be, D_tv]
            vmin,vmax = 0.0, 2e-3
            title = 'D'
        elif sim_name=='Sim2_Dstar':
            maps = [Ds_map, Ds_be, Ds_tv]
            vmin,vmax = 0.0, 0.02
            title = 'D*'
        else:
            maps = [f_map, f_be, f_tv]
            vmin,vmax = 0.0, 0.45
            title = 'f'

        # Plot reference row label only once
        if col==1:
            axes[0,1].set_title(bkey)

        # Plot the three rows for this b-set
        for row, arr in enumerate(maps):
            ax = axes[row, col]
            im = ax.imshow(arr, cmap=cfg['cmap'], vmin=vmin, vmax=vmax, interpolation='nearest')
            label = ''
            if row==1: label='BE'
            elif row==2: label='BE+TV'
            ax.set_title(f"{bkey}\n{label}" if row>0 else bkey)
            ax.axis('off')

    # Save figure
    outname = f"Fig3_{sim_name}_maps.png"
    fig.suptitle(f"{sim_name}: {title} maps — Reference / BE / BE+TV", y=1.02)
    fig.savefig(os.path.join(OUTDIR, outname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {outname}")

print("\nAll Fig.3 maps generated in:", OUTDIR)
