#!/usr/bin/env python3
"""
run_all_ivim_simulations.py

Updated: saves bull’s-eye phantoms and uses tighter fit bounds.
Implements Simulations 1–3 from:
  Malagi et al., “Effect of combination and number of b values...”
  Mag Reson Mater Phys Biol Med (2019) 32:519–527.

Produces:
 - phantom_D.png, phantom_Dstar.png, phantom_f.png
 - Sim1_D_accuracy.png, Sim2_Dstar_accuracy.png, Sim3_f_accuracy.png
"""

import os, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from skimage.restoration import denoise_tv_chambolle
from joblib import Parallel, delayed

# === User settings ===
OUTDIR    = "/Users/ayush/Desktop/project-internsip/Results/4_Simulations_Res"
SIZE      = 64
N_REAL    = 50
N_JOBS    = -1      # all cores
SNR       = 30
TV_WEIGHT = 0.02

os.makedirs(OUTDIR, exist_ok=True)

# === IVIM model ===
def ivim_model(b, D, D_star, f):
    return f * np.exp(-b * D_star) + (1 - f) * np.exp(-b * D)

# === Bull’s-eye phantom builder ===
def make_phantom(values):
    yy, xx = np.indices((SIZE, SIZE))
    c = (SIZE - 1) / 2
    R = np.sqrt((xx-c)**2 + (yy-c)**2)
    idx = np.floor(R / R.max() * len(values)).astype(int)
    idx[idx >= len(values)] = len(values) - 1
    return values[idx]

# === Define simulations inline ===
sims = {
    'Sim1_D':     {'vary':'D',      'values':np.array([0.7,1.0,1.3,1.6,1.9,2.2])*1e-3,
                   'fixed':{'D_star':13e-3, 'f':0.19}},
    'Sim2_Dstar': {'vary':'D_star', 'values':np.array([7,10,13,16,19,22])*1e-3,
                   'fixed':{'D':1.3e-3,    'f':0.19}},
    'Sim3_f':     {'vary':'f',      'values':np.array([0.03,0.11,0.19,0.27,0.35,0.43]),
                   'fixed':{'D':1.3e-3,    'D_star':13e-3}}
}

# === Save the three phantoms ===
for sim_name, cfg in sims.items():
    ph = make_phantom(cfg['values'])
    plt.figure(figsize=(4,4))
    plt.imshow(ph, cmap='viridis')
    plt.title(f"{sim_name} Phantom")
    cb = plt.colorbar(shrink=0.8)
    cb.set_label(sim_name.split('_')[-1])
    plt.axis('off')
    plt.tight_layout()
    fname = f"phantom_{sim_name.split('_')[-1]}.png"
    plt.savefig(os.path.join(OUTDIR, fname), dpi=150)
    plt.close()

# === Simulation helper functions ===
def simulate_ivim(D_map, Ds_map, f_map, bvals):
    S0 = 1.0
    signals = np.stack([ivim_model(b, D_map, Ds_map, f_map) for b in bvals], axis=-1)
    noise_sigma = S0 / SNR
    return signals + np.random.normal(0, noise_sigma, signals.shape)

def fit_voxel(y, bvals):
    try:
        popt, _ = curve_fit(ivim_model, bvals, y,
                            p0=(1.3e-3, 13e-3, 0.19),
                            bounds=([0,0,0], [3e-3,50e-3,1]),
                            maxfev=8000)
        return popt
    except:
        return (np.nan, np.nan, np.nan)

def rrmse(true_map, est_map):
    mask = ~np.isnan(est_map)
    return np.sqrt(((true_map[mask] - est_map[mask])**2).mean()) / true_map[mask].mean() * 100

# === b-value sets ===
b_sets = {
    '4b1': np.array([0,25,200,2000]),
    '4b2': np.array([0,50,150,2000]),
    '6b1': np.array([0,25,100,800,1250,2000]),
    '6b2': np.array([0,50,150,500,1500,2000]),
    '8b1': np.array([0,25,75,100,200,800,1250,2000]),
    '8b2': np.array([0,50,75,150,500,800,1500,2000]),
    '13b':np.array([0,25,50,75,100,150,200,500,800,1000,1250,1500,2000])
}

# === Main simulation loop ===
for sim_name, cfg in sims.items():
    print(f"\n=== {sim_name} ===")
    vals  = cfg['values']
    fixed = cfg['fixed']

    rmse_be = {k:[] for k in b_sets}
    rmse_tv = {k:[] for k in b_sets}
    start_sim = time.time()

    for bkey, bvals in b_sets.items():
        print(f" b-set={bkey}")
        t0 = time.time()

        for r in range(1, N_REAL+1):
            # True parameter maps
            D_map  = make_phantom(vals) if cfg['vary']=='D'      else np.full((SIZE,SIZE), fixed['D'])
            Ds_map = make_phantom(vals) if cfg['vary']=='D_star' else np.full((SIZE,SIZE), fixed['D_star'])
            f_map  = make_phantom(vals) if cfg['vary']=='f'      else np.full((SIZE,SIZE), fixed['f'])

            noisy = simulate_ivim(D_map, Ds_map, f_map, bvals)

            # BE fitting in parallel
            flat = noisy.reshape(-1, len(bvals))
            pops = Parallel(n_jobs=N_JOBS)(
                delayed(fit_voxel)(flat[i], bvals) for i in range(flat.shape[0])
            )
            pops = np.array(pops).reshape(SIZE, SIZE, 3)
            D_be, Ds_be, f_be = pops[:,:,0], pops[:,:,1], pops[:,:,2]

            # TV denoise
            D_tv  = denoise_tv_chambolle(D_be,  weight=TV_WEIGHT)
            Ds_tv = denoise_tv_chambolle(Ds_be, weight=TV_WEIGHT)
            f_tv  = denoise_tv_chambolle(f_be,  weight=TV_WEIGHT)

            # Select target map
            if sim_name=='Sim1_D':
                true_map, m_be, m_tv = D_map,  D_be,  D_tv
                ylabel, title, out_png = 'D', 'Sim1: D accuracy',  'Sim1_D_accuracy.png'
            elif sim_name=='Sim2_Dstar':
                true_map, m_be, m_tv = Ds_map, Ds_be, Ds_tv
                ylabel, title, out_png = 'D*','Sim2: D* accuracy', 'Sim2_Dstar_accuracy.png'
            else:
                true_map, m_be, m_tv = f_map,  f_be,  f_tv
                ylabel, title, out_png = 'f', 'Sim3: f accuracy',  'Sim3_f_accuracy.png'

            rmse_be[bkey].append(rrmse(true_map, m_be))
            rmse_tv[bkey].append(rrmse(true_map, m_tv))

            if r % 10 == 0:
                print(f"  Real {r}/{N_REAL}")

        # Average across realizations
        rmse_be[bkey] = np.mean(rmse_be[bkey])
        rmse_tv[bkey] = np.mean(rmse_tv[bkey])
        print(f"   → BE {rmse_be[bkey]:.1f}%, TV {rmse_tv[bkey]:.1f}% in {(time.time()-t0):.1f}s")

    # Plot bar chart
    labels = list(b_sets)
    x      = np.arange(len(labels))
    w      = 0.35

    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(x-w/2, [rmse_be[k] for k in labels], w, label='BE')
    ax.bar(x+w/2, [rmse_tv[k] for k in labels], w, label='BE+TV')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel(f'RRMSE {ylabel} (%)')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, out_png), dpi=150)
    plt.close(fig)

    print(f" Saved {out_png} | Completed in {(time.time()-start_sim)/60:.1f} min")

print("\nAll done! Check:", OUTDIR)
