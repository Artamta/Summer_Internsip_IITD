import numpy as np
import nibabel as nb
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

# IVIM-DKI model (linear parameter space)
def ivim_dki_model(b, f, D_star, D_slow, k):
    exp1 = np.exp(-b * D_star)
    exp2 = np.exp(-b * D_slow + (1/6) * (b ** 2) * (D_slow ** 2) * k)
    return f * exp1 + (1 - f) * exp2

# Paths
snr60 = "/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_SNR60/Data-1_Simulation-III_SNR-60.nii"

# Fitting settings (linear)
bvals = np.array([0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000])
p0 = [0.1, 0.01, 0.001, 0.5]  # f, D*, D, k
bounds = ([0, 0.0005, 0.0001, 0.01], [1, 0.1, 0.003, 3])

# Load data
snr60_data = nb.load(snr60)
data = snr60_data.get_fdata()
shape = data.shape[:3]
f_map_voxelwise = np.full(shape, np.nan)
Dstar_map_voxelwise = np.full(shape, np.nan)
Dslow_map_voxelwise = np.full(shape, np.nan)
k_map_voxelwise = np.full(shape, np.nan)

# --- Voxelwise Fitting ---
fitted_voxels = 0
for x in range(shape[0]):
    for y in range(shape[1]):
        for z in range(shape[2]):
            signal = data[x, y, z, :]
            if signal[0] <= 0 or np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
                continue
            signal = signal / signal[0]
            if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
                continue
            try:
                popt, _ = curve_fit(
                    ivim_dki_model, bvals, signal, p0=p0, bounds=bounds, maxfev=10000
                )
                f_map_voxelwise[x, y, z] = popt[0]
                Dstar_map_voxelwise[x, y, z] = popt[1]
                Dslow_map_voxelwise[x, y, z] = popt[2]
                k_map_voxelwise[x, y, z] = popt[3]
                fitted_voxels += 1
            except Exception:
                continue

print("Total fitted voxels:", fitted_voxels)
print("f_map_voxelwise: min", np.nanmin(f_map_voxelwise), "max", np.nanmax(f_map_voxelwise))