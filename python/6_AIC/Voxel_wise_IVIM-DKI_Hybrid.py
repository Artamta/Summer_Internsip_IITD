import numpy as np
import nibabel as nib
from scipy.optimize import curve_fit
import os

# --- IVIM-DKI model function ---
def ivim_dki_model(b, f, D_star, D_slow, k):
    exp1 = np.exp(-b * D_star)
    exp2 = np.exp(-b * D_slow + (1/6) * (b ** 2) * (D_slow ** 2) * k)
    return f * exp1 + (1 - f) * exp2

# --- Load data ---
nii_path = "/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_SNR60/Data-1_Simulation-III_SNR-60.nii"
img = nib.load(nii_path)
data = img.get_fdata()
bvals = np.array([0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000])

# --- Output arrays ---
shape = data.shape[:3]
f_map = np.full(shape, np.nan)
Dstar_map = np.full(shape, np.nan)
Dslow_map = np.full(shape, np.nan)
k_map = np.full(shape, np.nan)
aic_map = np.full(shape, np.nan)

# --- Fitting settings ---
bounds = ([0.0001, 0.0001, 0.001, 0.01], [0.05, 0.5, 1, 3])
p0 = [0.013, 0.013, 0.23, 1.1]

# --- Voxelwise fitting ---
for x in range(shape[0]):
    for y in range(shape[1]):
        for z in range(shape[2]):
            signal = data[x, y, z, :]
            if np.any(signal > 0) and signal[0] != 0:
                y_norm = signal / signal[0]
                if np.any(np.isnan(y_norm)) or np.any(np.isinf(y_norm)):
                    continue
                try:
                    popt, _ = curve_fit(
                        ivim_dki_model, bvals, y_norm, p0=p0, bounds=bounds, maxfev=10000
                    )
                    y_fit = ivim_dki_model(bvals, *popt)
                    rss = np.sum((y_norm - y_fit) ** 2)
                    n = len(bvals)
                    k_param = 4
                    aic = 2 * k_param + n * np.log(rss / n) if rss > 0 else np.nan
                    f_map[x, y, z], Dstar_map[x, y, z], Dslow_map[x, y, z], k_map[x, y, z] = popt
                    aic_map[x, y, z] = aic
                except:
                    continue

# --- Save results ---
out_dir = "/Users/ayush/Desktop/project-internsip/Results/6_Aic_CALC"
os.makedirs(out_dir, exist_ok=True)
affine = img.affine
nib.Nifti1Image(np.nan_to_num(f_map), affine).to_filename(os.path.join(out_dir, "f.nii.gz"))
nib.Nifti1Image(np.nan_to_num(Dstar_map), affine).to_filename(os.path.join(out_dir, "Dstar.nii.gz"))
nib.Nifti1Image(np.nan_to_num(Dslow_map), affine).to_filename(os.path.join(out_dir, "D.nii.gz"))
nib.Nifti1Image(np.nan_to_num(k_map), affine).to_filename(os.path.join(out_dir, "k.nii.gz"))
nib.Nifti1Image(np.nan_to_num(aic_map), affine).to_filename(os.path.join(out_dir, "AIC.nii.gz"))

# --- Print summary statistics ---
print("Fitting complete.")
print("f_map: min {:.4f}, max {:.4f}, mean {:.4f}".format(np.nanmin(f_map), np.nanmax(f_map), np.nanmean(f_map)))
print("Dstar_map: min {:.4f}, max {:.4f}, mean {:.4f}".format(np.nanmin(Dstar_map), np.nanmax(Dstar_map), np.nanmean(Dstar_map)))
print("Dslow_map: min {:.4f}, max {:.4f}, mean {:.4f}".format(np.nanmin(Dslow_map), np.nanmax(Dslow_map), np.nanmean(Dslow_map)))
print("k_map: min {:.4f}, max {:.4f}, mean {:.4f}".format(np.nanmin(k_map), np.nanmax(k_map), np.nanmean(k_map)))
print("AIC_map: min {:.2f}, max {:.2f}, mean {:.2f}".format(np.nanmin(aic_map), np.nanmax(aic_map), np.nanmean(aic_map)))