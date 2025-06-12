import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
import os
from lmfit import Model, Parameters

# IVIM-DKI model (not in log space for lmfit, use bounds instead)
def ivim_dki_model(b, f, D_star, D_slow, k):
    exp1 = np.exp(np.clip(-b * D_star, -100, 100))
    exp2 = np.exp(np.clip(-b * D_slow + (1/6) * (b ** 2) * (D_slow ** 2) * k, -100, 100))
    return f * exp1 + (1 - f) * exp2

# Paths ---
snr60 = "/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_SNR60/Data-1_Simulation-III_SNR-60.nii"
ref_f_path = "/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_f_map.nii"
toolbox_f_path = "/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-III_SNR-60_f.nii"
toolbox_d_path = "/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-III_SNR-60_D.nii"
toolbox_dstar_path = "/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-III_SNR-60_D_star.nii"
toolbox_k_path = "/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-III_SNR-60_k.nii"

# Fitting settings
bvals = np.array([0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000])
p0 = [0.013, 0.023, 0.23, 1.1]
lower_bound = [0.0001, 0.0001, 0.001, 0.01]
upper_bound = [0.05, 0.5, 2, 3]

# Load data
snr60_data = nb.load(snr60)
data = snr60_data.get_fdata()
shape = data.shape[:3]
f_map_voxelwise = np.full(shape, np.nan)
Dstar_map_voxelwise = np.full(shape, np.nan)
Dslow_map_voxelwise = np.full(shape, np.nan)
k_map_voxelwise = np.full(shape, np.nan)

# --- Voxelwise Fitting with lmfit ---
fitted_voxels = 0
model = Model(ivim_dki_model, independent_vars=['b'])

for x in range(shape[0]):
    for y in range(shape[1]):
        for z in range(shape[2]):
            signal = data[x, y, z, :]
            if np.all(signal == 0) or np.any(signal < 0) or np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
                continue
            # Normalize signal
            signal = signal / signal[0]
            # Skip if any value is nan/inf after normalization
            if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
                continue
            params = Parameters()
            params.add('f', value=p0[0], min=lower_bound[0], max=upper_bound[0])
            params.add('D_star', value=p0[1], min=lower_bound[1], max=upper_bound[1])
            params.add('D_slow', value=p0[2], min=lower_bound[2], max=upper_bound[2])
            params.add('k', value=p0[3], min=lower_bound[3], max=upper_bound[3])
            try:
                result = model.fit(signal, b=bvals, params=params, method='leastsq')
                f_map_voxelwise[x, y, z] = result.params['f'].value
                Dstar_map_voxelwise[x, y, z] = result.params['D_star'].value
                Dslow_map_voxelwise[x, y, z] = result.params['D_slow'].value
                k_map_voxelwise[x, y, z] = result.params['k'].value
                fitted_voxels += 1
            except Exception:
                continue

print("Total fitted voxels:", fitted_voxels)
print("f_map_voxelwise: min", np.nanmin(f_map_voxelwise), "max", np.nanmax(f_map_voxelwise))

# Save nii files (optional)
'''
output_dir = "/Users/ayush/Desktop/project-internsip/Results/final_lmfit.nii"
os.makedirs(output_dir, exist_ok=True)
affine = snr60_data.affine
nb.Nifti1Image(f_map_voxelwise, affine).to_filename(os.path.join(output_dir, "fitted_f_map.nii.gz"))
nb.Nifti1Image(Dstar_map_voxelwise, affine).to_filename(os.path.join(output_dir, "fitted_Dstar_map.nii.gz"))
nb.Nifti1Image(Dslow_map_voxelwise, affine).to_filename(os.path.join(output_dir, "fitted_Dslow_map.nii.gz"))
nb.Nifti1Image(k_map_voxelwise, affine).to_filename(os.path.join(output_dir, "fitted_k_map.nii.gz"))
'''

# Loading the reference and toolbox maps
ref_f = nb.load(ref_f_path).get_fdata()
f_toolbox = nb.load(toolbox_f_path).get_fdata()
d_toolbox = nb.load(toolbox_d_path).get_fdata()
dstar_toolbox = nb.load(toolbox_dstar_path).get_fdata()
k_toolbox = nb.load(toolbox_k_path).get_fdata()

# Error metrics function (same as before)
def error_metrics(ref, est, label):
    mask = np.isfinite(ref) & np.isfinite(est) & (ref != 0)
    ref_vals = ref[mask]
    est_vals = est[mask]
    rmse = np.sqrt(np.mean((est_vals - ref_vals) ** 2))
    rmse_norm = (rmse / np.mean(ref_vals)) * 100
    rel_bias = np.mean((est_vals - ref_vals) / ref_vals) * 100
    rel_param = np.mean(est_vals / ref_vals)
    print(f"{label} RMSE (normalized, %): {rmse_norm:.2f}")
    print(f"{label} Relative Bias (%): {rel_bias:.2f}")
    print(f"{label} Relative Parameter: {rel_param:.4f}")
    return mask

print("---- Error metrics for f ----")
mask_voxelwise = error_metrics(ref_f, f_map_voxelwise, "Voxelwise f")
mask_toolbox = error_metrics(ref_f, f_toolbox, "Toolbox f")

# --- Visualization ---
mid_slice = ref_f.shape[2] // 2
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(ref_f[:, :, mid_slice], cmap='jet', vmin=0, vmax=1)
plt.title('Reference f map')
plt.axis('off')
plt.colorbar()
plt.subplot(1, 3, 2)
plt.imshow(f_map_voxelwise[:, :, mid_slice], cmap='jet', vmin=0, vmax=1)
plt.title('Voxelwise f map')
plt.axis('off')
plt.colorbar()
plt.subplot(1, 3, 3)
plt.imshow(f_toolbox[:, :, mid_slice], cmap='jet', vmin=0, vmax=1)
plt.title('Toolbox f map')
plt.axis('off')
plt.colorbar()
plt.tight_layout()
plt.show()

#AIC/AICc calculation Function:
def sim_ivim_dki(bvals, D, Dstar, f, k):
    y_pred = np.zeros(D.shape + (len(bvals),))
    for idx, b in enumerate(bvals):
        exp1 = np.exp(np.clip(-b * Dstar, -100, 100))
        exp2 = np.exp(np.clip(-b * D + (1 / 6) * (b ** 2) * (D ** 2) * k, -100, 100))
        y_pred[..., idx] = (f * exp1) + (1 - f) * exp2
    return y_pred

parameters = 4
n = len(bvals)

def calc_aic_aicc(y_data, y_pred, mask):
    aic_map = np.full(y_data.shape[:3], np.nan)
    for i in range(y_data.shape[0]):
        for j in range(y_data.shape[1]):
            for k in range(y_data.shape[2]):
                if mask[i, j, k]:
                    residuals = y_data[i, j, k, :] - y_pred[i, j, k, :]
                    RSS = np.sum(residuals ** 2)
                    if RSS > 0:
                        aic_map[i, j, k] = 2 * parameters + n * np.log(RSS / n)
    aic = np.nanmean(aic_map)
    aicc = aic + (2 * parameters * (parameters + 1) / (n - parameters - 1))
    return aic, aicc

# Original data
y_data = data

# Voxelwise
d_voxelwise = Dslow_map_voxelwise
dstar_voxelwise = Dstar_map_voxelwise
k_voxelwise = k_map_voxelwise
y_pred_voxelwise = sim_ivim_dki(bvals, d_voxelwise, dstar_voxelwise, f_map_voxelwise, k_voxelwise)
aic_voxelwise, aicc_voxelwise = calc_aic_aicc(y_data, y_pred_voxelwise, mask_voxelwise)
print(f"Voxelwise  AIC: {aic_voxelwise:.2f}")
print(f"Voxelwise  AICc: {aicc_voxelwise:.2f}")

# Toolbox
y_pred_toolbox = sim_ivim_dki(bvals, d_toolbox, dstar_toolbox, f_toolbox, k_toolbox)
aic_toolbox, aicc_toolbox = calc_aic_aicc(y_data, y_pred_toolbox, mask_toolbox)
print(f"Toolbox  AIC: {aic_toolbox:.2f}")
print(f"Toolbox  AICc: {aicc_toolbox:.2f}")