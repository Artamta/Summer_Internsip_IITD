import numpy as np
import nibabel as nb
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

# IVIM-DKI model in log-parameter space
def ivim_dki_model_log(b, log_f, log_D_star, log_D_slow, log_k):
    f = np.exp(log_f)
    D_star = np.exp(log_D_star)
    D_slow = np.exp(log_D_slow)
    k = np.exp(log_k)
    exp1 = np.exp(-b * D_star)
    exp2 = np.exp(-b * D_slow + (1/6) * (b ** 2) * (D_slow ** 2) * k)
    return f * exp1 + (1 - f) * exp2

# Paths
snr60 = "/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_SNR60/Data-1_Simulation-III_SNR-60.nii"
ref_f_path = "/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_f_map.nii"
toolbox_f_path = "/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-III_SNR-60_f.nii"
toolbox_d_path = "/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-III_SNR-60_D.nii"
toolbox_dstar_path = "/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-III_SNR-60_D_star.nii"
toolbox_k_path = "/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-III_SNR-60_k.nii"

# Fitting settings (log scale)
bvals = np.array([0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000])
p0 = [0.1, 0.01, 0.001, 0.5]  # f, D*, D, k (all positive)
lower_bound = [0.0001, 0.0005, 0.0001, 0.01]
upper_bound = [1, 0.1, 0.003, 3]
log_p0 = np.log(p0)
log_bounds = (np.log(lower_bound), np.log(upper_bound))

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
                log_popt, _ = curve_fit(
                    ivim_dki_model_log, bvals, signal, p0=log_p0, bounds=log_bounds, maxfev=10000
                )
                popt = np.exp(log_popt)
                f_map_voxelwise[x, y, z] = popt[0]
                Dstar_map_voxelwise[x, y, z] = popt[1]
                Dslow_map_voxelwise[x, y, z] = popt[2]
                k_map_voxelwise[x, y, z] = popt[3]
                fitted_voxels += 1
            except Exception:
                continue

print("Total fitted voxels:", fitted_voxels)
print("f_map_voxelwise: min", np.nanmin(f_map_voxelwise), "max", np.nanmax(f_map_voxelwise))

# Loading the reference and toolbox maps
ref_f = nb.load(ref_f_path).get_fdata()
f_toolbox = nb.load(toolbox_f_path).get_fdata()
d_toolbox = nb.load(toolbox_d_path).get_fdata()
dstar_toolbox = nb.load(toolbox_dstar_path).get_fdata()
k_toolbox = nb.load(toolbox_k_path).get_fdata()

# Error metrics function
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
plt.title('Voxelwise f map (log fit)')
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