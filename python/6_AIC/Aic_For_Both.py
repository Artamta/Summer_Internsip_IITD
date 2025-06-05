import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import expit, logit

# --- File paths ---
nii_file_path = "/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_SNR60/Data-1_Simulation-III_SNR-60.nii"
b_values_array = np.array([0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000])

toolbox_dir = "/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps"
my_dir = "/Users/ayush/Desktop/project-internsip/Results/6_Aic_CALC"
ref_f_path = "/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_f_map.nii"

# --- Load parameter maps ---
f_toolbox = nb.load(f"{toolbox_dir}/Data-1_Simulation-III_SNR-60_f.nii").get_fdata()
Dstar_toolbox = nb.load(f"{toolbox_dir}/Data-1_Simulation-III_SNR-60_D_star.nii").get_fdata()
Dslow_toolbox = nb.load(f"{toolbox_dir}/Data-1_Simulation-III_SNR-60_D.nii").get_fdata()
k_toolbox = nb.load(f"{toolbox_dir}/Data-1_Simulation-III_SNR-60_k.nii").get_fdata()

# --- Load original 4D data ---
image_data = nb.load(nii_file_path).get_fdata()
shape = image_data.shape[:3]

# --- Logit/log-scale IVIM-DKI model for fitting ---
def ivim_dki_model_log(b, logit_f, log_D_star, log_D_slow, log_k):
    f = expit(logit_f)  # ensures 0 < f < 1
    D_star = np.exp(log_D_star)
    D_slow = np.exp(log_D_slow)
    k = np.exp(log_k)
    exp1 = np.exp(np.clip(-b * D_star, -100, 100))
    exp2 = np.exp(np.clip(-b * D_slow + (1/6) * (b ** 2) * (D_slow ** 2) * k, -100, 100))
    return f * exp1 + (1 - f) * exp2

# --- Tighter bounds and lower initial guess for f ---
p0 = [logit(0.07), np.log(0.013), np.log(0.23), np.log(1.1)]
bounds = ([logit(0.01), np.log(0.0001), np.log(0.001), np.log(0.01)],
          [logit(0.3), np.log(0.5), np.log(1), np.log(3)])

# --- Voxelwise logit/log-scale fitting with RSS masking ---
f_my = np.full(shape, np.nan)
Dstar_my = np.full(shape, np.nan)
Dslow_my = np.full(shape, np.nan)
k_my = np.full(shape, np.nan)

print("Starting voxelwise logit/log-scale fitting...")
for x in range(shape[0]):
    for y in range(shape[1]):
        for z in range(shape[2]):
            signal = image_data[x, y, z, :]
            if np.any(signal > 0) and signal[0] != 0:
                y_norm = signal / signal[0]
                if np.any(np.isnan(y_norm)) or np.any(np.isinf(y_norm)):
                    continue
                try:
                    popt, _ = curve_fit(
                        ivim_dki_model_log, b_values_array, y_norm, p0=p0, bounds=bounds, maxfev=10000
                    )
                    y_fit = ivim_dki_model_log(b_values_array, *popt)
                    rss = np.sum((y_norm - y_fit) ** 2)
                    # Mask out bad fits (tune threshold as needed)
                    if rss > 0.1:
                        continue
                    f_val = np.clip(expit(popt[0]), 0.01, 0.3)
                    f_my[x, y, z] = f_val
                    Dstar_my[x, y, z] = np.exp(popt[1])
                    Dslow_my[x, y, z] = np.exp(popt[2])
                    k_my[x, y, z] = np.exp(popt[3])
                except Exception:
                    continue
print("Voxelwise fitting complete.")

# --- Save voxelwise parameter maps (optional) ---
nb.Nifti1Image(f_my, nb.load(nii_file_path).affine).to_filename(f"{my_dir}/f_logfit.nii.gz")
nb.Nifti1Image(Dslow_my, nb.load(nii_file_path).affine).to_filename(f"{my_dir}/D_logfit.nii.gz")
nb.Nifti1Image(Dstar_my, nb.load(nii_file_path).affine).to_filename(f"{my_dir}/Dstar_logfit.nii.gz")
nb.Nifti1Image(k_my, nb.load(nii_file_path).affine).to_filename(f"{my_dir}/k_logfit.nii.gz")

# --- Mask for error metrics ---
signal_mask = image_data[..., 0] > 0
param_mask_toolbox = np.isfinite(f_toolbox) & np.isfinite(Dstar_toolbox) & np.isfinite(Dslow_toolbox) & np.isfinite(k_toolbox)
param_mask_my = np.isfinite(f_my) & np.isfinite(Dstar_my) & np.isfinite(Dslow_my) & np.isfinite(k_my)
mask = signal_mask & param_mask_toolbox & param_mask_my

# --- Error metrics for f-maps (no normalization) ---
def error_metrics(ref, est, mask):
    ref_vals = ref[mask]
    est_vals = est[mask]
    rmse = np.sqrt(np.mean((est_vals - ref_vals) ** 2))
    rmse_norm = (rmse / np.mean(ref_vals)) * 100
    rel_bias = np.mean((est_vals - ref_vals) / ref_vals) * 100
    rel_param = np.mean(est_vals / ref_vals)
    return rmse_norm, rel_bias, rel_param

# Reference f_map
ref_f_data = nb.load(ref_f_path).get_fdata()

rmse_toolbox, bias_toolbox, relp_toolbox = error_metrics(ref_f_data, f_toolbox, mask)
rmse_my, bias_my, relp_my = error_metrics(ref_f_data, f_my, mask)

print(f"Toolbox f RMSE (normalized, %): {rmse_toolbox:.2f}")
print(f"Toolbox f Relative Bias (%): {bias_toolbox:.2f}")
print(f"Toolbox f Relative Parameter: {relp_toolbox:.4f}")
print(f"Voxelwise f RMSE (normalized, %): {rmse_my:.2f}")
print(f"Voxelwise f Relative Bias (%): {bias_my:.2f}")
print(f"Voxelwise f Relative Parameter: {relp_my:.4f}")

# --- AIC and AICc calculation for f-maps (matches supervisor's reference) ---
def sim_ivim_dki(b_val, est_d, est_dp, est_f, est_k):
    y_pred = np.zeros(est_f.shape + (len(b_val),))
    for idx, b in enumerate(b_val):
        exp1 = np.exp(np.clip(-b * est_dp, -100, 100))
        exp2 = np.exp(np.clip(-b * est_d + (1 / 6) * est_k * (b ** 2) * (est_d ** 2), -100, 100))
        y_pred[..., idx] = (est_f * exp1) + (1 - est_f) * exp2
    return y_pred

def calc_aic_aicc(D, Dstar, f, k, y_data, bvals):
    y_pred = sim_ivim_dki(bvals, D, Dstar, f, k)
    mask = (f > 0)
    aic_map = np.full(f.shape, np.nan)
    n = len(bvals)
    k_param = 4
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            for l in range(f.shape[2]):
                if mask[i, j, l]:
                    residuals = y_data[i, j, l, :] - y_pred[i, j, l, :]
                    RSS = np.sum(residuals ** 2)
                    if RSS > 0:
                        aic = 2 * k_param + n * np.log(RSS / n)
                        aicc = aic + (2 * k_param * (k_param + 1)) / (n - k_param - 1)
                        aic_map[i, j, l] = aicc
    return aic_map

aic_toolbox = calc_aic_aicc(Dslow_toolbox, Dstar_toolbox, f_toolbox, k_toolbox, image_data, b_values_array)
aic_my = calc_aic_aicc(Dslow_my, Dstar_my, f_my, k_my, image_data, b_values_array)

print("Toolbox f AICc (mean):", np.nanmean(aic_toolbox))
print("Voxelwise f AICc (mean):", np.nanmean(aic_my))

# --- Visualization: 3 f-maps as subplots, correct color range ---
mid_slice = shape[2] // 2
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(ref_f_data[:, :, mid_slice], cmap='jet', vmin=0, vmax=0.3)
plt.title('Reference f map')
plt.axis('off')
plt.colorbar()
plt.subplot(1, 3, 2)
plt.imshow(f_toolbox[:, :, mid_slice], cmap='jet', vmin=0, vmax=0.3)
plt.title('Toolbox f map')
plt.axis('off')
plt.colorbar()
plt.subplot(1, 3, 3)
plt.imshow(f_my[:, :, mid_slice], cmap='jet', vmin=0, vmax=0.3)
plt.title('Voxelwise f map')
plt.axis('off')
plt.colorbar()
plt.tight_layout()
plt.show()