import os
import numpy as np
import nibabel as nb
from scipy.optimize import curve_fit
from scipy.special import expit, logit
import matplotlib.pyplot as plt

# 1. Load the NIfTI file and check data
nii_file_path = "/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_SNR60/Data-1_Simulation-III_SNR-60.nii"
nifti_image = nb.load(nii_file_path)
image_data = nifti_image.get_fdata()
print(f"NIfTI image shape: {nifti_image.shape}")

if image_data.ndim != 4:
    raise ValueError(f"Expected 4D data, got {image_data.shape}. Please check your input file.")

b_values_array = np.array([0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000])

# 2. Load toolbox parameter maps (for fixed values)
toolbox_f_path = "/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-III_SNR-60_f.nii"
toolbox_Dstar_path = "/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-III_SNR-60_D_star.nii"
toolbox_Dslow_path = "/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-III_SNR-60_D.nii"
toolbox_k_path = "/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-III_SNR-60_k.nii"

f_toolbox = nb.load(toolbox_f_path).get_fdata()
Dstar_toolbox = nb.load(toolbox_Dstar_path).get_fdata()
Dslow_toolbox = nb.load(toolbox_Dslow_path).get_fdata()
k_toolbox = nb.load(toolbox_k_path).get_fdata()

# Use mean values from toolbox for fixed parameters
Dstar_fixed = np.nanmean(Dstar_toolbox)
Dslow_fixed = np.nanmean(Dslow_toolbox)
k_fixed = np.nanmean(k_toolbox)

# 3. Define the IVIM-DKI model for f-only fitting
def ivim_dki_fonly(b, logit_f):
    f = expit(logit_f)
    exp1 = np.exp(np.clip(-b * Dstar_fixed, -100, 100))
    exp2 = np.exp(np.clip(-b * Dslow_fixed + (1/6) * (b ** 2) * (Dslow_fixed ** 2) * k_fixed, -100, 100))
    return f * exp1 + (1 - f) * exp2

# 4. Voxelwise fitting for f only
image_shape_3d = image_data.shape[:3]
f_map = np.full(image_shape_3d, np.nan)
p0 = [logit(0.13)]
bounds = ([logit(0.001)], [logit(1)])

num_voxels_x, num_voxels_y, num_voxels_z = image_shape_3d
fit_fail_count = 0

print("Starting voxelwise f-only fitting...")
for x_idx in range(num_voxels_x):
    for y_idx in range(num_voxels_y):
        for z_idx in range(num_voxels_z):
            voxel_signal_decay = image_data[x_idx, y_idx, z_idx, :]
            if np.any(voxel_signal_decay > 0) and voxel_signal_decay[0] != 0:
                y = voxel_signal_decay / voxel_signal_decay[0]
                if np.isnan(y).any() or np.isinf(y).any():
                    continue
                try:
                    popt, _ = curve_fit(
                        ivim_dki_fonly,
                        b_values_array,
                        y,
                        p0=p0,
                        bounds=bounds,
                        maxfev=10000
                    )
                    f_map[x_idx, y_idx, z_idx] = expit(popt[0])
                except Exception:
                    fit_fail_count += 1
                    continue

print("Voxelwise f-only fitting complete.")
num_fitted_voxels = np.sum(~np.isnan(f_map))
print(f"Number of voxels fitted: {num_fitted_voxels} / {f_map.size}")
print(f"Total fit failures: {fit_fail_count}")

# 5. Load reference f map
ref_f_path = "/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_f_map.nii"
ref_f = nb.load(ref_f_path).get_fdata()

# 6. Mask for error metrics
mask = np.isfinite(ref_f) & np.isfinite(f_toolbox) & np.isfinite(f_map)

# 7. Error metrics for f-maps
def print_f_metrics(ref_data, est_data, mask, label):
    ref_vals = ref_data[mask]
    est_vals = est_data[mask]
    rmse = np.sqrt(np.mean((est_vals - ref_vals) ** 2))
    rmse_norm = (rmse / np.mean(ref_vals)) * 100
    rel_bias = (np.mean((est_vals - ref_vals) / (ref_vals)) * 100)
    rel_param = np.mean(est_vals / ref_vals)
    print(f"{label} RMSE (normalized, %): {rmse_norm:.2f}")
    print(f"{label} Relative Bias (%): {rel_bias:.2f}")
    print(f"{label} Relative Parameter: {rel_param:.4f}")

print_f_metrics(ref_f, f_toolbox, mask, "Toolbox f")
print_f_metrics(ref_f, f_map, mask, "Voxelwise f")

# 8. AIC and AICc calculation 
def sim_ivim_dki(b_val, est_d, est_dp, est_f, est_k):
    y_predicted = np.zeros(est_f.shape + (len(b_val),))
    for k in range(len(b_val)):
        y_predicted[..., k] = (est_f * np.exp(-b_val[k] * est_dp)) + \
                              (1 - est_f) * np.exp(-b_val[k] * est_d + (1 / 6) * est_k * (b_val[k] ** 2) * (est_d ** 2))
    return y_predicted

# For AIC/AICc, use the same fixed D, D*, k for voxelwise f
est_d = Dslow_fixed
est_dstar = Dstar_fixed
est_k = k_fixed

# Prepare arrays for AIC calculation
y_data = image_data
parameters = 1  # Only f is fitted
n = len(b_values_array)

# Toolbox AIC/AICc
y_pred_toolbox = sim_ivim_dki(b_values_array, Dslow_toolbox, Dstar_toolbox, f_toolbox, k_toolbox)
aic_toolbox_map = np.zeros(f_toolbox.shape)
for i in range(f_toolbox.shape[0]):
    for j in range(f_toolbox.shape[1]):
        for k in range(f_toolbox.shape[2]):
            if not (np.isfinite(f_toolbox[i, j, k]) and np.isfinite(Dstar_toolbox[i, j, k]) and np.isfinite(Dslow_toolbox[i, j, k]) and np.isfinite(k_toolbox[i, j, k])):
                aic_toolbox_map[i, j, k] = np.nan
                continue
            residuals = y_data[i, j, k, :] - y_pred_toolbox[i, j, k, :]
            RSS = np.sum(residuals ** 2)
            if RSS > 0:
                aic = 2 * parameters + n * np.log(RSS / n)
                aicc = aic + (2 * parameters * (parameters + 1)) / (n - parameters - 1)
                aic_toolbox_map[i, j, k] = aicc
            else:
                aic_toolbox_map[i, j, k] = np.nan

# Voxelwise AIC/AICc (using fixed D, D*, k)
y_pred_voxelwise = sim_ivim_dki(b_values_array, est_d, est_dstar, f_map, est_k)
aic_voxelwise_map = np.zeros(f_map.shape)
for i in range(f_map.shape[0]):
    for j in range(f_map.shape[1]):
        for k in range(f_map.shape[2]):
            if not np.isfinite(f_map[i, j, k]):
                aic_voxelwise_map[i, j, k] = np.nan
                continue
            residuals = y_data[i, j, k, :] - y_pred_voxelwise[i, j, k, :]
            RSS = np.sum(residuals ** 2)
            if RSS > 0:
                aic = 2 * parameters + n * np.log(RSS / n)
                aicc = aic + (2 * parameters * (parameters + 1)) / (n - parameters - 1)
                aic_voxelwise_map[i, j, k] = aicc
            else:
                aic_voxelwise_map[i, j, k] = np.nan

print("Toolbox f AICc (mean):", np.nanmean(aic_toolbox_map))
print("Voxelwise f AICc (mean):", np.nanmean(aic_voxelwise_map))

# 9. Visualization: 3 f-maps as subplots, correct color range
mid_slice = f_map.shape[2] // 2

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(ref_f[:, :, mid_slice], cmap='jet', vmin=0, vmax=0.5)
plt.title('Reference f map')
plt.axis('off')
plt.colorbar()
plt.subplot(1, 3, 2)
plt.imshow(f_toolbox[:, :, mid_slice], cmap='jet', vmin=0, vmax=0.5)
plt.title('Toolbox f map')
plt.axis('off')
plt.colorbar()
plt.subplot(1, 3, 3)
plt.imshow(f_map[:, :, mid_slice], cmap='jet', vmin=0, vmax=0.5)
plt.title('Voxelwise f map')
plt.axis('off')
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure()
plt.hist(f_map[np.isfinite(f_map)].flatten(), bins=50)
plt.title("Histogram of fitted voxelwise f values")
plt.xlabel("f")
plt.ylabel("Count")
plt.show()