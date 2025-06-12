import os
import numpy as np
import nibabel as nb
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 1. Define the IVIM-DKI model in log scale (all parameters)
def ivim_dki_model_log(b_values, log_f, log_D_star, log_D_slow, log_k):
    f = np.exp(log_f)
    D_star = np.exp(log_D_star)
    D_slow = np.exp(log_D_slow)
    k = np.exp(log_k)
    exp1 = np.exp(np.clip(-b_values * D_star, -100, 100))
    exp2 = np.exp(np.clip(-b_values * D_slow + (1/6) * (b_values ** 2) * (D_slow ** 2) * k, -100, 100))
    return f * exp1 + (1 - f) * exp2

def safe_for_nifti(arr):
    return np.nan_to_num(arr, nan=0.0)

# 2. Load the NIfTI file and check data
nii_file_path = "/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_SNR60/Data-1_Simulation-III_SNR-60.nii"
nifti_image = nb.load(nii_file_path)
image_data = nifti_image.get_fdata()
print(f"NIfTI image shape: {nifti_image.shape}")

if image_data.ndim != 4:
    raise ValueError(f"Expected 4D data, got {image_data.shape}. Please check your input file.")

b_values_array = np.array([0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000])

# 3. Prepare output arrays for parameter maps and AIC
image_shape_3d = image_data.shape[:3]
f_map = np.full(image_shape_3d, np.nan)
D_star_map = np.full(image_shape_3d, np.nan)
D_slow_map = np.full(image_shape_3d, np.nan)
k_map = np.full(image_shape_3d, np.nan)
aic_map = np.full(image_shape_3d, np.nan)

# 4. Set bounds and initial guesses in log scale
p0 = [np.log(0.13), np.log(0.013), np.log(0.23), np.log(1.1)]
bounds = ([np.log(0.01), np.log(0.0001), np.log(0.001), np.log(0.01)],
          [np.log(0.3), np.log(0.5), np.log(1), np.log(3)])

num_voxels_x, num_voxels_y, num_voxels_z = image_shape_3d

# 5. Main voxel-wise fitting loop (log scale)
fit_fail_count = 0
for x_idx in range(num_voxels_x):
    for y_idx in range(num_voxels_y):
        for z_idx in range(num_voxels_z):
            voxel_signal_decay = image_data[x_idx, y_idx, z_idx, :]
            if np.any(voxel_signal_decay > 0) and voxel_signal_decay[0] != 0:
                y = voxel_signal_decay / voxel_signal_decay[0]
                if np.isnan(y).any() or np.isinf(y).any():
                    fit_fail_count += 1
                    continue
                try:
                    fitted_params, _ = curve_fit(
                        ivim_dki_model_log,
                        b_values_array,
                        y,
                        p0=p0,
                        bounds=bounds,
                    )
                    y_hat = ivim_dki_model_log(b_values_array, *fitted_params)
                    residuals = y - y_hat
                    rss = np.sum(residuals ** 2)
                    n = len(y)
                    k_param = len(fitted_params)
                    aic = 2 * k_param + n * np.log(rss / n) if rss > 0 and n > 0 else np.nan
                    # Store fitted parameters (convert back from log)
                    f_map[x_idx, y_idx, z_idx] = np.exp(fitted_params[0])
                    D_star_map[x_idx, y_idx, z_idx] = np.exp(fitted_params[1])
                    D_slow_map[x_idx, y_idx, z_idx] = np.exp(fitted_params[2])
                    k_map[x_idx, y_idx, z_idx] = np.exp(fitted_params[3])
                    aic_map[x_idx, y_idx, z_idx] = aic
                except Exception as e:
                    fit_fail_count += 1
                    continue

print("Voxel-wise IVIM-DKI log fitting complete.")
num_fitted_voxels = np.sum(~np.isnan(f_map))
print(f"Number of voxels fitted: {num_fitted_voxels} / {f_map.size}")
print(f"Total fit failures: {fit_fail_count}")

# 6. Load reference and toolbox f maps
ref_f_path = "/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_f_map.nii"
ref_f = nb.load(ref_f_path).get_fdata()
toolbox_f_path = "/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-III_SNR-60_f.nii"
f_toolbox = nb.load(toolbox_f_path).get_fdata()

# 7. Mask for error metrics
mask = np.isfinite(ref_f) & np.isfinite(f_toolbox) & np.isfinite(f_map)

# 8. Error metrics for f-maps
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

# 9. Visualization: 3 f-maps as subplots, correct color range
mid_slice = f_map.shape[2] // 2

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(ref_f[:, :, mid_slice], cmap='jet', vmin=0, vmax=0.3)
plt.title('Reference f map')
plt.axis('off')
plt.colorbar()
plt.subplot(1, 3, 2)
plt.imshow(f_toolbox[:, :, mid_slice], cmap='jet', vmin=0, vmax=0.3)
plt.title('Toolbox f map')
plt.axis('off')
plt.colorbar()
plt.subplot(1, 3, 3)
plt.imshow(f_map[:, :, mid_slice], cmap='jet', vmin=0, vmax=0.3)
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