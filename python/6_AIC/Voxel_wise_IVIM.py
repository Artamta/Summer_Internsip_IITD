import numpy as np
import nibabel as nb
from scipy.optimize import curve_fit

def ivim_model(b_values, f, D_star, D_slow):
    return f * np.exp(-b_values * D_star) + (1 - f) * np.exp(-b_values * D_slow)

nii_file_path = "/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_SNR15/Data-1_Simulation-III_SNR-15.nii"
nifti_image = nb.load(nii_file_path)
print(f"NIfTI image shape: {nifti_image.shape}")
image_data = nifti_image.get_fdata()
b_values_array = np.array([0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000])

image_shape_3d = image_data.shape[:3]
f_map = np.full(image_shape_3d, np.nan)
D_star_map = np.full(image_shape_3d, np.nan)
D_slow_map = np.full(image_shape_3d, np.nan)
aic_map = np.full(image_shape_3d, np.nan)

parameter_bounds = ([0.0001, 0.0001, 0.001], [0.05, 0.5, 1])
initial_guesses = [0.00013, 0.013, 0.23]

num_voxels_x, num_voxels_y, num_voxels_z = image_shape_3d

for x_idx in range(num_voxels_x):
    for y_idx in range(num_voxels_y):
        for z_idx in range(num_voxels_z):
            voxel_signal_decay = image_data[x_idx, y_idx, z_idx, :]
            if np.any(voxel_signal_decay > 0) and voxel_signal_decay[0] != 0:
                # y: measured normalized signal
                y = voxel_signal_decay / voxel_signal_decay[0]
                # Fit IVIM model to get parameters
                fitted_params, _ = curve_fit(
                    ivim_model,
                    b_values_array,
                    y,
                    p0=initial_guesses,
                    bounds=parameter_bounds,
                )
                # y_hat: predicted signal from fitted model
                y_hat = ivim_model(b_values_array, *fitted_params)
                # Calculate residuals and RSS
                residuals = y - y_hat
                rss = np.sum(residuals ** 2)
                # Calculate AIC
                n = len(y)
                k = len(fitted_params)
                if rss > 0 and n > 0:
                    aic = 2 * k + n * np.log(rss / n)
                else:
                    aic = np.nan
                # Store results
                f_map[x_idx, y_idx, z_idx] = fitted_params[0]
                D_star_map[x_idx, y_idx, z_idx] = fitted_params[1]
                D_slow_map[x_idx, y_idx, z_idx] = fitted_params[2]
                aic_map[x_idx, y_idx, z_idx] = aic

print("Voxel-wise IVIM fitting complete.")
num_fitted_voxels = np.sum(~np.isnan(f_map))
print(f"Number of voxels where fitting was attempted and did not fail before storing: {num_fitted_voxels} out of {f_map.size}")

if num_fitted_voxels > 0:
    print(f"f_map min/max: {np.nanmin(f_map):.4f} / {np.nanmax(f_map):.4f}")
    print(f"D_star_map min/max: {np.nanmin(D_star_map):.4f} / {np.nanmax(D_star_map):.4f}")
    print(f"D_slow_map min/max: {np.nanmin(D_slow_map):.6f} / {np.nanmax(D_slow_map):.6f}")
    print(f"AIC_map min/max: {np.nanmin(aic_map):.2f} / {np.nanmax(aic_map):.2f}")
else:
    print("No voxels were successfully fitted. Check data, b-values, bounds, or initial guesses.")