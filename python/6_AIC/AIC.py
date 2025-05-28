import numpy as np
import nibabel as nb
from scipy.optimize import curve_fit

def ivim_model(b, f, D_star, D_slow):
    return f * np.exp(-b * D_star) + (1 - f) * np.exp(-b * D_slow)

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

parameter_bounds = ([0, 0.005, 0], [0.5, 0.1, 0.005])
initial_guesses = [0.1, 0.015, 0.001]

num_voxels_x, num_voxels_y, num_voxels_z = image_shape_3d

for x in range(num_voxels_x):
    for y in range(num_voxels_y):
        for z in range(num_voxels_z):
            voxel_signal = image_data[x, y, z, :]
            if np.any(voxel_signal > 0) and voxel_signal[0] != 0:
                norm_signal = voxel_signal / voxel_signal[0]
                try:
                    params, _ = curve_fit(
                        ivim_model,
                        b_values_array,
                        norm_signal,
                        p0=initial_guesses,
                        bounds=parameter_bounds
                    )
                    pred = ivim_model(b_values_array, *params)
                    rss = np.sum((norm_signal - pred) ** 2)
                    n = len(norm_signal)
                    k = len(params)
                    aic = 2 * k + n * np.log(rss / n) if rss > 0 and n > 0 else np.nan
                    f_map[x, y, z] = params[0]
                    D_star_map[x, y, z] = params[1]
                    D_slow_map[x, y, z] = params[2]
                    aic_map[x, y, z] = aic
                except Exception:
                    pass

print("Voxel-wise IVIM fitting complete.")
num_fitted = np.sum(~np.isnan(f_map))
print(f"Number of voxels fitted: {num_fitted} out of {f_map.size}")

if num_fitted > 0:
    print(f"f_map min/max: {np.nanmin(f_map):.4f} / {np.nanmax(f_map):.4f}")
    print(f"D_star_map min/max: {np.nanmin(D_star_map):.4f} / {np.nanmax(D_star_map):.4f}")
    print(f"D_slow_map min/max: {np.nanmin(D_slow_map):.6f} / {np.nanmax(D_slow_map):.6f}")
    print(f"AIC_map min/max: {np.nanmin(aic_map):.2f} / {np.nanmax(aic_map):.2f}")
else:
    print("No voxels were successfully fitted. Check data, b-values, bounds, or initial guesses.")