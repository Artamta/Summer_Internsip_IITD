import os
import numpy as np
import nibabel as nb
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def ivim_dki_model(b_values, f, D_star, D_slow, k):
    # IVIM-DKI hybrid model: S/S0 = f*exp(-b*D*) + (1-f)*exp(-b*D + (1/6)*b^2*D^2*k)
    # Clamp exponentials to avoid overflow and clamp output to avoid huge values
    exp1 = np.exp(np.clip(-b_values * D_star, -100, 100))
    exp2 = np.exp(np.clip(-b_values * D_slow + (1/6) * (b_values ** 2) * (D_slow ** 2) * k, -100, 100))
    result = f * exp1 + (1 - f) * exp2
    return np.clip(result, 0, 2)

# Load NIfTI file
nii_file_path = "/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_SNR60/Data-1_Simulation-III_SNR-60.nii"
nifti_image = nb.load(nii_file_path)
print(f"NIfTI image shape: {nifti_image.shape}")
image_data = nifti_image.get_fdata()

# Check for NaNs or Infs in the data
print("Any NaNs in data?", np.isnan(image_data).any())
print("Any Infs in data?", np.isinf(image_data).any())

# Check the shape (should be 4D)
if image_data.ndim != 4:
    raise ValueError(f"Expected 4D data, got {image_data.shape}. Please check your input file.")

b_values_array = np.array([0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000])

# Print a sample voxel's signal for debugging
print("Sample voxel [55,55,5,:] signal:", image_data[55, 55, 5, :])

# Check b=0 image statistics
b0 = image_data[..., 0]
print("b=0 min:", np.min(b0), "max:", np.max(b0), "mean:", np.mean(b0))
print("Number of voxels with b=0 == 0:", np.sum(b0 == 0))

image_shape_3d = image_data.shape[:3]
f_map = np.full(image_shape_3d, np.nan)
D_star_map = np.full(image_shape_3d, np.nan)
D_slow_map = np.full(image_shape_3d, np.nan)
k_map = np.full(image_shape_3d, np.nan)
aic_map = np.full(image_shape_3d, np.nan)

# Bounds and initial guesses for [f, D*, D, k] (as per supervisor)
parameter_bounds = ([0.0001, 0.0001, 0.001, 0.01], [0.05, 0.5, 1, 3])
# Use much smaller initial guesses to avoid overflow
initial_guesses = [0.01, 0.01, 0.01, 0.1]

num_voxels_x, num_voxels_y, num_voxels_z = image_shape_3d

# Try fitting a single voxel interactively for debugging
test_voxel = (55, 55, 5)
signal = image_data[test_voxel[0], test_voxel[1], test_voxel[2], :]
if signal[0] != 0:
    y_test = signal / signal[0]
    print("y_test (normalized signal):", y_test)
    model_out = ivim_dki_model(b_values_array, *initial_guesses)
    print("Model output with initial guess:", model_out)
    print("Any NaNs in model output?", np.isnan(model_out).any())
    print("Any Infs in model output?", np.isinf(model_out).any())
    try:
        popt, _ = curve_fit(
            ivim_dki_model,
            b_values_array,
            y_test,
            p0=initial_guesses,
            bounds=parameter_bounds,
            maxfev=10000
        )
        print(f"Single voxel fit succeeded at {test_voxel}, params: {popt}")
    except Exception as e:
        print(f"Single voxel fit failed at {test_voxel}: {e}")

# Main voxel-wise fitting loop with debug prints
fit_fail_count = 0
for x_idx in range(num_voxels_x):
    for y_idx in range(num_voxels_y):
        for z_idx in range(num_voxels_z):
            voxel_signal_decay = image_data[x_idx, y_idx, z_idx, :]
            if np.any(voxel_signal_decay > 0) and voxel_signal_decay[0] != 0:
                y = voxel_signal_decay / voxel_signal_decay[0]
                # Debug: Check for NaNs or Infs in normalized signal
                if np.isnan(y).any() or np.isinf(y).any():
                    if fit_fail_count < 10:
                        print(f"Skipping voxel ({x_idx},{y_idx},{z_idx}): y contains NaN or Inf")
                    fit_fail_count += 1
                    continue
                # Debug: Check model output for initial guess
                model_out = ivim_dki_model(b_values_array, *initial_guesses)
                if np.isnan(model_out).any() or np.isinf(model_out).any():
                    if fit_fail_count < 10:
                        print(f"Skipping voxel ({x_idx},{y_idx},{z_idx}): model output contains NaN or Inf")
                    fit_fail_count += 1
                    continue
                try:
                    fitted_params, _ = curve_fit(
                        ivim_dki_model,
                        b_values_array,
                        y,
                        p0=initial_guesses,
                        bounds=parameter_bounds,
                        maxfev=10000
                    )
                    y_hat = ivim_dki_model(b_values_array, *fitted_params)
                    residuals = y - y_hat
                    rss = np.sum(residuals ** 2)
                    n = len(y)
                    k_param = len(fitted_params)
                    aic = 2 * k_param + n * np.log(rss / n) if rss > 0 and n > 0 else np.nan
                    f_map[x_idx, y_idx, z_idx] = fitted_params[0]
                    D_star_map[x_idx, y_idx, z_idx] = fitted_params[1]
                    D_slow_map[x_idx, y_idx, z_idx] = fitted_params[2]
                    k_map[x_idx, y_idx, z_idx] = fitted_params[3]
                    aic_map[x_idx, y_idx, z_idx] = aic
                except Exception as e:
                    fit_fail_count += 1
                    if fit_fail_count < 10:  # Only print the first 10 failures to avoid flooding
                        print(f"Fit failed at ({x_idx},{y_idx},{z_idx}): {e}")
                    continue

print("Voxel-wise IVIM-DKI fitting complete.")
num_fitted_voxels = np.sum(~np.isnan(f_map))
print(f"Number of voxels fitted: {num_fitted_voxels} / {f_map.size}")
print(f"Total fit failures: {fit_fail_count}")

if num_fitted_voxels > 0:
    print(f"f_map min/max: {np.nanmin(f_map):.4f} / {np.nanmax(f_map):.4f}")
    print(f"D_star_map min/max: {np.nanmin(D_star_map):.4f} / {np.nanmax(D_star_map):.4f}")
    print(f"D_slow_map min/max: {np.nanmin(D_slow_map):.6f} / {np.nanmax(D_slow_map):.6f}")
    print(f"k_map min/max: {np.nanmin(k_map):.4f} / {np.nanmax(k_map):.4f}")
    print(f"AIC_map min/max: {np.nanmin(aic_map):.2f} / {np.nanmax(aic_map):.2f}")
else:
    print("No voxels were successfully fitted. Check data, b-values, bounds, or initial guesses.")

# Plot central slice of each map for visual inspection
slice_idx = num_voxels_z // 2

def plot_map(map_data, title, cmap='viridis'):
    plt.figure(figsize=(6, 5))
    plt.imshow(map_data[:, :, slice_idx], cmap=cmap, origin='lower')
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.show()

if num_fitted_voxels > 0:
    plot_map(f_map, "f_map (Perfusion Fraction)")
    plot_map(D_star_map, "D*_map (Pseudo-diffusion)")
    plot_map(D_slow_map, "D_slow_map (True Diffusion)")
    plot_map(k_map, "k_map (Kurtosis)")
    plot_map(aic_map, "AIC_map (Model AIC)", cmap='hot')

# Save all parameter maps as NIfTI files
output_dir = "/Users/ayush/Desktop/project-internsip/Results/6_Aic_CALC"
os.makedirs(output_dir, exist_ok=True)

affine = nifti_image.affine
nb.Nifti1Image(D_slow_map, affine).to_filename(os.path.join(output_dir, "D.nii.gz"))
nb.Nifti1Image(D_star_map, affine).to_filename(os.path.join(output_dir, "Dstar.nii.gz"))
nb.Nifti1Image(f_map, affine).to_filename(os.path.join(output_dir, "f.nii.gz"))
nb.Nifti1Image(k_map, affine).to_filename(os.path.join(output_dir, "k.nii.gz"))
nb.Nifti1Image(aic_map, affine).to_filename(os.path.join(output_dir, "AIC.nii.gz"))

print(f"Parameter maps saved to {output_dir}")

# Print statistics for quality check
def print_stats(name, arr):
    print(f"{name}: min={np.nanmin(arr):.4f}, max={np.nanmax(arr):.4f}, mean={np.nanmean(arr):.4f}, std={np.nanstd(arr):.4f}")

print_stats("f_map", f_map)
print_stats("D_star_map", D_star_map)
print_stats("D_slow_map", D_slow_map)
print_stats("k_map", k_map)
print_stats("AIC_map", aic_map)

print("hello world")