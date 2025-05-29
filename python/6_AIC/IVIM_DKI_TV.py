import numpy as np
import nibabel as nb
from scipy.optimize import curve_fit

def ivim_dki_model(b, f, D_star, D_slow, k):
    return f * np.exp(-b * D_star) + (1 - f) * np.exp(-b * D_slow + (1/6) * (b ** 2) * (D_slow ** 2) * k)

def simple_tv_penalty(param_map):
    # Simple TV: sum of absolute differences with neighbors in 3D
    tv = 0
    tv += np.nansum(np.abs(param_map[1:, :, :] - param_map[:-1, :, :]))
    tv += np.nansum(np.abs(param_map[:, 1:, :] - param_map[:, :-1, :]))
    tv += np.nansum(np.abs(param_map[:, :, 1:] - param_map[:, :, :-1]))
    return tv

def fit_voxelwise_with_tv(image_data, bvals, alpha=0.01):
    shape = image_data.shape[:3]
    mask = np.any(image_data > 0, axis=3) & (image_data[..., 0] != 0)
    f_map = np.full(shape, np.nan)
    D_star_map = np.full(shape, np.nan)
    D_slow_map = np.full(shape, np.nan)
    k_map = np.full(shape, np.nan)
    aic_map = np.full(shape, np.nan)  # <-- Add this line
    # Initial guesses and bounds
    bounds = ([0.0001, 0.0001, 0.001, 0.01], [0.05, 0.5, 1, 3])
    p0 = [0.0008,0.00913,0.12,0.9]
    # Fit each voxel independently
    for idx in zip(*np.where(mask)):
        y = image_data[idx]
        if y[0] == 0: continue
        y = y / y[0]
        try:
            params, _ = curve_fit(ivim_dki_model, bvals, y, p0=p0, bounds=bounds, maxfev=10000)
            f_map[idx], D_star_map[idx], D_slow_map[idx], k_map[idx] = params
            # --- Calculate AIC for this voxel ---
            y_pred = ivim_dki_model(bvals, *params)
            residuals = y - y_pred
            rss = np.sum(residuals ** 2)
            n = len(y)
            k_param = len(params)
            if rss > 0 and n > 0:
                aic = n * np.log(rss / n) + 2 * k_param
                aic_map[idx] = aic
        except Exception:
            continue
    # Apply simple TV smoothing to each map
    f_map_smooth = f_map.copy()
    D_star_map_smooth = D_star_map.copy()
    D_slow_map_smooth = D_slow_map.copy()
    k_map_smooth = k_map.copy()
    # One step of TV smoothing (can repeat for more smoothing)
    for param_map, param_map_smooth in zip(
        [f_map, D_star_map, D_slow_map, k_map],
        [f_map_smooth, D_star_map_smooth, D_slow_map_smooth, k_map_smooth]
    ):
        param_map_smooth[1:-1, :, :] = (param_map[:-2, :, :] + param_map[1:-1, :, :] + param_map[2:, :, :]) / 3
        param_map_smooth[:, 1:-1, :] = (param_map[:, :-2, :] + param_map[:, 1:-1, :] + param_map[:, 2:, :]) / 3
        param_map_smooth[:, :, 1:-1] = (param_map[:, :, :-2] + param_map[:, :, 1:-1] + param_map[:, :, 2:]) / 3
    # Print min/max
    print("TV-smoothed IVIM-DKI maps:")
    print(f"f_map min/max: {np.nanmin(f_map_smooth):.4f} / {np.nanmax(f_map_smooth):.4f}")
    print(f"D_star_map min/max: {np.nanmin(D_star_map_smooth):.4f} / {np.nanmax(D_star_map_smooth):.4f}")
    print(f"D_slow_map min/max: {np.nanmin(D_slow_map_smooth):.6f} / {np.nanmax(D_slow_map_smooth):.6f}")
    print(f"k_map min/max: {np.nanmin(k_map_smooth):.4f} / {np.nanmax(k_map_smooth):.4f}")
    print(f"AIC min/max: {np.nanmin(aic_map):.2f} / {np.nanmax(aic_map):.2f}")  # <-- Add this line
    return f_map_smooth, D_star_map_smooth, D_slow_map_smooth, k_map_smooth, aic_map

# --- Load data ---
nii_file_path = "/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_SNR15/Data-1_Simulation-III_SNR-15.nii"
nifti_image = nb.load(nii_file_path)
image_data = nifti_image.get_fdata()
b_values_array = np.array([0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000])

# --- Run voxelwise fit + simple TV smoothing ---
fit_voxelwise_with_tv(image_data, b_values_array, alpha=0.01)