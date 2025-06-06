import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt

# --- File paths ---
nii_file_path = "/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_SNR60/Data-1_Simulation-III_SNR-60.nii"
b_values_array = np.array([0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000])

toolbox_dir = "/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps"
my_dir = "/Users/ayush/Desktop/project-internsip/Results/6_Aic_CALC"

# Loading parameter maps
f_toolbox = nb.load(f"{toolbox_dir}/Data-1_Simulation-III_SNR-60_f.nii").get_fdata()
Dstar_toolbox = nb.load(f"{toolbox_dir}/Data-1_Simulation-III_SNR-60_D_star.nii").get_fdata()
Dslow_toolbox = nb.load(f"{toolbox_dir}/Data-1_Simulation-III_SNR-60_D.nii").get_fdata()
k_toolbox = nb.load(f"{toolbox_dir}/Data-1_Simulation-III_SNR-60_k.nii").get_fdata()

f_my = nb.load(f"{my_dir}/f.nii.gz").get_fdata()
Dstar_my = nb.load(f"{my_dir}/Dstar.nii.gz").get_fdata()
Dslow_my = nb.load(f"{my_dir}/D.nii.gz").get_fdata()
k_my = nb.load(f"{my_dir}/k.nii.gz").get_fdata()

image_data = nb.load(nii_file_path).get_fdata()

# --- IVIM-DKI model ---
def ivim_dki_model(b_values, f, D_star, D_slow, k):
    exp1 = np.exp(np.clip(-b_values * D_star, -100, 100))
    exp2 = np.exp(np.clip(-b_values * D_slow + (1/6) * (b_values ** 2) * (D_slow ** 2) * k, -100, 100))
    result = f * exp1 + (1 - f) * exp2
    return np.clip(result, 0, 2)

shape = f_toolbox.shape
aic_toolbox = np.full(shape, np.nan)
aic_my = np.full(shape, np.nan)
rss_toolbox_map = np.full(shape, np.nan)
rss_my_map = np.full(shape, np.nan)
num_voxels_x, num_voxels_y, num_voxels_z = shape
n = len(b_values_array)
k_param = 4

# --- mask:
signal_mask = image_data[..., 0] > 0
param_mask_toolbox = np.isfinite(f_toolbox) & np.isfinite(Dstar_toolbox) & np.isfinite(Dslow_toolbox) & np.isfinite(k_toolbox)
param_mask_my = np.isfinite(f_my) & np.isfinite(Dstar_my) & np.isfinite(Dslow_my) & np.isfinite(k_my)
mask = signal_mask & param_mask_toolbox & param_mask_my

# --- Calculate AIC and RSS for both methods ---
for idx in zip(*np.where(mask)):
    x, y, z = idx
    signal = image_data[x, y, z, :]
    y_true = signal / signal[0]
    if np.isnan(y_true).any() or np.isinf(y_true).any():
        continue
    # Toolbox method
    f, D_star, D_slow, k = f_toolbox[x, y, z], Dstar_toolbox[x, y, z], Dslow_toolbox[x, y, z], k_toolbox[x, y, z]
    y_pred_toolbox = ivim_dki_model(b_values_array, f, D_star, D_slow, k)
    rss_toolbox = np.sum((y_true - y_pred_toolbox) ** 2)
    aic_toolbox[x, y, z] = 2 * k_param + n * np.log(rss_toolbox / n) if rss_toolbox > 0 else np.nan
    rss_toolbox_map[x, y, z] = rss_toolbox
    # Voxelwise method
    f, D_star, D_slow, k = f_my[x, y, z], Dstar_my[x, y, z], Dslow_my[x, y, z], k_my[x, y, z]
    y_pred_my = ivim_dki_model(b_values_array, f, D_star, D_slow, k)
    rss_my = np.sum((y_true - y_pred_my) ** 2)
    aic_my[x, y, z] = 2 * k_param + n * np.log(rss_my / n) if rss_my > 0 else np.nan
    rss_my_map[x, y, z] = rss_my

# --- Save AIC maps ---
affine = nb.load(nii_file_path).affine
nb.Nifti1Image(np.nan_to_num(aic_toolbox, nan=0.0), affine).to_filename(
    f"{toolbox_dir}/Data-1_Simulation-III_SNR-60_AIC.nii"
)
nb.Nifti1Image(np.nan_to_num(aic_my, nan=0.0), affine).to_filename(
    f"{my_dir}/AIC.nii.gz"
)
print("Saved toolbox and voxelwise AIC maps.")

# --- Print mean and max AIC values ---
mean_aic_toolbox = np.nanmean(aic_toolbox[mask])
mean_aic_my = np.nanmean(aic_my[mask])
max_aic_toolbox = np.nanmax(aic_toolbox[mask])
max_aic_my = np.nanmax(aic_my[mask])
print(f"Mean AIC (toolbox): {mean_aic_toolbox:.2f} | Max: {max_aic_toolbox:.2f}")
print(f"Mean AIC (voxelwise): {mean_aic_my:.2f} | Max: {max_aic_my:.2f}")

# --- Print mean RSS values ---
mean_rss_toolbox = np.nanmean(rss_toolbox_map[mask])
mean_rss_my = np.nanmean(rss_my_map[mask])
print(f"Mean RSS (toolbox): {mean_rss_toolbox:.4f}")
print(f"Mean RSS (voxelwise): {mean_rss_my:.4f}")

# --- RMSE, Relative Bias, Relative Parameter for f-maps ---
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

# Reference f_map 
ref_f_path = "/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_f_map.nii"
ref_f_data = nb.load(ref_f_path).get_fdata()

print_f_metrics(ref_f_data, f_toolbox, mask, "Toolbox f")
print_f_metrics(ref_f_data, f_my, mask, "Voxelwise f")

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