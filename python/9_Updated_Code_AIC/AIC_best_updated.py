import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# --- File paths ---
bvals = np.array([0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000])
toolbox_dir = "/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps"
my_dir = "/Users/ayush/Desktop/project-internsip/Results/6_Aic_CALC"
ref_f_path = "/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_f_map.nii"

# --- Load f-maps ---
ref_f = nib.load(ref_f_path).get_fdata()
f_toolbox = nib.load(f"{toolbox_dir}/Data-1_Simulation-III_SNR-60_f.nii").get_fdata()
f_my = nib.load(f"{my_dir}/f.nii.gz").get_fdata()

# --- Error metrics (no normalization) ---
def error_metrics(ref, est):
    mask = (ref > 0)
    ref_vals = ref[mask].flatten()
    est_vals = est[mask].flatten()
    rmse = np.sqrt(np.mean((est_vals - ref_vals) ** 2))
    rmse_norm = (rmse / np.mean(ref_vals)) * 100
    rel_bias = np.mean((est_vals - ref_vals) / ref_vals) * 100
    rel_param = np.mean(est_vals / ref_vals)
    return rmse_norm, rel_bias, rel_param

rmse_toolbox, bias_toolbox, relp_toolbox = error_metrics(ref_f, f_toolbox)
rmse_my, bias_my, relp_my = error_metrics(ref_f, f_my)

print(f"Toolbox f RMSE (normalized, %): {rmse_toolbox:.2f}")
print(f"Toolbox f Relative Bias (%): {bias_toolbox:.2f}")
print(f"Toolbox f Relative Parameter: {relp_toolbox:.4f}")
print(f"Voxelwise f RMSE (normalized, %): {rmse_my:.2f}")
print(f"Voxelwise f Relative Bias (%): {bias_my:.2f}")
print(f"Voxelwise f Relative Parameter: {relp_my:.4f}")

# --- AIC and AICc calculation for f-maps ---
def sim_ivim_dki(b_val, est_d, est_dp, est_f, est_k):
    y_pred = np.zeros(est_f.shape + (len(b_val),))
    for idx, b in enumerate(b_val):
        exp1 = np.exp(np.clip(-b * est_dp, -100, 100))
        exp2 = np.exp(np.clip(-b * est_d + (1 / 6) * est_k * (b ** 2) * (est_d ** 2), -100, 100))
        y_pred[..., idx] = (est_f * exp1) + (1 - est_f) * exp2
    return y_pred

# Load all parameter maps for both methods
D_toolbox = nib.load(f"{toolbox_dir}/Data-1_Simulation-III_SNR-60_D.nii").get_fdata()
Dstar_toolbox = nib.load(f"{toolbox_dir}/Data-1_Simulation-III_SNR-60_D_star.nii").get_fdata()
k_toolbox = nib.load(f"{toolbox_dir}/Data-1_Simulation-III_SNR-60_k.nii").get_fdata()
D_my = nib.load(f"{my_dir}/D.nii.gz").get_fdata()
Dstar_my = nib.load(f"{my_dir}/Dstar.nii.gz").get_fdata()
k_my = nib.load(f"{my_dir}/k.nii.gz").get_fdata()
nii_file_path = "/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_SNR60/Data-1_Simulation-III_SNR-60.nii"
y_data = nib.load(nii_file_path).get_fdata()

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

aic_toolbox = calc_aic_aicc(D_toolbox, Dstar_toolbox, f_toolbox, k_toolbox, y_data, bvals)
aic_my = calc_aic_aicc(D_my, Dstar_my, f_my, k_my, y_data, bvals)

print("Toolbox f AICc (mean):", np.nanmean(aic_toolbox))
print("Voxelwise f AICc (mean):", np.nanmean(aic_my))

# --- Debug: Check voxelwise f map range ---
print("Voxelwise f map min:", np.nanmin(f_my))
print("Voxelwise f map max:", np.nanmax(f_my))
print("Voxelwise f map mean:", np.nanmean(f_my))

# --- Visualization: show central slice of all f-maps ---
mid = ref_f.shape[2] // 2
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(ref_f[:, :, mid], cmap='jet', vmin=0, vmax=0.3)
plt.title('Reference f map')
plt.axis('off')
plt.colorbar()
plt.subplot(1, 3, 2)
plt.imshow(f_toolbox[:, :, mid], cmap='jet', vmin=0, vmax=0.3)
plt.title('Toolbox f map')
plt.axis('off')
plt.colorbar()
plt.subplot(1, 3, 3)
plt.imshow(f_my[:, :, mid], cmap='jet', vmin=0, vmax=0.3)
plt.title('Voxelwise f map')
plt.axis('off')
plt.colorbar()
plt.tight_layout()
plt.show()

# --- Optional: visualize voxelwise f map with auto scaling if blank ---
plt.figure()
plt.imshow(f_my[:, :, mid], cmap='jet')
plt.title('Voxelwise f map (auto scale)')
plt.axis('off')
plt.colorbar()
plt.show()

plt.imshow(f_my[:, :, mid], cmap='jet', vmin=0, vmax=0.02)
plt.title('Voxelwise f map (vmax=0.02)')
plt.axis('off')
plt.colorbar()
plt.show()