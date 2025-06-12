import numpy as np
import nibabel as nb
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# --- IVIM-DKI model in log-parameter space ---
def ivim_dki_model_log(b, log_f, log_D_star, log_D_slow, log_k):
    f = np.exp(log_f)
    D_star = np.exp(log_D_star)
    D_slow = np.exp(log_D_slow)
    k = np.exp(log_k)
    exp1 = np.exp(np.clip(-b * D_star, -100, 100))
    exp2 = np.exp(np.clip(-b * D_slow + (1/6) * (b ** 2) * (D_slow ** 2) * k, -100, 100))
    return f * exp1 + (1 - f) * exp2

# --- Paths ---
snr60 = "/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_SNR60/Data-1_Simulation-III_SNR-60.nii"

# --- Fitting settings (log scale) ---
bvals = np.array([0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000])
p0 = [0.05, 0.01, 0.1, 0.5]
lower_bound = [0.001, 0.001, 0.01, 0.01]
upper_bound = [0.7, 0.2, 0.5, 2]
log_bounds = (np.log(lower_bound), np.log(upper_bound))
log_p0 = np.log(p0)

# --- Load data ---
snr60_data = nb.load(snr60)
data = snr60_data.get_fdata()
shape = data.shape[:3]
f_map_voxelwise = np.full(shape, np.nan)
Dstar_map_voxelwise = np.full(shape, np.nan)
Dslow_map_voxelwise = np.full(shape, np.nan)
k_map_voxelwise = np.full(shape, np.nan)

# --- Fast test: fit only a small central region ---
xmid, ymid, zmid = shape[0]//2, shape[1]//2, shape[2]//2
x_range = range(xmid-2, xmid+2)
y_range = range(ymid-2, ymid+2)
z_range = [zmid]

fitted_voxels = 0
for x in x_range:
    for y in y_range:
        for z in z_range:
            signal = data[x, y, z, :]
            if np.all(signal == 0) or np.any(signal < 0) or np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
                continue
            if signal[0] <= 0 or np.isnan(signal[0]) or np.isinf(signal[0]):
                continue
            signal = signal / signal[0]
            try:
                log_ppot, _ = curve_fit(
                    ivim_dki_model_log, bvals, signal, p0=log_p0, bounds=log_bounds
                )
                ppot = np.exp(log_ppot)
                f_map_voxelwise[x, y, z] = ppot[0]
                Dstar_map_voxelwise[x, y, z] = ppot[1]
                Dslow_map_voxelwise[x, y, z] = ppot[2]
                k_map_voxelwise[x, y, z] = ppot[3]
                fitted_voxels += 1
            except Exception as e:
                print(f"Fit failed at ({x},{y},{z}):", e)
                continue

print("Total fitted voxels:", fitted_voxels)
print("f_map_voxelwise: min", np.nanmin(f_map_voxelwise), "max", np.nanmax(f_map_voxelwise))

# --- Visualization ---
mid_slice = zmid
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(f_map_voxelwise[:, :, mid_slice], cmap='jet', vmin=0, vmax=0.7)
plt.title('Fitted f map (middle slice)')
plt.axis('off')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(Dstar_map_voxelwise[:, :, mid_slice], cmap='jet')
plt.title('Fitted D* map (middle slice)')
plt.axis('off')
plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(Dslow_map_voxelwise[:, :, mid_slice], cmap='jet')
plt.title('Fitted Dslow map (middle slice)')
plt.axis('off')
plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow(k_map_voxelwise[:, :, mid_slice], cmap='jet')
plt.title('Fitted k map (middle slice)')
plt.axis('off')
plt.colorbar()

plt.tight_layout()
plt.show()