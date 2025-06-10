import numpy as np
import nibabel as nb
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

# IVIM-DKI model in log-parameter space ---
#correction
#Fix the Bonds - Search in web
#use np and check element wise or array wise multiplication
#use np.exp np.log use np function 
#f * np.exp(-b * D_star) + (1 - f) * np.exp(-b * D_slow + (1/6) * (b ** 2) * (D_slow ** 2) * k)

def ivim_dki_model_log(b, log_f, log_D_star, log_D_slow, log_k):
    f = np.exp(log_f)
    D_star = np.exp(log_D_star)
    D_slow = np.exp(log_D_slow)
    k = np.exp(log_k)
    # Clip the exponentials to avoid overflow/underflow
    exp1 = np.exp(np.clip(-b * D_star, -100, 100))
    exp2 = np.exp(np.clip(-b * D_slow + (1/6) * (b ** 2) * (D_slow ** 2) * k, -100, 100))
    return f * exp1 + (1 - f) * exp2
#  Paths ---
snr60 = "/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_SNR60/Data-1_Simulation-III_SNR-60.nii"
ref_f_path = "/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_f_map.nii"
toolbox_f_path = "/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-III_SNR-60_f.nii"
toolbox_d_path = "/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-III_SNR-60_D.nii"
toolbox_dstar_path = "/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-III_SNR-60_D_star.nii"
toolbox_k_path = "/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-III_SNR-60_k.nii"

# Fitting settings (log scale)
bvals = np.array([0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000])
p0 = [0.013, 0.013, 0.23, 1.1]
#log_bounds = (np.log(bounds[0]), np.log(bounds[1]))
#make two bounds lower upper
#lower_bound=[0.0001, 0.0001, 0.001, 0.01]
#upper_bound=[0.05, 0.5, 1, 3]
#Just other bounds
lower_bound = [0.001, 0.001, 0.01, 0.05]
upper_bound = [0.5, 0.2, 0.5, 2]

log_bounds=(np.log(lower_bound),np.log(upper_bound))

log_p0 = np.log(p0)

# Load data
snr60_data = nb.load(snr60)
data = snr60_data.get_fdata()
shape = data.shape[:3]
f_map_voxelwise = np.full(shape, np.nan)
Dstar_map_voxelwise = np.full(shape, np.nan)
Dslow_map_voxelwise = np.full(shape, np.nan)
k_map_voxelwise = np.full(shape, np.nan)

# --- Voxelwise Fitting ---
# --- Voxelwise Fitting ---
fitted_voxels = 0
for x in range(shape[0]):
    for y in range(shape[1]):
        for z in range(shape[2]):
            signal = data[x, y, z, :]
            # Skip bad voxels
            if np.all(signal == 0) or np.any(signal < 0) or np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
                continue
            # Normalize only if b=0 is positive and not nan/inf
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
                continue

print("Total fitted voxels:", fitted_voxels)
print("f_map_voxelwise: min", np.nanmin(f_map_voxelwise), "max", np.nanmax(f_map_voxelwise))


# --- Visualization ---
mid_slice = shape[2] // 2

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(f_map_voxelwise[:, :, mid_slice], cmap='jet', vmin=0, vmax=1)
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