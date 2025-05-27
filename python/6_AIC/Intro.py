import numpy as np
import nibabel as nb
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Load your .nii image
nii = nb.load("/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_SNR15/Data-1_Simulation-III_SNR-15.nii")
data = nii.get_fdata()  # shape: (X, Y, Z, N)
b_values = np.array([0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000])

# Print min/max of b=0 image
print("b=0 min/max:", np.min(data[:, :, :, 0]), np.max(data[:, :, :, 0]))

# Print signal at a central voxel
center = tuple(s//2 for s in data.shape[:3])
print("Signal at center voxel:", data[center[0], center[1], center[2], :])

# Prepare empty maps
shape = data.shape[:3]
f_map = np.full(shape, np.nan)
Dstar_map = np.full(shape, np.nan)
D_map = np.full(shape, np.nan)
AIC_map = np.full(shape, np.nan)

# IVIM model function
def ivim(b, f, Dstar, D):
    return f * np.exp(-b * Dstar) + (1 - f) * np.exp(-b * D)

# Loop over all voxels
for x in range(shape[0]):
    for y in range(shape[1]):
        for z in range(shape[2]):
            s = data[x, y, z, :]
            if np.any(s > 0):  # Try to fit if any signal is present
                if s[0] != 0:
                    y_data = s / s[0]  # S/S0 normalization
                else:
                    y_data = s  # If b=0 is zero, skip normalization
                try:
                    popt, _ = curve_fit(ivim, b_values, y_data, bounds=([0,0,0],[1,0.1,0.01]))
                    residuals = y_data - ivim(b_values, *popt)
                    rss = np.sum(residuals**2)
                    k = len(popt)
                    n = len(y_data)
                    aic = 2 * k + n * np.log(rss / n)
                    f_map[x, y, z] = popt[0]
                    Dstar_map[x, y, z] = popt[1]
                    D_map[x, y, z] = popt[2]
                    AIC_map[x, y, z] = aic
                except:
                    pass  # leave as np.nan

# Print debug info
print("Valid voxels (f_map):", np.sum(~np.isnan(f_map)))
print("f_map min/max:", np.nanmin(f_map), np.nanmax(f_map))
print("Dstar_map min/max:", np.nanmin(Dstar_map), np.nanmax(Dstar_map))
print("D_map min/max:", np.nanmin(D_map), np.nanmax(D_map))
print("AIC_map min/max:", np.nanmin(AIC_map), np.nanmax(AIC_map))

# Example: visualize a central slice of the maps
z_idx = f_map.shape[2] // 2

plt.figure(figsize=(16, 4))
plt.subplot(1, 4, 1)
plt.imshow(np.ma.masked_invalid(f_map[:, :, z_idx]), cmap='viridis')
plt.title('f map')
plt.axis('off')
plt.colorbar()

plt.subplot(1, 4, 2)
plt.imshow(np.ma.masked_invalid(Dstar_map[:, :, z_idx]), cmap='viridis')
plt.title('D* map')
plt.axis('off')
plt.colorbar()

plt.subplot(1, 4, 3)
plt.imshow(np.ma.masked_invalid(D_map[:, :, z_idx]), cmap='viridis')
plt.title('D map')
plt.axis('off')
plt.colorbar()

plt.subplot(1, 4, 4)
plt.imshow(np.ma.masked_invalid(AIC_map[:, :, z_idx]), cmap='magma')
plt.title('AIC map')
plt.axis('off')
plt.colorbar()

plt.tight_layout()
plt.show()