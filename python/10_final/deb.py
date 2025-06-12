import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def ivim_dki_model(b, f, D_star, D, k):
    exp1 = np.exp(-b * D_star)
    exp2 = np.exp(-b * D + (1/6) * (b ** 2) * (D ** 2) * k)
    return f * exp1 + (1 - f) * exp2

nii_path = "/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_SNR60/Data-1_Simulation-III_SNR-60.nii"
bvals = np.array([0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000])

img = nib.load(nii_path)
data = img.get_fdata()
shape = data.shape[:3]

f_map = np.full(shape, np.nan)
Dstar_map = np.full(shape, np.nan)
D_map = np.full(shape, np.nan)
k_map = np.full(shape, np.nan)
fitted_voxels = 0
debug_prints = 0

for x in range(shape[0]):
    for y in range(shape[1]):
        for z in range(shape[2]):
            signal = data[x, y, z, :]
            if signal[0] <= 0 or np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
                continue
            signal = signal / signal[0]
            if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
                continue
            try:
                p0 = [0.1, 0.01, 0.001, 0.5]  # f, D*, D, k
                bounds = ([0, 0.0005, 0.0001, 0.01], [1, 0.1, 0.003, 3])
                popt, _ = curve_fit(ivim_dki_model, bvals, signal, p0=p0, bounds=bounds, maxfev=10000)
                f_map[x, y, z] = popt[0]
                Dstar_map[x, y, z] = popt[1]
                D_map[x, y, z] = popt[2]
                k_map[x, y, z] = popt[3]
                fitted_voxels += 1
                if debug_prints < 5:
                    print(f"Voxel ({x},{y},{z}) fit: f={popt[0]:.3f}, D*={popt[1]:.4f}, D={popt[2]:.4f}, k={popt[3]:.3f}")
                    debug_prints += 1
            except Exception:
                continue

print(f"Total fitted voxels: {fitted_voxels}")
print(f"f_map: min={np.nanmin(f_map):.4f}, max={np.nanmax(f_map):.4f}")
print(f"Dstar_map: min={np.nanmin(Dstar_map):.5f}, max={np.nanmax(Dstar_map):.5f}")
print(f"D_map: min={np.nanmin(D_map):.5f}, max={np.nanmax(D_map):.5f}")
print(f"k_map: min={np.nanmin(k_map):.3f}, max={np.nanmax(k_map):.3f}")
print("\n--- Literature reference ranges ---")
print("f: 0–0.2 (brain), up to 0.3 (tumor/liver)")
print("D: 0.0005–0.0015 mm^2/s (brain)")
print("D*: 0.005–0.05 mm^2/s (brain)")
print("k: 0–3 (varies)")

mid_z = shape[2] // 2
plt.figure(figsize=(6, 5))
plt.imshow(f_map[:, :, mid_z], cmap='jet', vmin=0, vmax=1)
plt.title('IVIM-DKI f map (mid-slice)')
plt.axis('off')
plt.colorbar()
plt.tight_layout()
plt.show()