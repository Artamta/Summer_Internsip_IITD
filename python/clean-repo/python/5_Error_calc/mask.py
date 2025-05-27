import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Loading Images
ref_img = nib.load("/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_f_map.nii")
est_img = nib.load("/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-III_SNR-15_f.nii")
ref_data = ref_img.get_fdata()
est_data = est_img.get_fdata()

# Shape of Phantom
print("Reference shape:", ref_data.shape)
print("Estimated shape:", est_data.shape)

# Generating Mask
mask = ref_data > 0

# Masking array
ref_masked = np.where(mask, ref_data, np.nan)
est_masked = np.where(mask, est_data, np.nan)

# Error Calculation

ref_vals = ref_data[mask]
est_vals = est_data[mask]
N = ref_vals.size

# RMSE 
rmse = np.sqrt(np.mean((est_vals - ref_vals) ** 2))
rmse_norm = (rmse / np.mean(ref_vals)) * 100

# Relative Bias 
rel_bias = (np.mean(est_vals - ref_vals) / np.mean(ref_vals)) * 100

# Relative Parameter
rel_param = np.mean(est_vals / ref_vals)

print(f"RMSE (normalized, %): {rmse_norm:.2f}")
print(f"Relative Bias (%): {rel_bias:.2f}")
print(f"Relative Parameter: {rel_param:.4f}")

# Visualization:  (original, estimated, mask, masked original, masked estimated)
mid_slice = ref_data.shape[2] // 2
plt.figure(figsize=(20, 4))
plt.subplot(1, 5, 1)
plt.imshow(ref_data[:, :, mid_slice], cmap='gray')
plt.title('Reference (full)')
plt.axis('off')
plt.subplot(1, 5, 2)
plt.imshow(est_data[:, :, mid_slice], cmap='gray')
plt.title('Estimated (full)')
plt.axis('off')
plt.subplot(1, 5, 3)
plt.imshow(mask[:, :, mid_slice], cmap='gray')
plt.title('Mask')
plt.axis('off')
plt.subplot(1, 5, 4)
plt.imshow(ref_masked[:, :, mid_slice], cmap='gray')
plt.title('Reference (masked)')
plt.axis('off')
plt.subplot(1, 5, 5)
plt.imshow(est_masked[:, :, mid_slice], cmap='gray')
plt.title('Estimated (masked)')
plt.axis('off')
plt.tight_layout()
plt.show()