import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os

# ---- User: Set your file paths here ----
raw_nii_path = '/Users/ayush/Desktop/project-internsip/Datasets/patient_data/Liver_HCC/Gauri/Gauri1.nii'
fitted_nii_path = '/Users/ayush/Desktop/project-internsip/Results/15_Patient_Output/Liver_HCC/Gauri/Output_Parameter_Maps/Gauri1_D.nii'

# ---- Load NIfTI images ----
raw_img = nib.load(raw_nii_path).get_fdata()
fitted_img = nib.load(fitted_nii_path).get_fdata()

# ---- Print shapes for debugging ----
print("raw_img shape:", raw_img.shape)
print("fitted_img shape:", fitted_img.shape)

# ---- Choose a representative slice (middle slice) ----
slice_idx = raw_img.shape[2] // 2

# ---- Handle 3D or 4D images for both ----
if raw_img.ndim == 4:
    raw_slice = raw_img[:, :, slice_idx, 0]
else:
    raw_slice = raw_img[:, :, slice_idx]

if fitted_img.ndim == 4:
    fitted_slice = fitted_img[:, :, slice_idx, 0]
else:
    fitted_slice = fitted_img[:, :, slice_idx]

# ---- Qualitative assessment (difference map) ----
qualitative_map = np.abs(raw_slice - fitted_slice)

# ---- Output directory ----
output_dir = '/Users/ayush/Desktop/project-internsip/Results/17_random'
os.makedirs(output_dir, exist_ok=True)

# ---- Plot and save each image separately ----

# (a) Raw diffusion data
plt.figure(figsize=(6, 6))
plt.imshow(raw_slice, cmap='gray')
plt.title('Raw Diffusion Data')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'raw_diffusion.png'), bbox_inches='tight', dpi=300)
plt.close()

# (b) Fitted parameter map
plt.figure(figsize=(6, 6))
im1 = plt.imshow(fitted_slice, cmap='viridis')
plt.title('IDTV Fitted Map')
plt.axis('off')
plt.colorbar(im1, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'idtv_fitted_map.png'), bbox_inches='tight', dpi=300)
plt.close()

# (c) Qualitative assessment (difference map)
plt.figure(figsize=(6, 6))
im2 = plt.imshow(qualitative_map, cmap='hot')
plt.title('Qualitative Assessment (|Raw - Fitted|)')
plt.axis('off')
plt.colorbar(im2, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'qualitative_assessment.png'), bbox_inches='tight', dpi=300)
plt.close()

print("All plots saved in:", output_dir)