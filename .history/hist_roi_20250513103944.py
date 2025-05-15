import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nibabel.processing import conform

# Load the NIfTI image and mask
img_path = '/Users/ayush/Desktop/project-internsip/Datasets/ivim_chest.nii'
mask_path = '/Users/ayush/Desktop/project-internsip/Datasets/tumor.nii'

img = nib.load(img_path)
mask = nib.load(mask_path)

# Check if shapes match
if img.shape != mask.shape:
    print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
    print("Resampling mask to match image...")
    mask = conform(mask, img.shape, voxel_size=img.header.get_zooms()[:3], orientation=img.affine)
    print(f"New mask shape: {mask.shape}")

data = img.get_fdata()
mask_data = mask.get_fdata()

# Ensure mask is binary
roi_mask = mask_data > 0

# Extract ROI data
roi_data = data[roi_mask]

# Histogram analysis
plt.figure(figsize=(8, 6))
plt.hist(roi_data.flatten(), bins=50, color='blue', alpha=0.7)
plt.title('Histogram of ROI Intensities')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Print basic statistics
print(f"ROI voxel count: {roi_data.size}")
print(f"Mean intensity: {roi_data.mean():.2f}")
print(f"Std intensity: {roi_data.std():.2f}")