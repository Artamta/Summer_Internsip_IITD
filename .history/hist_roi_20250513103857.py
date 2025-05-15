import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load the NIfTI image and mask
nii_path = '/Users/ayush/Desktop/project-internsip/Datasets/tumor.nii.gz'
mask_path = '/Users/ayush/Desktop/project-internsip/Datasets/tumor_mask.nii.gz'  # Update with your mask path

img = nib.load(nii_path)
data = img.get_fdata()

mask_img = nib.load(mask_path)
mask_data = mask_img.get_fdata()

# Ensure mask is binary
roi_mask = mask_data > 0

# Extract ROI data
roi_data = data[roi_mask]

# Histogram analysis
plt.figure(figsize=(8, 6))
plt.hist(roi_data.flatten(), bins=50, color='blue', alpha=0.7)
plt.title('Histogram of Tumor ROI Intensities')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Print basic statistics
print(f"ROI voxel count: {roi_data.size}")
print(f"Mean intensity: {roi_data.mean():.2f}")
print(f"Std intensity: {roi_data.std():.2f}")