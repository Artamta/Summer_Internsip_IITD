import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load the NIfTI image
nii_path = '/Users/ayush/Desktop/project-internsip/Datasets/tumor.nii.gz'
img = nib.load(nii_path)
data = img.get_fdata()

# Define ROI: Example using intensity threshold (modify as needed)
# For demonstration, select voxels with intensity > 0
roi_mask = data > 0
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