import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, entropy

# Load the masked tumor image
img_path = '/Users/ayush/Desktop/project-internsip/Datasets/tumor.nii'
img = nib.load(img_path)
img_data = img.get_fdata()

# Only consider nonzero voxels (the ROI)
roi_data = img_data[img_data > 0]

# Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(roi_data, bins=50, color='blue', alpha=0.7)
plt.title('Histogram of Tumor ROI Intensities')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Show middle slice
slice_idx = img_data.shape[2] // 2
plt.figure(figsize=(6, 6))
plt.imshow(img_data[:, :, slice_idx], cmap='gray')
plt.title('Tumor ROI (Middle Slice)')
plt.axis('off')
plt.show()

# Print statistics
print(f"ROI voxel count: {roi_data.size}")
print(f"Min intensity: {roi_data.min():.2f}")
print(f"Max intensity: {roi_data.max():.2f}")
print(f"Mean intensity: {roi_data.mean():.2f}")
print(f"Median intensity: {np.median(roi_data):.2f}")
print(f"Std intensity: {roi_data.std():.2f}")
print(f"Skewness: {skew(roi_data):.2f}")
print(f"Kurtosis: {kurtosis(roi_data):.2f}")

# Entropy (using normalized histogram)
hist, _ = np.histogram(roi_data, bins=50)
hist_norm = hist / hist.sum()
roi_entropy = entropy(hist_norm)
print(f"Entropy: {roi_entropy:.4f}")