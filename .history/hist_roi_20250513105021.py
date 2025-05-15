import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nibabel.processing import conform
from scipy.stats import skew, kurtosis, entropy

# Load the NIfTI image and mask
img_path = '/Users/ayush/Desktop/project-internsip/Datasets/ivim_chest.nii'
mask_path = '/Users/ayush/Desktop/project-internsip/Datasets/tumor.nii'

img = nib.load(img_path)
mask = nib.load(mask_path)

# Select the first 3D volume if image is 4D
if len(img.shape) == 4:
    print(f"Image is 4D with shape {img.shape}, selecting first volume.")
    img_data = img.get_fdata()[..., 0]
else:
    img_data = img.get_fdata()

mask_data = mask.get_fdata()

# Check mask values for correct binarization
print("Unique values in mask:", np.unique(mask_data))
# If mask is not binary, adjust the threshold as needed
roi_mask = mask_data > 0  # or mask_data == 1 if mask is labeled

# Check if shapes match
if img_data.shape != mask_data.shape:
    print(f"Image shape: {img_data.shape}, Mask shape: {mask_data.shape}")
    print("Resampling mask to match image...")
    mask = conform(mask, img_data.shape, voxel_size=img.header.get_zooms()[:3])
    mask_data = mask.get_fdata()
    print(f"New mask shape: {mask_data.shape}")
    roi_mask = mask_data > 0  # Recompute after resampling

# Extract ROI data
roi_data = img_data[roi_mask]

# ROI portion
roi_portion = roi_mask.sum() / roi_mask.size
print(f"ROI voxel count: {roi_mask.sum()}")
print(f"ROI portion (fraction of image): {roi_portion:.6f}")

# Histogram analysis
hist, bin_edges = np.histogram(roi_data, bins=50)
plt.figure(figsize=(8, 6))
plt.hist(roi_data.flatten(), bins=50, color='blue', alpha=0.7)
plt.title('Histogram of ROI Intensities')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Print histogram frequencies (all bins)
print("Histogram frequencies (all bins):", hist)

# Print basic statistics
print(f"Min intensity: {roi_data.min():.2f}")
print(f"Max intensity: {roi_data.max():.2f}")
print(f"Mean intensity: {roi_data.mean():.2f}")
print(f"Median intensity: {np.median(roi_data):.2f}")
print(f"Std intensity: {roi_data.std():.2f}")
print(f"Skewness: {skew(roi_data):.2f}")
print(f"Kurtosis: {kurtosis(roi_data):.2f}")

# Entropy (using normalized histogram)
hist_norm = hist / hist.sum()
roi_entropy = entropy(hist_norm)
print(f"Entropy: {roi_entropy:.4f}")

# Display ROI mask (middle slice)
slice_idx = img_data.shape[2] // 2
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(mask_data[:, :, slice_idx], cmap='gray')
plt.title('Mask Only (Middle Slice)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(roi_mask[:, :, slice_idx], cmap='gray')
plt.title('ROI Mask (Middle Slice)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_data[:, :, slice_idx], cmap='gray')
plt.imshow(mask_data[:, :, slice_idx], cmap='Reds', alpha=0.3)
plt.title('Mask Overlay on Image')
plt.axis('off')
plt.tight_layout()
plt.show()