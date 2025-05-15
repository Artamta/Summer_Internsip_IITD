import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, entropy

# Paths
img_path = '/Users/ayush/Desktop/project-internsip/Datasets/ivim_chest.nii'
mask_path = '/Users/ayush/Desktop/project-internsip/Datasets/tumor.nii'

# Load image and mask
img = nib.load(img_path)
mask = nib.load(mask_path)

# If image is 4D, select the first volume
img_data = img.get_fdata()
if img_data.ndim == 4:
    print(f"Image is 4D with shape {img_data.shape}, selecting first volume.")
    img_data = img_data[..., 0]

mask_data = mask.get_fdata()

# Resample mask if needed
if img_data.shape != mask_data.shape:
    from nibabel.processing import conform
    print(f"Resampling mask from {mask_data.shape} to {img_data.shape} ...")
    mask = conform(mask, img_data.shape, voxel_size=img.header.get_zooms()[:3])
    mask_data = mask.get_fdata()

# Ensure mask is binary
roi_mask = mask_data > 0

# Extract ROI intensity values
roi_data = img_data[roi_mask]

# Voxel size and volume
voxel_volume = np.prod(img.header.get_zooms()[:3])
roi_volume = roi_data.size * voxel_volume

# Statistics
q25, q75 = np.percentile(roi_data, [25, 75])
print(f"ROI voxel count: {roi_data.size}")
print(f"ROI volume: {roi_volume:.2f} mm³")
print(f"Min intensity: {roi_data.min():.2f}")
print(f"Max intensity: {roi_data.max():.2f}")
print(f"Mean intensity: {roi_data.mean():.2f}")
print(f"Median intensity: {np.median(roi_data):.2f}")
print(f"Std intensity: {roi_data.std():.2f}")
print(f"25th percentile: {q25:.2f}")
print(f"75th percentile: {q75:.2f}")
print(f"IQR: {(q75-q25):.2f}")
print(f"Skewness: {skew(roi_data):.2f}")
print(f"Kurtosis: {kurtosis(roi_data):.2f}")

# Entropy (using normalized histogram)
hist, _ = np.histogram(roi_data, bins=50)
hist_norm = hist / hist.sum()
roi_entropy = entropy(hist_norm)
print(f"Entropy: {roi_entropy:.4f}")

# Professional Histogram with KDE
plt.figure(figsize=(10, 6))
sns.histplot(roi_data, bins=50, kde=True, color='royalblue', stat='density')
plt.title('Histogram & KDE of Tumor ROI Intensities', fontsize=16)
plt.xlabel('Intensity', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Boxplot
plt.figure(figsize=(8, 2))
sns.boxplot(x=roi_data, color='lightcoral')
plt.title('Boxplot of Tumor ROI Intensities', fontsize=14)
plt.xlabel('Intensity', fontsize=12)
plt.tight_layout()
plt.show()

# Show middle slice with mask overlay
slice_idx = img_data.shape[2] // 2
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_data[:, :, slice_idx], cmap='gray')
plt.title('Image (Middle Slice)')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img_data[:, :, slice_idx], cmap='gray')
plt.imshow(roi_mask[:, :, slice_idx], cmap='Reds', alpha=0.3)
plt.title('Tumor Mask Overlay')
plt.axis('off')
plt.tight_layout()
plt.show()