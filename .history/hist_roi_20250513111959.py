import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, entropy
import os
import csv

# Paths
img_path = '/Users/ayush/Desktop/project-internsip/Datasets/ivim_chest.nii'
mask_path = '/Users/ayush/Desktop/project-internsip/Datasets/tumor.nii'
results_dir = '/Users/ayush/Desktop/project-internsip/Results'
os.makedirs(results_dir, exist_ok=True)

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
stats = {
    "ROI voxel count": roi_data.size,
    "ROI volume (mm^3)": roi_volume,
    "Min intensity": roi_data.min(),
    "Max intensity": roi_data.max(),
    "Mean intensity": roi_data.mean(),
    "Median intensity": np.median(roi_data),
    "Std intensity": roi_data.std(),
    "25th percentile": q25,
    "75th percentile": q75,
    "IQR": q75 - q25,
    "Skewness": skew(roi_data),
    "Kurtosis": kurtosis(roi_data)
}

# Entropy (using normalized histogram)
hist, _ = np.histogram(roi_data, bins=50)
hist_norm = hist / hist.sum()
roi_entropy = entropy(hist_norm)
stats["Entropy"] = roi_entropy

# Save statistics to CSV
stats_csv_path = os.path.join(results_dir, "tumor_roi_statistics.csv")
with open(stats_csv_path, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Parameter", "Value"])
    for k, v in stats.items():
        writer.writerow([k, v])
print(f"Saved statistics to {stats_csv_path}")

# Professional Histogram with KDE
plt.figure(figsize=(10, 6))
sns.histplot(roi_data, bins=50, kde=True, color='royalblue', stat='density')
plt.title('Histogram & KDE of Tumor ROI Intensities', fontsize=16)
plt.xlabel('Intensity', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
hist_path = os.path.join(results_dir, "tumor_roi_histogram_kde.png")
plt.savefig(hist_path)
plt.close()
print(f"Saved histogram to {hist_path}")

# Boxplot
plt.figure(figsize=(8, 2))
sns.boxplot(x=roi_data, color='lightcoral')
plt.title('Boxplot of Tumor ROI Intensities', fontsize=14)
plt.xlabel('Intensity', fontsize=12)
plt.tight_layout()
boxplot_path = os.path.join(results_dir, "tumor_roi_boxplot.png")
plt.savefig(boxplot_path)
plt.close()
print(f"Saved boxplot to {boxplot_path}")

# Frequency vs Intensity Histogram (classic)
plt.figure(figsize=(10, 6))
sns.histplot(roi_data, bins=50, color='seagreen', stat='count')
plt.title('Frequency vs Intensity Histogram (Tumor ROI)', fontsize=16)
plt.xlabel('Intensity', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
freq_hist_path = os.path.join(results_dir, "tumor_roi_histogram_frequency.png")
plt.savefig(freq_hist_path)
plt.close()
print(f"Saved frequency vs intensity histogram to {freq_hist_path}")

# filepath: /Users/ayush/Desktop/project-internsip/hist_roi.py
# Find the slice with the most tumor voxels
tumor_counts = roi_mask.sum(axis=(0, 1))
best_slice = np.argmax(tumor_counts)
print(f"Slice with largest tumor area: {best_slice}")

tumor_only_slice = tumor_only[:, :, best_slice]

plt.figure(figsize=(6, 6))
plt.imshow(tumor_only_slice, cmap='gray')
plt.title(f'Tumor Only (Slice {best_slice})', fontsize=14)
plt.axis('off')
tumor_only_path = os.path.join(results_dir, f"tumor_only_masked_slice_{best_slice}.png")
plt.savefig(tumor_only_path)
plt.close()
print(f"Saved tumor-only masked image to {tumor_only_path}")

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
overlay_path = os.path.join(results_dir, "tumor_mask_overlay_middle_slice.png")
plt.savefig(overlay_path)
plt.close()
print(f"Saved mask overlay to {overlay_path}")