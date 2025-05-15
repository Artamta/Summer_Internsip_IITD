import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

img_path = '/Users/ayush/Desktop/project-internsip/Datasets/ivim_chest.nii.gz'
mask_path = '/Users/ayush/Desktop/project-internsip/Datasets/tumor.nii'
results_dir = '/Users/ayush/Desktop/project-internsip/Results'
os.makedirs(results_dir, exist_ok=True)

# Load images
img = nib.load(img_path)
mask = nib.load(mask_path)
img_data = img.get_fdata()
mask_data = mask.get_fdata()

print(f"Image shape: {img_data.shape}")
print(f"Mask shape: {mask_data.shape}")

# Select a slice to display (middle slice)
slice_idx = img_data.shape[2] // 2

# Plot image slice
plt.figure(figsize=(6, 6))
plt.imshow(img_data[:, :, slice_idx], cmap='gray')
plt.title(f'Image (Slice {slice_idx})')
plt.axis('off')
img_plot_path = os.path.join(results_dir, f"ivim_chest_slice_{slice_idx}.png")
plt.savefig(img_plot_path)
plt.close()
print(f"Saved image slice to {img_plot_path}")

# Plot mask slice
plt.figure(figsize=(6, 6))
plt.imshow(mask_data[:, :, slice_idx], cmap='gray')
plt.title(f'Mask (Slice {slice_idx})')
plt.axis('off')
mask_plot_path = os.path.join(results_dir, f"tumor_mask_slice_{slice_idx}.png")
plt.savefig(mask_plot_path)
plt.close()
print(f"Saved mask slice to {mask_plot_path}")