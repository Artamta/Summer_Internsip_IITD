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

# If image is 4D, select volumes for b=9 (last), b=8 (second last), and b=2 (third)
if img_data.ndim == 4:
    last_vol = img_data[..., -1]
    second_last_vol = img_data[..., -2]
    third_vol = img_data[..., 2]
else:
    last_vol = second_last_vol = third_vol = img_data

# Middle slice index for image and mask
mid_slice_img = last_vol.shape[2] // 2
mid_slice_mask = mask_data.shape[2] // 2

# Save last b-value (volume 9) mid slice
plt.figure(figsize=(6, 6))
plt.imshow(last_vol[:, :, mid_slice_img], cmap='gray')
plt.title(f'ivim_chest.nii.gz - Last Volume (b=9), Mid Slice {mid_slice_img}')
plt.axis('off')
plt.savefig(os.path.join(results_dir, f"ivim_chest_last_b9_mid_slice_{mid_slice_img}.png"))
plt.close()

# Save second last b-value (volume 8) mid slice
plt.figure(figsize=(6, 6))
plt.imshow(second_last_vol[:, :, mid_slice_img], cmap='gray')
plt.title(f'ivim_chest.nii.gz - Second Last Volume (b=8), Mid Slice {mid_slice_img}')
plt.axis('off')
plt.savefig(os.path.join(results_dir, f"ivim_chest_second_last_b8_mid_slice_{mid_slice_img}.png"))
plt.close()

# Save third volume (b=2) mid slice
plt.figure(figsize=(6, 6))
plt.imshow(third_vol[:, :, mid_slice_img], cmap='gray')
plt.title(f'ivim_chest.nii.gz - Third Volume (b=2), Mid Slice {mid_slice_img}')
plt.axis('off')
plt.savefig(os.path.join(results_dir, f"ivim_chest_third_b2_mid_slice_{mid_slice_img}.png"))
plt.close()

# Save middle slice of tumor mask
plt.figure(figsize=(6, 6))
plt.imshow(mask_data[:, :, mid_slice_mask], cmap='gray')
plt.title(f'Tumor Mask - Mid Slice {mid_slice_mask}')
plt.axis('off')
plt.savefig(os.path.join(results_dir, f"tumor_mask_mid_slice_{mid_slice_mask}.png"))
plt.close()

print("Saved all requested slices to Results folder.")