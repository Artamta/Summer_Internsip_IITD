import nibabel as nib
import numpy as np
from aic_ivim_dki import calculate_aic_map_ivim_dki

# Load your data and parameter maps
image_data = nib.load("/path/to/original_4d_data.nii").get_fdata()
f_map = nib.load("/path/to/f_map.nii").get_fdata()
Dstar_map = nib.load("/path/to/Dstar_map.nii").get_fdata()
Dslow_map = nib.load("/path/to/Dslow_map.nii").get_fdata()
k_map = nib.load("/path/to/k_map.nii").get_fdata()
b_values = np.array([0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000])

# Optional: mask (e.g., where f_map > 0)
mask = f_map > 0

# Calculate AIC map and mean AIC
aic_map, mean_aic = calculate_aic_map_ivim_dki(
    image_data, f_map, Dstar_map, Dslow_map, k_map, b_values, mask=mask
)

print("Mean AIC:", mean_aic)