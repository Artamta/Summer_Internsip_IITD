import numpy as np
import nibabel as nb

# Load original 4D data
nii_file_path = "/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_SNR60/Data-1_Simulation-III_SNR-60.nii"
image_data = nb.load(nii_file_path).get_fdata()
b_values_array = np.array([0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000])

# Load toolbox parameter maps
toolbox_dir = "/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps"
f_toolbox = nb.load(f"{toolbox_dir}/Data-1_Simulation-III_SNR-60_f.nii").get_fdata()
Dstar_toolbox = nb.load(f"{toolbox_dir}/Data-1_Simulation-III_SNR-60_D_star.nii").get_fdata()
Dslow_toolbox = nb.load(f"{toolbox_dir}/Data-1_Simulation-III_SNR-60_D.nii").get_fdata()
k_toolbox = nb.load(f"{toolbox_dir}/Data-1_Simulation-III_SNR-60_k.nii").get_fdata()

# IVIM-DKI model
def ivim_dki_model(b_values, f, D_star, D_slow, k):
    exp1 = np.exp(np.clip(-b_values * D_star, -100, 100))
    exp2 = np.exp(np.clip(-b_values * D_slow + (1/6) * (b_values ** 2) * (D_slow ** 2) * k, -100, 100))
    result = f * exp1 + (1 - f) * exp2
    return np.clip(result, 0, 2)

# Prepare output AIC map
aic_toolbox = np.full(f_toolbox.shape, np.nan)
num_voxels_x, num_voxels_y, num_voxels_z = f_toolbox.shape

for x in range(num_voxels_x):
    for y in range(num_voxels_y):
        for z in range(num_voxels_z):
            signal = image_data[x, y, z, :]
            if np.any(signal > 0) and signal[0] != 0:
                y_true = signal / signal[0]
                if np.isnan(y_true).any() or np.isinf(y_true).any():
                    continue
                f = f_toolbox[x, y, z]
                D_star = Dstar_toolbox[x, y, z]
                D_slow = Dslow_toolbox[x, y, z]
                k = k_toolbox[x, y, z]
                y_pred = ivim_dki_model(b_values_array, f, D_star, D_slow, k)
                residuals = y_true - y_pred
                rss = np.sum(residuals ** 2)
                n = len(y_true)
                k_param = 4
                aic = 2 * k_param + n * np.log(rss / n) if rss > 0 and n > 0 else np.nan
                aic_toolbox[x, y, z] = aic

# Save AIC map for toolbox method
affine = nb.load(nii_file_path).affine
nb.Nifti1Image(np.nan_to_num(aic_toolbox, nan=0.0), affine).to_filename(
    "/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-III_SNR-60_AIC.nii"
)
print("Saved toolbox AIC map.")