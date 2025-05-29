import numpy as np
import nibabel as nb
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def ivim_dki_model(b, f, D_star, D_slow, k):
    return f * np.exp(-b * D_star) + (1 - f) * np.exp(-b * D_slow + (1/6) * (b ** 2) * (D_slow ** 2) * k)


nii_file_path = "/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_SNR15/Data-1_Simulation-III_SNR-15.nii"
nifti_image = nb.load(nii_file_path)
image_data = nifti_image.get_fdata()
b_values_array = np.array([0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000])

mask = np.any(image_data > 0, axis=3) & (image_data[..., 0] != 0)


mean_signal = np.nanmean(image_data[mask], axis=0)

mean_signal = mean_signal / mean_signal[0]

bounds = ([0, 0, 0, 0], [0.2, 0.1, 0.005, 2])
p0 = [0.05, 0.01, 0.001, 0.5]
try:
    if not np.all(np.isfinite(mean_signal)):
        raise ValueError("Mean signal contains non-finite values!")
    params, _ = curve_fit(ivim_dki_model, b_values_array, mean_signal, p0=p0, bounds=bounds, maxfev=10000)
    f, D_star, D_slow, k = params
    y_pred = ivim_dki_model(b_values_array, *params)
    residuals = mean_signal - y_pred
    rss = np.sum(residuals ** 2)
    n = len(mean_signal)
    k_param = len(params)
    aic = 2 * k_param + n * np.log(rss / n) if rss > 0 and n > 0 else np.nan
    print("Mean IVIM-DKI fit results:")
    print(f"f: {f:.4f}")
    print(f"D_star: {D_star:.4f}")
    print(f"D_slow: {D_slow:.6f}")
    print(f"k: {k:.4f}")
    print(f"AIC: {aic:.2f}")
except Exception as e:
    print("Fitting failed:", e)

