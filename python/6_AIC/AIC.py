import numpy as np
import nibabel as nb
from scipy.optimize import curve_fit
from lmfit import Model

# Load your .nii image (example: take mean signal as y_data)
nii = nb.load("/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_SNR15/Data-1_Simulation-III_SNR-15.nii")
data = nii.get_fdata()
y_data = np.mean(data, axis=(0, 1, 2))  # Example: mean signal per volume

# Example b-values (replace with your actual b-values)
b_values = np.array([0, 50, 100, 200, 400, 800])

# Ensure y_data and b_values have the same length
y_data = y_data[:len(b_values)]

# IVIM model function
def ivim(b, f, Dstar, D):
    return f * np.exp(-b * Dstar) + (1 - f) * np.exp(-b * D)

# Fit using scipy.optimize.curve_fit
popt, pcov = curve_fit(ivim, b_values, y_data, bounds=([0, 0, 0], [1, 0.1, 0.01]))
residuals = y_data - ivim(b_values, *popt)
rss = np.sum(residuals**2)
k = len(popt)
n = len(y_data)
aic = 2 * k + n * np.log(rss / n)
print("AIC (scipy.optimize):", aic)

# Fit using lmfit
ivim_model = Model(ivim)
params = ivim_model.make_params(f=0.1, Dstar=0.01, D=0.001)
params['f'].min = 0
params['f'].max = 1
params['Dstar'].min = 0
params['Dstar'].max = 0.1
params['D'].min = 0
params['D'].max = 0.01

result = ivim_model.fit(y_data, params, b=b_values)
print("AIC (lmfit):", result.aic)
print(result.fit_report())