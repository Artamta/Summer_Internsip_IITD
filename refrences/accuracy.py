import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def sim_ivim_dki(b_val, est_d, est_dp, est_f, est_k):
    y_predicted = np.zeros(y_data.shape)

    for k in range(np.size(b_val)):
        y_predicted [:,:,:,k] =  (est_f * np.exp(-b_val[k] *est_dp)) + (1 - est_f) * np.exp(
            (-b_val[k] * est_d) + (1 / 6) * est_k * np.square(-b_val[k]) * np.square(est_d))

    return y_predicted

# Loading Images
b_val = [0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000]
ref_img = nib.load("C:/Users/eshab/Documents/MATLAB//Simulation-II_Dstar_map.nii")
# ref_data = nib.load("C:/Users/eshab/Documents/MATLAB//Simulation-III_IVIM_DKI_data.nii")
ref_data = nib.load("C:/Users/eshab/Documents/MATLAB//Data-1_Simulation-II_SNR-40.nii")

est_img_d = nib.load("C:/Users/eshab/PycharmProjects/IVIM_AUTOGRAD-20240805T083344Z-001/IVIM_AUTOGRAD/Output_Parameter_Maps/Data-1_Simulation-II_SNR-40_D.nii")
est_img_dstar = nib.load("C:/Users/eshab/PycharmProjects/IVIM_AUTOGRAD-20240805T083344Z-001/IVIM_AUTOGRAD/Output_Parameter_Maps/Data-1_Simulation-II_SNR-40_D_star.nii")
est_img_f = nib.load("C:/Users/eshab/PycharmProjects/IVIM_AUTOGRAD-20240805T083344Z-001/IVIM_AUTOGRAD/Output_Parameter_Maps/Data-1_Simulation-II_SNR-40_f.nii")
est_img_k = nib.load("C:/Users/eshab/PycharmProjects/IVIM_AUTOGRAD-20240805T083344Z-001/IVIM_AUTOGRAD/Output_Parameter_Maps/Data-1_Simulation-II_SNR-40_k.nii")

est_d = est_img_d.get_fdata()
est_dstar = est_img_dstar.get_fdata()
est_f = est_img_f.get_fdata()
est_k = est_img_k.get_fdata()

est_prm = est_dstar
ref_prm = ref_img.get_fdata()
y_data = ref_data.get_fdata()

# Shape of Phantom
print("Reference shape:", ref_prm.shape)
print("Estimated shape:", est_prm.shape)
print("Y-data shape:", y_data.shape)

# Error Calculation

ref_prm_vals = ref_prm.flatten()
est_prm_vals = est_prm.flatten()

# RMSE 
rmse = np.sqrt(np.mean((est_prm_vals - ref_prm_vals) ** 2))
rmse_norm = (rmse / np.mean(ref_prm_vals)) * 100

# Relative Bias 
aa = (np.subtract(est_prm_vals,  ref_prm_vals))
bb = np.divide(aa, ref_prm_vals)
rel_bias = np.mean(bb)*100

# Relative Parameter
rel_param = np.mean(est_prm_vals / ref_prm_vals)

# AIC = 2n + nln(RSS/n)
y_predicted = sim_ivim_dki(b_val, est_d, est_dstar, est_f, est_k)
aic_map = np.zeros(est_d.shape)
parameters = 4  # Number of parameters
n = np.size(b_val)  # Number of data points

for i in range(est_d.shape[0]):
    for j in range(est_d.shape[1]):
        for k in range(est_d.shape[2]):
            residuals = np.subtract(y_data[i,j,k,:], y_predicted[i,j,k,:])
            RSS = np.sum(residuals**2)
            aic_map[i,j,k] = 2 * parameters + n*np.log(RSS / n)

aic = np.nanmean(aic_map.flatten())

#AICc = AIC + (2k(k+1))/(n-k-1)
aicc = aic + (2*parameters*(parameters+1)/(n-parameters-1))

print(f"RMSE (normalized, %): ", rmse_norm)
print(f"Relative Bias (%): ",rel_bias)
print(f"Relative Parameter: ",rel_param)
print("AIC: ", aic)
print("AICc: ", aicc)

# Visualization:  (original, estimated)
mid_slice = ref_prm.shape[2] // 2
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(ref_prm[:, :, mid_slice], cmap='jet', vmin=0, vmax=0.03)
plt.title('Reference parameter map')
plt.axis('off')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(est_prm[:, :, mid_slice], cmap='jet', vmin=0, vmax=0.03)
plt.title('Estimated parameters map')
plt.axis('off')
plt.colorbar()
plt.tight_layout()

plt.show()
