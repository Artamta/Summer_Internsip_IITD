import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def sim_ivim_dki(b_val, est_d, est_dp, est_f, est_k):
    y_predicted = np.zeros(y_data.shape)
    for k in range(np.size(b_val)):
        y_predicted[:,:,:,k] = (est_f * np.exp(-b_val[k] * est_dp)) + (1 - est_f) * np.exp(
            (-b_val[k] * est_d) + (1 / 6) * est_k * np.square(-b_val[k]) * np.square(est_d))
    return y_predicted

# Loading Images
b_val = [0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000]

# Load original parametric map for accuracy calculation
ref_img = nib.load("/Users/ayush/Desktop/project-internsip/new_data/Simulation-I_D_map.nii")

# Load simulated IVIM-DKI data
ref_data = nib.load("/Users/ayush/Desktop/project-internsip/new_data/Data-1_Simulation-I_SNR-60.nii")

# Load estimated Parametric map for advanced model IDTV
est_img_d_tv = nib.load("/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-I_SNR-60_D.nii")
est_img_dstar_tv = nib.load("/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-I_SNR-60_D_star.nii")
est_img_f_tv = nib.load("/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-I_SNR-60_f.nii")
est_img_k_tv = nib.load("/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-I_SNR-60_k.nii")

# --- HY model lines commented out ---
# est_img_d = nib.load("...")
# est_img_dstar = nib.load("...")
# est_img_f = nib.load("...")
# est_img_k = nib.load("...")

# est_d = est_img_d.get_fdata()
# est_dstar = est_img_dstar.get_fdata()
# est_f = est_img_f.get_fdata()
# est_k = est_img_k.get_fdata()

est_d_tv = est_img_d_tv.get_fdata()
est_dstar_tv = est_img_dstar_tv.get_fdata()
est_f_tv = est_img_f_tv.get_fdata()
est_k_tv = est_img_k_tv.get_fdata()

# est_prm = est_d  # change this for different simulations
est_prm_tv = est_d_tv # change this for different simulations
ref_prm = ref_img.get_fdata()
y_data = ref_data.get_fdata()

# Shape of Phantom
print("Reference shape:", ref_prm.shape)
# print("Estimated shape:", est_prm.shape)
print("Estimated shape:", est_prm_tv.shape)
print("Y-data shape:", y_data.shape)

# Error Calculation

ref_prm_vals = ref_prm.flatten()
# est_prm_vals = est_prm.flatten()
est_prm_tv_vals = est_prm_tv.flatten()

# RMSE 
# rmse = np.sqrt(np.mean((est_prm_vals - ref_prm_vals) ** 2))
# rmse_norm_hy = (rmse / np.mean(ref_prm_vals)) * 100

rmse_tv = np.sqrt(np.mean((est_prm_tv_vals - ref_prm_vals) ** 2))
rmse_norm_tv = (rmse_tv / np.mean(ref_prm_vals)) * 100

# Relative Bias 
# aa = (np.subtract(est_prm_vals,  ref_prm_vals))
# bb = np.divide(aa, ref_prm_vals)
# rel_bias_hy = np.mean(bb)*100

aa = (np.subtract(est_prm_tv_vals,  ref_prm_vals))
bb = np.divide(aa, ref_prm_vals)
rel_bias_tv = np.mean(bb)*100

# Relative Parameter
# rel_param_hy = np.mean(est_prm_vals / ref_prm_vals)
rel_param_tv = np.mean(est_prm_tv_vals / ref_prm_vals)

# AIC = 2n + nln(RSS/n)
# y_predicted_hy = sim_ivim_dki(b_val, est_d, est_dstar, est_f, est_k)
y_predicted_tv = sim_ivim_dki(b_val, est_d_tv, est_dstar_tv, est_f_tv, est_k_tv)

# aic_map_hy = np.zeros(est_d.shape)
aic_map_tv = np.zeros(est_d_tv.shape)

parameters = 4  # Number of parameters
n = np.size(b_val)  # Number of data points

for i in range(est_d_tv.shape[0]):
    for j in range(est_d_tv.shape[1]):
        for k in range(est_d_tv.shape[2]):
            # residuals_hy = np.subtract(y_data[i,j,k,:], y_predicted_hy[i,j,k,:])
            # RSS_hy = np.sum(residuals_hy**2)
            # aic_map_hy[i,j,k] = 2 * parameters + n*np.log(RSS_hy / n)
            residuals_tv = np.subtract(y_data[i, j, k, :], y_predicted_tv[i, j, k, :])
            RSS_tv = np.sum(residuals_tv ** 2)
            aic_map_tv[i, j, k] = 2 * parameters + n * np.log(RSS_tv / n)

# aic_hy = np.nanmean(aic_map_hy.flatten())
aic_tv = np.nanmean(aic_map_tv.flatten())

#AICc = AIC + (2k(k+1))/(n-k-1)
# aicc_hy = aic_hy + (2*parameters*(parameters+1)/(n-parameters-1))
aicc_tv = aic_tv + (2*parameters*(parameters+1)/(n-parameters-1))

# print("RMSE normalized: HYmodel = %, IDTV model = %", rmse_norm_hy, rmse_norm_tv)
# print("Relative Bias: HYmodel = %, IDTV model = %",rel_bias_hy, rel_bias_tv)
# print("Relative Parameter: HYmodel = %, IDTV model = %",rel_param_hy, rel_param_tv)
# print("AIC: HYmodel = %, IDTV model = %", aic_hy, aicc_tv)
# print("AIC Corrected: HYmodel = %, IDTV model = %", aicc_hy, aicc_tv)

print("RMSE normalized: IDTV model = %", rmse_norm_tv)
print("Relative Bias: IDTV model = %", rel_bias_tv)
print("Relative Parameter: IDTV model = %", rel_param_tv)
print("AIC: IDTV model = %", aic_tv)
print("AIC Corrected: IDTV model = %", aicc_tv)

# Visualization:  (original, estimated)
mid_slice = ref_prm.shape[2] // 2
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(ref_prm[:, :, mid_slice], cmap='jet', vmin=0, vmax=0.003)
plt.title('Reference parameter map')
plt.axis('off')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(est_prm_tv[:, :, mid_slice], cmap='jet', vmin=0, vmax=0.003)
plt.title('Estimated parameters map (TV)')
plt.axis('off')
plt.colorbar()
plt.tight_layout()
plt.show()