import os
import numpy as np
import nibabel as nib
import pandas as pd

def sim_ivim_dki(b_val, est_d, est_dp, est_f, est_k, y_data_shape):
    y_predicted = np.zeros(y_data_shape)
    for k in range(np.size(b_val)):
        y_predicted[:,:,:,k] = (est_f * np.exp(-b_val[k] * est_dp)) + (1 - est_f) * np.exp(
            (-b_val[k] * est_d) + (1 / 6) * est_k * np.square(-b_val[k]) * np.square(est_d))
    return y_predicted

# Simulation settings
simulations = {
    1: {
        "param": "D",
        "sim_dir": "/Users/ayush/Desktop/project-internsip/output/simulation1",
        "file_pattern": "Data-{}_Simulation-I_SNR-{}_{}.nii",
        "ref_map": "/Users/ayush/Desktop/project-internsip/reference_maps/Simulation-I_D_map.nii",
        "sim_str": "I"
    },
    2: {
        "param": "D_star",
        "sim_dir": "/Users/ayush/Desktop/project-internsip/output/simulation2/Output_Parameter_Maps",
        "file_pattern": "Data-{}_Simulation-II_SNR-{}_{}.nii",
        "ref_map": "/Users/ayush/Desktop/project-internsip/reference_maps/Simulation-II_Dstar_map.nii",
        "sim_str": "II"
    },
    3: {
        "param": "f",
        "sim_dir": "/Users/ayush/Desktop/project-internsip/output/simulation3/Output_Parameter_Maps",
        "file_pattern": "Data-{}_Simulation-III_SNR-{}_{}.nii",
        "ref_map": "/Users/ayush/Desktop/project-internsip/reference_maps/Simulation-III_f_map.nii",
        "sim_str": "III"
    },
    4: {
        "param": "k",
        "sim_dir": "/Users/ayush/Desktop/project-internsip/output/simulation4/Output_Parameter_Maps",
        "file_pattern": "Data-{}_Simulation-IV_SNR-{}_{}.nii",
        "ref_map": "/Users/ayush/Desktop/project-internsip/reference_maps/Simulation-IV_k_map.nii",
        "sim_str": "IV"
    }
}

b_val = [0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000]
snr_list = [15, 25, 40, 60]
data_list = [1, 2, 3, 4, 5]
results = []

for sim_num, sim_info in simulations.items():
    param = sim_info["param"]
    sim_dir = sim_info["sim_dir"]
    file_pattern = sim_info["file_pattern"]
    ref_img = nib.load(sim_info["ref_map"])
    ref_prm = ref_img.get_fdata()
    ref_prm_vals = ref_prm.flatten()
    sim_str = sim_info["sim_str"]

    for data_num in data_list:
        for snr in snr_list:
            # Estimated parameter map
            est_path = os.path.join(sim_dir, file_pattern.format(data_num, snr, param))
            # 4D signal path
            y_data_path = f"/Users/ayush/Desktop/project-internsip/Simulation data/Simulation-{sim_str}_Nifty-data/Simulation-{sim_str}_SNR{snr}/Data-{data_num}_Simulation-{sim_str}_SNR-{snr}.nii"

            # For AIC, need all parameter maps
            est_d_path = os.path.join(sim_dir, file_pattern.format(data_num, snr, "D"))
            est_dstar_path = os.path.join(sim_dir, file_pattern.format(data_num, snr, "D_star"))
            est_f_path = os.path.join(sim_dir, file_pattern.format(data_num, snr, "f"))
            est_k_path = os.path.join(sim_dir, file_pattern.format(data_num, snr, "k"))

            # Check all required files
            if not (os.path.exists(est_path) and os.path.exists(y_data_path) and
                    os.path.exists(est_d_path) and os.path.exists(est_dstar_path) and
                    os.path.exists(est_f_path) and os.path.exists(est_k_path)):
                print(f"Skipping Simulation-{sim_num} Data-{data_num}, SNR-{snr} (missing file)")
                continue

            try:
                est_map = nib.load(est_path).get_fdata()
                est_d = nib.load(est_d_path).get_fdata()
                est_dstar = nib.load(est_dstar_path).get_fdata()
                est_f = nib.load(est_f_path).get_fdata()
                est_k = nib.load(est_k_path).get_fdata()
                y_data = nib.load(y_data_path).get_fdata()
            except Exception as e:
                print(f"Error loading files for Simulation-{sim_num} Data-{data_num}, SNR-{snr}: {e}")
                continue

            est_vals = est_map.flatten()

            # RMSE
            rmse = np.sqrt(np.mean((est_vals - ref_prm_vals) ** 2))
            rmse_norm = (rmse / np.mean(ref_prm_vals)) * 100

            # Relative Bias
            rel_bias = np.mean((est_vals - ref_prm_vals) / ref_prm_vals) * 100

            # Relative Parameter
            rel_param = np.mean(est_vals / ref_prm_vals)

            # AIC/AICc
            y_predicted = sim_ivim_dki(b_val, est_d, est_dstar, est_f, est_k, y_data.shape)
            parameters = 4
            n = len(b_val)
            aic_map = np.zeros(est_map.shape)
            for i in range(est_map.shape[0]):
                for j in range(est_map.shape[1]):
                    for k in range(est_map.shape[2]):
                        residuals = y_data[i, j, k, :] - y_predicted[i, j, k, :]
                        RSS = np.sum(residuals ** 2)
                        aic_map[i, j, k] = 2 * parameters + n * np.log(RSS / n + 1e-8)
            aic = np.nanmean(aic_map.flatten())
            aicc = aic + (2*parameters*(parameters+1)/(n-parameters-1))

            results.append({
                "Simulation": sim_num,
                "Data": data_num,
                "SNR": snr,
                "Parameter": param,
                "RMSE_norm": rmse_norm,
                "Rel_Bias": rel_bias,
                "Rel_Param": rel_param,
                "AIC": aic,
                "AICc": aicc
            })

# Save all results to CSV
df = pd.DataFrame(results)
csv_path = "/Users/ayush/Desktop/project-internsip/reference_maps/accuracy_metrics_all_simulations.csv"
df = df.sort_values(['Simulation', 'Data', 'SNR'])
df.to_csv(csv_path, index=False)
print(f"Saved all results for all simulations to {csv_path}")