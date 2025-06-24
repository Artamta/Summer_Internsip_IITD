import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
import pandas as pd

def sim_ivim_dki(bvals, D, D_star, f, k, shape):
    """Simulate IVIM-DKI signal for all voxels and b-values."""
    y_pred = np.zeros(shape)
    for idx, b in enumerate(bvals):
        y_pred[..., idx] = (f * np.exp(-b * D_star)) + (1 - f) * np.exp(
            (-b * D) + (1 / 6) * k * (b ** 2) * (D ** 2)
        )
    return y_pred

# --- Setup ---
bvals = [0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000]
snrs = [15, 25, 40, 60]
datas = [1, 2, 3, 4, 5]
simulations = {
    1: {"param": "D",      "sim_str": "I",   "ref_map": "Simulation-I_D_map.nii"},
    2: {"param": "D_star", "sim_str": "II",  "ref_map": "Simulation-II_Dstar_map.nii"},
    3: {"param": "f",      "sim_str": "III", "ref_map": "Simulation-III_f_map.nii"},
    4: {"param": "k",      "sim_str": "IV",  "ref_map": "Simulation-IV_k_map.nii"},
}
base_ref = "/Users/ayush/Desktop/project-internsip/reference_maps"
base_data = "/Users/ayush/Desktop/project-internsip/Simulation data"
base_est = "/Users/ayush/Desktop/project-internsip/new_output"

results = []

# --- Main Loop ---
for sim_num, sim_info in simulations.items():
    # Load reference parameter map for this simulation
    ref_path = os.path.join(base_ref, sim_info["ref_map"])
    ref_map = nib.load(ref_path).get_fdata()
    ref_flat = ref_map.flatten()

    for snr in snrs:
        for data_idx in datas:
            # Build file paths for all estimated parameter maps and signal data
            est_dir = os.path.join(base_est, f"simulation{sim_num}", f"snr{snr}", "Output_Parameter_Maps")
            est_paths = {
                "D":      os.path.join(est_dir, f"Data-{data_idx}_Simulation-{sim_info['sim_str']}_SNR-{snr}_D.nii"),
                "D_star": os.path.join(est_dir, f"Data-{data_idx}_Simulation-{sim_info['sim_str']}_SNR-{snr}_D_star.nii"),
                "f":      os.path.join(est_dir, f"Data-{data_idx}_Simulation-{sim_info['sim_str']}_SNR-{snr}_f.nii"),
                "k":      os.path.join(est_dir, f"Data-{data_idx}_Simulation-{sim_info['sim_str']}_SNR-{snr}_k.nii"),
            }
            y_path = os.path.join(
                base_data,
                f"Simulation-{sim_info['sim_str']}_Nifty-data",
                f"Simulation-{sim_info['sim_str']}_SNR{snr}",
                f"Data-{data_idx}_Simulation-{sim_info['sim_str']}_SNR-{snr}.nii"
            )

            # Skip if any file is missing
            if not all(os.path.exists(p) for p in est_paths.values()) or not os.path.exists(y_path):
                print(f"Missing file for Sim {sim_num}, SNR {snr}, Data {data_idx}")
                continue

            # Load all estimated parameter maps and signal data
            est_maps = {k: nib.load(v).get_fdata() for k, v in est_paths.items()}
            y_data = nib.load(y_path).get_fdata()

            # Select the estimated map for the parameter of interest
            est_param_map = est_maps[sim_info["param"]]
            est_flat = est_param_map.flatten()

            # --- Metrics ---
            rmse = np.sqrt(np.mean((est_flat - ref_flat) ** 2))
            rmse_norm = (rmse / np.mean(ref_flat)) * 100
            rel_bias = np.mean((est_flat - ref_flat) / ref_flat) * 100
            rel_param = np.mean(est_flat / ref_flat)

            # --- AIC/AICc ---
            y_pred = sim_ivim_dki(bvals, est_maps["D"], est_maps["D_star"], est_maps["f"], est_maps["k"], y_data.shape)
            parameters = 4
            n = len(bvals)
            aic_map = np.zeros(est_param_map.shape)
            for i in range(est_param_map.shape[0]):
                for j in range(est_param_map.shape[1]):
                    for k in range(est_param_map.shape[2]):
                        residuals = y_data[i, j, k, :] - y_pred[i, j, k, :]
                        RSS = np.sum(residuals ** 2)
                        aic_map[i, j, k] = 2 * parameters + n * np.log(RSS / n + 1e-8)
            aic = np.nanmean(aic_map)
            aicc = aic + (2 * parameters * (parameters + 1) / (n - parameters - 1))

            # --- Print results ---
            print(f"Sim {sim_num} ({sim_info['param']}), SNR {snr}, Data {data_idx}: "
                  f"RMSE={rmse_norm:.4f}, Bias={rel_bias:.4f}, RelParam={rel_param:.4f}, "
                  f"AIC={aic:.4f}, AICc={aicc:.4f}")

            # --- Save results for CSV ---
            results.append({
                "Simulation": sim_num,
                "Parameter": sim_info["param"],
                "SNR": snr,
                "Data": data_idx,
                "RMSE_norm": rmse_norm,
                "Rel_Bias": rel_bias,
                "Rel_Param": rel_param,
                "AIC": aic,
                "AICc": aicc
            })

# --- Save all results to CSV ---
df = pd.DataFrame(results)
csv_path = "/Users/ayush/Desktop/project-internsip/new_output/accuracy_metrics_all_simulations.csv"
df.to_csv(csv_path, index=False)
print(f"Saved all results to {csv_path}")

# --- Plot all 4 reference and estimated maps for a chosen SNR and Data ---
snr_plot = 60
data_plot = 1

ref_maps = []
est_maps = []
titles = []

for sim_num, sim_info in simulations.items():
    # Reference map
    ref_path = os.path.join(base_ref, sim_info["ref_map"])
    ref_map = nib.load(ref_path).get_fdata()
    ref_maps.append(ref_map)
    titles.append(f"Ref {sim_info['param']}")

    # Estimated map
    est_dir = os.path.join(base_est, f"simulation{sim_num}", f"snr{snr_plot}", "Output_Parameter_Maps")
    est_path = os.path.join(est_dir, f"Data-{data_plot}_Simulation-{sim_info['sim_str']}_SNR-{snr_plot}_{sim_info['param']}.nii")
    if os.path.exists(est_path):
        est_map = nib.load(est_path).get_fdata()
    else:
        est_map = np.zeros_like(ref_map)
    est_maps.append(est_map)
    titles.append(f"Est {sim_info['param']}")

mid = ref_maps[0].shape[2] // 2
plt.figure(figsize=(16, 8))
for i in range(4):
    plt.subplot(2, 4, i+1)
    plt.imshow(ref_maps[i][:, :, mid], cmap='jet')
    plt.title(titles[2*i])
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(2, 4, i+5)
    plt.imshow(est_maps[i][:, :, mid], cmap='jet')
    plt.title(titles[2*i+1])
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig("/Users/ayush/Desktop/project-internsip/new_output/all_maps_overview.png", dpi=300)
plt.show()
print("Saved overview plot of all reference and estimated maps.")