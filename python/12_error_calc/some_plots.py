import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Simulation settings
simulations = {
    1: {
        "param": "D",
        "sim_dir": "/Users/ayush/Desktop/project-internsip/output/simulation1",
        "file_pattern": "Data-{}_Simulation-I_SNR-{}_D.nii",
        "ref_map": "/Users/ayush/Desktop/project-internsip/reference_maps/Simulation-I_D_map.nii",
        "sim_str": "I"
    },
    2: {
        "param": "D_star",
        "sim_dir": "/Users/ayush/Desktop/project-internsip/output/simulation2/Output_Parameter_Maps",
        "file_pattern": "Data-{}_Simulation-II_SNR-{}_D_star.nii",
        "ref_map": "/Users/ayush/Desktop/project-internsip/reference_maps/Simulation-II_Dstar_map.nii",
        "sim_str": "II"
    },
    3: {
        "param": "f",
        "sim_dir": "/Users/ayush/Desktop/project-internsip/output/simulation3/Output_Parameter_Maps",
        "file_pattern": "Data-{}_Simulation-III_SNR-{}_f.nii",
        "ref_map": "/Users/ayush/Desktop/project-internsip/reference_maps/Simulation-III_f_map.nii",
        "sim_str": "III"
    },
    4: {
        "param": "k",
        "sim_dir": "/Users/ayush/Desktop/project-internsip/output/simulation4/Output_Parameter_Maps",
        "file_pattern": "Data-{}_Simulation-IV_SNR-{}_k.nii",
        "ref_map": "/Users/ayush/Desktop/project-internsip/reference_maps/Simulation-IV_k_map.nii",
        "sim_str": "IV"
    }
}

snr_list = [15, 25, 40, 60]
data_list = [1, 2, 3, 4, 5]
save_dir = "/Users/ayush/Desktop/project-internsip/reference_maps"

for sim_num, sim_info in simulations.items():
    param = sim_info["param"]
    sim_dir = sim_info["sim_dir"]
    file_pattern = sim_info["file_pattern"]
    ref_img = nib.load(sim_info["ref_map"])
    ref_prm = ref_img.get_fdata()
    mid_slice = ref_prm.shape[2] // 2

    fig, axes = plt.subplots(len(data_list), len(snr_list)*2, figsize=(16, 10))
    for i, data_num in enumerate(data_list):
        for j, snr in enumerate(snr_list):
            est_path = os.path.join(sim_dir, file_pattern.format(data_num, snr))
            if os.path.exists(est_path):
                est_map = nib.load(est_path).get_fdata()
            else:
                est_map = np.zeros_like(ref_prm)
            # Reference
            ax_ref = axes[i, j*2]
            im1 = ax_ref.imshow(ref_prm[:, :, mid_slice], cmap='jet')
            ax_ref.set_title(f'Data{data_num} SNR{snr}\nRef')
            ax_ref.axis('off')
            # Estimated
            ax_est = axes[i, j*2+1]
            im2 = ax_est.imshow(est_map[:, :, mid_slice], cmap='jet')
            ax_est.set_title(f'Data{data_num} SNR{snr}\nEst')
            ax_est.axis('off')
    plt.suptitle(f'Simulation {sim_num} ({param}) Reference vs Estimated', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, f"Sim{sim_num}_{param}_summary.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")