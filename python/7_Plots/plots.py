import os
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt

# --- Paths ---
toolbox_dir = "/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps"
my_dir = "/Users/ayush/Desktop/project-internsip/Results/6_Aic_CALC"
plot_dir = "/Users/ayush/Desktop/project-internsip/Results/7_Plots"
os.makedirs(plot_dir, exist_ok=True)

# Reference f_map for normalization and masking
ref_f_path = "/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_f_map.nii"
ref_f_img = nb.load(ref_f_path)
ref_f_data = ref_f_img.get_fdata()
slice_idx = ref_f_data.shape[2] // 2

# --- Load all maps ---
def load_map(path):
    return nb.load(path).get_fdata()

# Toolbox maps
f_toolbox = load_map(os.path.join(toolbox_dir, "Data-1_Simulation-III_SNR-60_f.nii"))
Dstar_toolbox = load_map(os.path.join(toolbox_dir, "Data-1_Simulation-III_SNR-60_D_star.nii"))
Dslow_toolbox = load_map(os.path.join(toolbox_dir, "Data-1_Simulation-III_SNR-60_D.nii"))
k_toolbox = load_map(os.path.join(toolbox_dir, "Data-1_Simulation-III_SNR-60_k.nii"))

# Your maps
f_my = load_map(os.path.join(my_dir, "f.nii.gz"))
Dstar_my = load_map(os.path.join(my_dir, "Dstar.nii.gz"))
Dslow_my = load_map(os.path.join(my_dir, "D.nii.gz"))
k_my = load_map(os.path.join(my_dir, "k.nii.gz"))

# --- Mask: Only show voxels where reference f_map is nonzero ---
mask = ref_f_data > 0

def masked_minmax(data, mask):
    masked = data[mask]
    if masked.size == 0:
        return 0, 1
    return np.nanmin(masked), np.nanmax(masked)

def plot_f_maps(ref_f, toolbox_f, my_f, mask, fname):
    plt.figure(figsize=(18, 5))
    titles = ["Reference f_map", "Toolbox f_map", "Voxelwise f_map"]
    cmaps = ['viridis'] * 3
    maps = [ref_f, toolbox_f, my_f]
    for i, (data, title, cmap) in enumerate(zip(maps, titles, cmaps)):
        plt.subplot(1, 3, i+1)
        vmin, vmax = masked_minmax(data, mask)
        data_plot = np.where(mask, data, np.nan)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6
        im = plt.imshow(data_plot[:, :, slice_idx], cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        plt.title(title)
        plt.axis('off')
        plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, fname), bbox_inches='tight', dpi=200)
    plt.close()

def plot_two_maps(toolbox_map, my_map, mask, fname, param_name, cmap):
    plt.figure(figsize=(12, 5))
    titles = [f"Toolbox {param_name}", f"Voxelwise {param_name}"]
    maps = [toolbox_map, my_map]
    for i, (data, title) in enumerate(zip(maps, titles)):
        plt.subplot(1, 2, i+1)
        vmin, vmax = masked_minmax(data, mask)
        data_plot = np.where(mask, data, np.nan)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6
        im = plt.imshow(data_plot[:, :, slice_idx], cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        plt.title(title)
        plt.axis('off')
        plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, fname), bbox_inches='tight', dpi=200)
    plt.close()

# --- Plot f_maps (3-panel) ---
plot_f_maps(ref_f_data, f_toolbox, f_my, mask, "f_maps_subplot.png")

# --- Plot D* maps (2-panel) ---
plot_two_maps(Dstar_toolbox, Dstar_my, mask, "Dstar_maps_subplot.png", "D* map", 'plasma')

# --- Plot D maps (2-panel) ---
plot_two_maps(Dslow_toolbox, Dslow_my, mask, "D_maps_subplot.png", "D map", 'magma')

# --- Plot k maps (2-panel) ---
plot_two_maps(k_toolbox, k_my, mask, "k_maps_subplot.png", "k map", 'cividis')

print(f"All subplot images saved to {plot_dir}")