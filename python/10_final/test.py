import numpy as np
import nibabel as nb
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#Function for IVIMDKI Hybrid
def ivim_dki_model(b,f,D_star,D_slow,k):
     return f * np.exp(-b * D_star) + (1 - f) * np.exp(-b * D_slow + (1/6) * (b ** 2) * (D_slow ** 2) * k)

#importing files paths:
original_fmap_Path="/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_f_map.nii"
#main-FILE
snr60="/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_SNR60/Data-1_Simulation-III_SNR-60.nii"

#output_by_toolbox
snr60_toolbox_out_D_star="/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-III_SNR-60_D_star.nii"
snr60_toolbox_out_Fmap="/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-III_SNR-60_f.nii"
snr60_toolbox_out_D="/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-III_SNR-60_D.nii"
snr60_toolbox_out_K="/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-III_SNR-60_k.nii"

#opening files--tool_box
original_fmap = nb.load(original_fmap_Path)
snr60_data=nb.load(snr60)
fmap_toolbox=nb.load(snr60_toolbox_out_Fmap)
dmap_toolbox=nb.load(snr60_toolbox_out_D)
d_star_map_toolbox=nb.load(snr60_toolbox_out_D_star)
k_map_toolbox=nb.load(snr60_toolbox_out_K)

# --- Fitting settings ---
bvals = np.array([0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000])
bounds = ([0.0001, 0.0001, 0.001, 0.01], [0.05, 0.5, 1, 3])
p0 = [0.013, 0.013, 0.23, 1.1]

# Output arrays
data=snr60_data.get_fdata()
shape=data.shape[:3]
f_map_voxelwise=np.full(shape,np.nan)
Dstar_map_voxelwise=np.full(shape,np.nan)
Dslow_map_voxelwise=np.full(shape,np.nan)
k_map_voxelwise=np.full(shape,np.nan)

#Voxelwise Fitting
for x in range(shape[0]):
     for y in range(shape[1]):
          for z in range(shape[2]):
               signal=data[x,y,z,:]
               





