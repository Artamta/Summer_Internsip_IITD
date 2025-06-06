import numpy as np
#from scipy.optimize import curvefit
import nibabel as nb
import matplotlib.pyplot as plt


#importing files paths:
original_fmap_Path="/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_f_map.nii"
snr60="/Users/ayush/Desktop/project-internsip/new_work/OneDrive_2_23-05-2025/Simulation-III_Nifty-data/Simulation-III_SNR60/Data-1_Simulation-III_SNR-60.nii"

#output_by_toolbox
snr60_toolbox_out_D_star="/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-III_SNR-60_D_star.nii"
snr60_toolbox_out_Fmap="/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-III_SNR-60_f.nii"
snr60_toolbox_out_D="/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-III_SNR-60_D.nii"
snr60_toolbox_out_K="/Users/ayush/Desktop/project-internsip/Output_Parameter_Maps/Data-1_Simulation-III_SNR-60_k.nii"

#opening files
#tool_box
original_fmap = nb.load(original_fmap_Path)
snr60_data=nb.load(snr60)
fmap_toolbox=nb.load(snr60_toolbox_out_Fmap)
dmap_toolbox=nb.load(snr60_toolbox_out_D)
d_star_map_toolbox=nb.load(snr60_toolbox_out_D_star)
k_map_toolbox=nb.load(snr60_toolbox_out_K)

#Def_AIC

#Voxel


