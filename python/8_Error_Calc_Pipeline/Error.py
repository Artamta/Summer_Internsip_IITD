import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def compute_metrics(ref_data, est_data, mask):
    """Compute RMSE, normalized RMSE, relative bias, and relative parameter."""
    ref_vals = ref_data[mask]
    est_vals = est_data[mask]
    rmse = np.sqrt(np.mean((est_vals - ref_vals) ** 2))
    rmse_norm = (rmse / np.mean(ref_vals)) * 100
    rel_bias = (np.mean(est_vals - ref_vals) / np.mean(ref_vals)) * 100
    rel_param = np.mean(est_vals / ref_vals)
    return rmse, rmse_norm, rel_bias, rel_param

def compute_mean_aic(aic_data, mask):
    """Compute mean AIC value over the mask."""
    aic_vals = aic_data[mask]
    return np.nanmean(aic_vals)

def visualize_maps(ref_data, est_data, mask, title_suffix=""):
    """Show reference, estimated, and mask slices."""
    mid_slice = ref_data.shape[2] // 2
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(ref_data[:, :, mid_slice], cmap='gray')
    plt.title(f'Reference{title_suffix}')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(est_data[:, :, mid_slice], cmap='gray')
    plt.title(f'Estimated{title_suffix}')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(mask[:, :, mid_slice], cmap='gray')
    plt.title('Mask')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main(ref_path, est_path, aic_path=None, visualize=True):
    # ---- Load data ----
    ref_img = nib.load(ref_path)
    est_img = nib.load(est_path)
    ref_data = ref_img.get_fdata()
    est_data = est_img.get_fdata()

    # ---- Generate mask: Only compare where reference is nonzero ----
    mask = ref_data > 0

    # ---- Metrics ----
    rmse, rmse_norm, rel_bias, rel_param = compute_metrics(ref_data, est_data, mask)
    print(f"RMSE: {rmse:.4f}")
    print(f"Normalized RMSE (%): {rmse_norm:.2f}")
    print(f"Relative Bias (%): {rel_bias:.2f}")
    print(f"Relative Parameter: {rel_param:.4f}")

    # ---- Mean AIC (optional) ----
    if aic_path is not None:
        aic_data = nib.load(aic_path).get_fdata()
        mean_aic = compute_mean_aic(aic_data, mask)
        print(f"Mean AIC: {mean_aic:.2f}")
    else:
        print("AIC map not provided, skipping mean AIC.")

    # ---- Visualization (optional) ----
    if visualize:
        visualize_maps(ref_data, est_data, mask)

if __name__ == "__main__":
    # ---- User: Set your file paths here ----
    ref_path = "/path/to/reference_map.nii"
    est_path = "/path/to/estimated_map.nii"
    aic_path = None  # Set to "/path/to/aic_map.nii" if you want mean AIC, else leave as None

    main(ref_path, est_path, aic_path)