import numpy as np

def ivim_dki_model(b_values, f, D_star, D_slow, k):
    """IVIM-DKI model function."""
    exp1 = np.exp(np.clip(-b_values * D_star, -100, 100))
    exp2 = np.exp(np.clip(-b_values * D_slow + (1/6) * (b_values ** 2) * (D_slow ** 2) * k, -100, 100))
    result = f * exp1 + (1 - f) * exp2
    return np.clip(result, 0, 2)

def calculate_aic_map_ivim_dki(image_data, f_map, Dstar_map, Dslow_map, k_map, b_values, mask=None):
    """
    Calculate AIC map and mean AIC for IVIM-DKI model.
    image_data: 4D numpy array (x, y, z, b)
    f_map, Dstar_map, Dslow_map, k_map: 3D parameter maps
    b_values: 1D numpy array of b-values
    mask: 3D boolean array (optional)
    Returns: aic_map, mean_aic
    """
    shape = f_map.shape
    aic_map = np.full(shape, np.nan)
    if mask is None:
        mask = np.ones(shape, dtype=bool)
    n = len(b_values)
    k_param = 4
    for idx in np.ndindex(shape):
        if not mask[idx]:
            continue
        signal = image_data[idx[0], idx[1], idx[2], :]
        if np.any(signal > 0) and signal[0] != 0:
            y_true = signal / signal[0]
            if np.isnan(y_true).any() or np.isinf(y_true).any():
                continue
            f = f_map[idx]
            D_star = Dstar_map[idx]
            D_slow = Dslow_map[idx]
            k = k_map[idx]
            y_pred = ivim_dki_model(b_values, f, D_star, D_slow, k)
            residuals = y_true - y_pred
            rss = np.sum(residuals ** 2)
            aic = 2 * k_param + n * np.log(rss / n) if rss > 0 and n > 0 else np.nan
            aic_map[idx] = aic
    mean_aic = np.nanmean(aic_map[mask])
    return aic_map, mean_aic