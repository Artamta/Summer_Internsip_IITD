import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
import os
warnings.filterwarnings('ignore')

# Set matplotlib style for publication-quality plots
plt.style.use('default')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3

def ivim_model(b, S0, f, D, Dstar):
    """
    Bi-exponential IVIM model.
    S(b) = S0 * [f * exp(-b * D*) + (1-f) * exp(-b * D)]
    """
    return S0 * (f * np.exp(-b * Dstar) + (1 - f) * np.exp(-b * D))

def ivim_dki_model(b, S0, f, D, Dstar, K):
    """
    Hybrid IVIM-DKI model.
    S(b) = S0 * [f * exp(-b * D*) + (1-f) * exp(-b * D + (1/6) * b² * D² * K)]
    """
    return S0 * (f * np.exp(-b * Dstar) + (1 - f) * np.exp(-b * D + (1/6) * b**2 * D**2 * K))

# Realistic b-values for brain IVIM studies
b_values = np.array([0, 50, 100, 200, 400, 600, 800, 1000, 1500, 2000])

# Realistic parameters for brain tissue
true_S0 = 1000
true_f = 0.12  # Perfusion fraction (typical for brain)
true_D = 0.0008  # ADC in mm²/s (typical for brain)
true_Dstar = 0.015  # Pseudo-diffusion coefficient
true_K = 0.7  # Kurtosis parameter

# Generate realistic noisy data
np.random.seed(42)
noise_level = 25
signal_clean = ivim_model(b_values, true_S0, true_f, true_D, true_Dstar)
signal_noisy = signal_clean + np.random.normal(0, noise_level, len(b_values))
signal_noisy = np.maximum(signal_noisy, 5)  # Ensure positive values

print("Generating synthetic brain diffusion data...")
print(f"True parameters: S0={true_S0}, f={true_f:.3f}, D={true_D:.6f} mm²/s, D*={true_Dstar:.4f} mm²/s")

# IVIM fitting
initial_guess_ivim = [np.max(signal_noisy), 0.1, 0.001, 0.01]
bounds_ivim = ([0, 0, 0.0001, 0.001], [np.inf, 0.8, 0.003, 0.1])

try:
    popt_ivim, pcov_ivim = curve_fit(
        ivim_model, b_values, signal_noisy,
        p0=initial_guess_ivim, bounds=bounds_ivim, maxfev=10000
    )
    S0_ivim, f_ivim, D_ivim, Dstar_ivim = popt_ivim
    
    # Calculate R²
    ss_res_ivim = np.sum((signal_noisy - ivim_model(b_values, *popt_ivim)) ** 2)
    ss_tot = np.sum((signal_noisy - np.mean(signal_noisy)) ** 2)
    r2_ivim = 1 - (ss_res_ivim / ss_tot)
    
    print(f"IVIM fit: S0={S0_ivim:.0f}, f={f_ivim:.3f}, D={D_ivim:.6f}, D*={Dstar_ivim:.4f}, R²={r2_ivim:.3f}")
    
except Exception as e:
    print(f"IVIM fitting failed: {e}")
    popt_ivim = None

# IVIM-DKI fitting
initial_guess_dki = [np.max(signal_noisy), 0.1, 0.001, 0.01, 0.5]
bounds_dki = ([0, 0, 0.0001, 0.001, 0], [np.inf, 0.8, 0.003, 0.1, 2.5])

try:
    popt_dki, pcov_dki = curve_fit(
        ivim_dki_model, b_values, signal_noisy,
        p0=initial_guess_dki, bounds=bounds_dki, maxfev=10000
    )
    S0_dki, f_dki, D_dki, Dstar_dki, K_dki = popt_dki
    
    # Calculate R²
    ss_res_dki = np.sum((signal_noisy - ivim_dki_model(b_values, *popt_dki)) ** 2)
    r2_dki = 1 - (ss_res_dki / ss_tot)
    
    print(f"IVIM-DKI fit: S0={S0_dki:.0f}, f={f_dki:.3f}, D={D_dki:.6f}, D*={Dstar_dki:.4f}, K={K_dki:.3f}, R²={r2_dki:.3f}")
    
except Exception as e:
    print(f"IVIM-DKI fitting failed: {e}")
    popt_dki = None

# Create smooth curves for plotting
b_smooth = np.linspace(0, max(b_values), 300)

# Generate fitted curves
if popt_ivim is not None:
    fitted_ivim = ivim_model(b_smooth, *popt_ivim)
    residuals_ivim = signal_noisy - ivim_model(b_values, *popt_ivim)
else:
    fitted_ivim = None
    residuals_ivim = None

if popt_dki is not None:
    fitted_dki = ivim_dki_model(b_smooth, *popt_dki)
    residuals_dki = signal_noisy - ivim_dki_model(b_values, *popt_dki)
else:
    fitted_dki = None
    residuals_dki = None

# Create the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

# Main signal plot
ax1.scatter(b_values, signal_noisy, color='black', s=60, alpha=0.8, zorder=5, label='Observed Signal')

if fitted_ivim is not None:
    ax1.plot(b_smooth, fitted_ivim, 'b-', linewidth=2.5, label=f'IVIM Fit (R² = {r2_ivim:.3f})', alpha=0.9)

if fitted_dki is not None:
    ax1.plot(b_smooth, fitted_dki, 'r--', linewidth=2.5, label=f'IVIM-DKI Fit (R² = {r2_dki:.3f})', alpha=0.9)

ax1.set_ylabel('Signal Intensity', fontweight='bold')
ax1.set_title('Voxel-wise Nonlinear Model Fitting: IVIM vs IVIM-DKI', fontsize=14, fontweight='bold', pad=20)
ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.set_xlim(-50, max(b_values) + 100)
ax1.set_ylim(0, max(signal_noisy) * 1.1)

# Add parameter text boxes
if popt_ivim is not None:
    ivim_text = f'IVIM: D = {D_ivim:.2e} mm²/s\nf = {f_ivim:.3f}, D* = {Dstar_ivim:.3f} mm²/s'
    ax1.text(0.02, 0.65, ivim_text, transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

if popt_dki is not None:
    dki_text = f'IVIM-DKI: D = {D_dki:.2e} mm²/s\nf = {f_dki:.3f}, D* = {Dstar_dki:.3f} mm²/s\nK = {K_dki:.3f}'
    ax1.text(0.02, 0.35, dki_text, transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

# Residuals plot - FIXED: Handle LineCollection properly
if residuals_ivim is not None:
    markerline, stemlines, baseline = ax2.stem(b_values, residuals_ivim, linefmt='b-', markerfmt='bo', basefmt=' ', 
             label='IVIM Residuals')
    # Apply alpha to the returned objects
    markerline.set_alpha(0.7)
    stemlines.set_alpha(0.7)  # stemlines is a LineCollection, not iterable

if residuals_dki is not None:
    markerline, stemlines, baseline = ax2.stem(b_values, residuals_dki, linefmt='r--', markerfmt='rs', basefmt=' ', 
             label='IVIM-DKI Residuals')
    # Apply alpha to the returned objects
    markerline.set_alpha(0.7)
    stemlines.set_alpha(0.7)  # stemlines is a LineCollection, not iterable


ax2.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.7)
ax2.set_xlabel('b-value (s/mm²)', fontweight='bold')
ax2.set_ylabel('Residuals', fontweight='bold')
ax2.set_title('Fitting Residuals Analysis', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.set_xlim(-50, max(b_values) + 100)

# Calculate and display RMSE
if residuals_ivim is not None:
    rmse_ivim = np.sqrt(np.mean(residuals_ivim**2))
    ax2.text(0.7, 0.8, f'RMSE_IVIM = {rmse_ivim:.1f}', transform=ax2.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))

if residuals_dki is not None:
    rmse_dki = np.sqrt(np.mean(residuals_dki**2))
    ax2.text(0.7, 0.5, f'RMSE_DKI = {rmse_dki:.1f}', transform=ax2.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.6))

plt.tight_layout()

# Create results directory if it doesn't exist
results_dir = "/Users/ayush/Desktop/project-internsip/Results/16_b_report"
os.makedirs(results_dir, exist_ok=True)

# Save the figure
output_path = os.path.join(results_dir, 'single_voxel_fit.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nFigure saved as '{output_path}'")

plt.show()

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("This visualization demonstrates:")
print("• Voxel-wise nonlinear least-squares fitting")
print("• Parameter bounds implementation for stability")
print("• Model comparison (IVIM vs IVIM-DKI)")
print("• Quantitative fit assessment (R², RMSE)")
print("• Publication-quality visualization")
