# Voxel‑Wise IVIM / IVIM‑DKI Modeling with Total Variation Regularization

**Indian Institute of Technology Delhi**  
Centre for Biomedical Engineering  
**Summer Research Internship (July 2025)**  
**Author:** Ayush Raj  
**Supervisors:** Dr. Amit Mehndiratta, Dr. Esha Baidya Kayal

---

## Abstract

This repository contains the complete codebase, analysis pipeline, and selected results from my summer research internship at IIT Delhi (CBME), focused on advanced quantitative diffusion MRI. The project implements voxel-wise bi-exponential IVIM and hybrid IVIM-DKI models for diffusion-weighted imaging, with robust parameter estimation using Total Variation (TV) regularization. The workflow covers data I/O (DICOM/NIfTI), ROI selection, nonlinear least-squares fitting (scipy.optimize.curve_fit, lmfit), synthetic simulations across SNR levels, and quantitative evaluation using RMSE, bias, and AIC. TV regularization substantially improves the consistency and reliability of parameter maps, especially under low SNR conditions.

- **Code:** [GitHub Python methods](https://github.com/Artamta/Summer_Internsip_IITD/tree/main/python)
- **Simulation results:** [Google Sheet](https://docs.google.com/spreadsheets/d/1XKLll_YyjdOcbk6JbbiA-nZcQGm2m198/edit?usp=drive_link&ouid=106252698340703105305&rtpof=true&sd=true)
- **Patient results:** [Google Sheet](https://docs.google.com/spreadsheets/d/1ksJqD7cg9zU0MC4sbZ6edHCH5_-vi0tc/edit?usp=drive_link&ouid=106252698340703105305&rtpof=true&sd=true)

---

## Table of Contents

- [Project Motivation](#project-motivation)
- [Scientific Background](#scientific-background)
- [Methodology](#methodology)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation & Usage](#installation--usage)
- [Reproducibility](#reproducibility)
- [Data & Ethics](#data--ethics)
- [References](#references)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Contact](#contact)

---

## Project Motivation

Quantitative diffusion MRI modeling, especially using advanced techniques such as IVIM-DKI with TV regularization, is at the forefront of biomedical imaging research. These methods enable improved tissue characterization, early disease detection, and personalized treatment planning. The Centre for Biomedical Engineering (CBME) at IIT Delhi is pioneering research in this domain, and this project contributes to the development and validation of robust computational frameworks for quantitative MRI.

---

## Scientific Background

**IVIM Model:**  
Separates true molecular diffusion from microvascular perfusion using multi-b-value DWI.  
Equation:  
_S(b) = S₀ [ f·exp(−bD_) + (1−f)·exp(−bD) ]_  
where S₀ is baseline signal, D is diffusion coefficient, D_ is pseudo-diffusion coefficient, and f is perfusion fraction.

**DKI Model:**  
Extends the mono-exponential model to capture non-Gaussian diffusion via kurtosis (K):  
_S(b)/S₀ ≈ exp( −bD + (1/6)b²D²K )_

**Hybrid IVIM-DKI:**  
Combines perfusion and kurtosis effects for richer tissue characterization.

**Total Variation Regularization:**  
Promotes spatial smoothness in parameter maps, reducing noise while preserving anatomical edges.

---

## Methodology

1. **Data Loading & Preprocessing**

   - DICOM/NIfTI images loaded using NiBabel and SimpleITK.
   - ROI selection via MRIcron; masks imported for analysis.

2. **Histogram & ROI Analysis**

   - Intensity histograms and kernel density estimates computed for tumor ROIs.
   - Summary statistics (mean, median, std) for both raw and fitted parameters.

3. **Visualization**

   - Matplotlib used for single-slice visualizations: raw images, fitted parameter maps, and residuals.

4. **Nonlinear Model Fitting**

   - Voxel-wise nonlinear least-squares fitting using `scipy.optimize.curve_fit` and `lmfit`.
   - Physiological bounds and robust initial guesses for convergence.
   - RMSE, bias, and AIC used for quantitative assessment.

5. **Synthetic Phantom Generation**

   - 2D bulls-eye phantom created in NumPy for controlled validation.
   - Composite visualizations of reference, estimated, and masked maps.

6. **Custom IVIM-Hybrid Model**

   - Developed and evaluated custom implementations for IVIM and IVIM-DKI.
   - Compared against toolbox (IDTV) with TV regularization.

7. **Total Variation Regularization**

   - Used IDTV model toolbox for TV-regularized fitting and statistical analysis.

8. **SNR Simulations**

   - Synthetic data generated at SNR levels 15, 25, 40, 60.
   - Parameter maps and ROI statistics computed for each noise regime.

9. **Real Patient Data Analysis**
   - Processed IVIM data from tumor and healthy tissue regions.
   - Extracted comprehensive ROI statistics for clinical evaluation.

**Code Availability:**  
All code for data processing, model fitting, and analysis is available in the [python directory](https://github.com/Artamta/Summer_Internsip_IITD/tree/main/python).

---

## Results

### Synthetic Simulation Data

- **Validation:** Monte Carlo simulations at multiple SNR levels.
- **Metrics:** RMSE, relative bias, relative parameter error, AIC.
- **Findings:**
  - D recovered with low bias across SNRs.
  - D\* estimation improved with SNR but remained challenging.
  - f and K showed systematic biases.
  - TV regularization reduced RMSE and parameter variability by 20–40%.

### Clinical Data Analysis

- **Dataset:** 14 patients, 4 organ systems (Liver HCC, Lymphoma, Prostate, Rectum).
- **Comparisons:** Benign vs. tumor tissue parameter distributions.
- **Statistical Analysis:** Two-sample t-tests; no significant differences detected (sample size limitation).
- **Trends:** Benign tissues generally showed higher D and f; tumor tissues had higher kurtosis.

### Visualizations

You can add figures here, e.g.:

![Example Parameter Map](docs/figures/parameter_map_example.png)
_Figure: Comparison of standard and TV-regularized parameter maps._

---

## Repository Structure

```
python/           # Scripts and notebooks for analysis
Results/          # Output tables, figures, logs
docs/             # Manuscript, figures, supplementary materials
data/             # (Not tracked) Raw, interim, processed data
README.md         # This file
requirements.txt  # Python dependencies
.gitignore        # Excludes large datasets, output folders
```

---

## Installation & Usage

**Environment Setup (macOS):**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy scipy matplotlib seaborn nibabel simpleitk lmfit scikit-image pandas tqdm
```

**Typical Workflow:**

1. Prepare inputs: NIfTI DWI, b-values, ROI masks.
2. Run analysis:
   ```bash
   python python/run_simulation.py
   python python/run_fitting.py --dwi data/dwi.nii.gz --bvals data/dwi.bval --mask data/mask.nii.gz --model ivim-dki --tv --lambda 0.1
   ```
3. Inspect results: Output tables, figures, and LaTeX summaries in `Results/`.

---

## Reproducibility

- Fixed random seeds in simulation modules.
- Explicit parameter bounds and initial guesses.
- ROI-based evaluation to minimize confounds.
- Metrics: RMSE, normalized RMSE, relative bias, relative parameter, AIC.

---

## Data & Ethics

- Clinical data are not distributed; only summary tables are shared.
- All patient data handled under institutional ethical guidelines and anonymization protocols.

---

## References

1. Malagi, A. V., et al. (2023). IVIM‑DKI with parametric reconstruction… Clinical Imaging 101. [doi](https://doi.org/10.1016/j.clinimag.2023.05.011)
2. Malagi, A. V., et al. (2019). Effect of combination and number of b values… MAGMA 32, 567–579. [doi](https://doi.org/10.1007/s10334-019-00764-0)
3. Le Bihan, D., et al. (1988). Separation of diffusion and perfusion in intravoxel incoherent motion MR imaging. Radiology 168(2), 497–505. [doi](https://doi.org/10.1148/radiology.168.2.3393671)
4. NiBabel (neuroimaging I/O). [website](https://nipy.org/nibabel/)
5. SciPy curve_fit. [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)
6. NIBIB MRI overview. [website](https://www.nibib.nih.gov/science-education/science-topics/magnetic-resonance-imaging-mri)

---

## Acknowledgements

- Centre for Biomedical Engineering, IIT Delhi
- Dr. Amit Mehndiratta, Dr. Esha Baidya Kayal
- Developers of the IDTV model toolbox

---

## License

If you intend open distribution, add an MIT or BSD license file. Otherwise, this repository is “All rights reserved” by default.

---

## Contact

For questions or collaboration, open an issue or contact:  
Ayush Raj — Research Intern, CBME, IIT Delhi  
Code: [GitHub Python methods](https://github.com/Artamta/Summer_Internsip_IITD/tree/main/python)

---

## Thank You

Thank you for your interest in this project and for your valuable time and consideration.
