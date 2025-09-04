Quantitative MRI: Implementation of IVIM and IVIM-DKI Models with Total Variation Regularization

Overview: This project (Summer 2025, IIT Delhi) implements voxel-wise bi-exponential (IVIM) and hybrid IVIM-DKI diffusion MRI models with advanced Total Variation (TV) regularization for robust parameter estimation. The code (Python) handles DICOM/NIfTI data loading (NiBabel/SimpleITK) and ROI definition, and performs nonlinear least-squares fitting (using scipy.optimize.curve_fit and lmfit) to extract diffusion parameters. Comprehensive synthetic simulations and real patient data analyses were conducted. Quantitative metrics (RMSE, bias, AIC) show that applying TV-regularization significantly stabilizes parameter maps under low SNR.

Motivation: Advanced quantitative diffusion MRI (IVIM/IVIM-DKI) is increasingly used to noninvasively characterize tissue microstructure and perfusion. Incorporating kurtosis (DKI) captures non-Gaussian diffusion, and IVIM separates diffusion from microvascular flow. However, voxel-wise fitting can be highly sensitive to noise, yielding speckled parameter maps. TV-regularization enforces spatial smoothness, reducing errors and preserving edges. This work contributes to a cutting-edge field; IVIM-DKI with TV is being explored globally (e.g. cancer and liver imaging), and at IIT Delhi’s CBME it helps position the group at the forefront of quantitative MRI research in India.

Methodology: We developed a Python-based pipeline for diffusion MRI analysis. DICOM/NIfTI scans are loaded via MRIcron (for ROI masks) and NiBabel/SimpleITK (to NumPy arrays). Intensity histograms and summary statistics of ROIs guide data quality checks. Nonlinear model fitting is performed voxel-wise: initial parameter guesses (from literature and simple models) seed a Levenberg-Marquardt or trust-region fit using scipy.optimize.curve_fit and lmfit. Both standard fitting and TV-penalized fitting (via an existing IDTV model toolbox) were implemented and compared. Synthetic data (a custom 2D “bulls-eye” phantom) were generated for validation, enabling controlled analysis of parameter recovery. Model performance is quantitatively assessed by RMSE, relative bias, parameter error, and Akaike Information Criterion (AIC) across SNR levels.

Data and Preprocessing: Real MRI data from a multi-organ clinical dataset (14 patients with liver-HCC, lymphoma, prostate, and rectal lesions) were used for pilot testing. ROIs were drawn in MRIcron and exported as masks. We converted images to NumPy arrays (NiBabel) and performed any necessary resampling (SimpleITK). Example ROI analysis (signal histograms) helped verify tissue intensity ranges and guide thresholding. All patient data processing steps (including NIfTI I/O and ROI extraction) are fully scripted for reproducibility.

Models Implemented: The IVIM (bi-exponential) model 
𝑆
(
𝑏
)
=
𝑆
0
(
𝑓
𝑒
−
𝑏
𝐷
∗
+
(
1
−
𝑓
)
𝑒
−
𝑏
𝐷
)
S(b)=S
0
	​

(fe
−bD
∗
+(1−f)e
−bD
) and the IVIM-DKI model (which adds a kurtosis term 
𝐾
K to capture non-monoexponential decay) were implemented following standard formulations. TV-regularization was applied to each parameter map by minimizing spatial gradients (piecewise-smooth constraint). We leveraged the IDTV toolbox (Kayal et al.) for the TV-penalized fitting, which consistently produced smoother, more stable maps than standard fitting. In code, each model is fit voxel-wise to generate maps of diffusion coefficient 
𝐷
D, pseudo-diffusion 
𝐷
∗
D
∗
, perfusion fraction 
𝑓
f, and kurtosis 
𝐾
K.

Results and Visualizations:
Figure: Example diffusion-weighted MRI (brain DWI axial slice). This illustrates the kind of DWI data used for fitting (here b=0 image). In our framework, model fitting produces spatial parameter maps and residuals for such data.

Synthetic Simulations: We conducted Monte Carlo simulations at SNR levels 15, 25, 40, and 60. As expected, all IVIM-DKI parameters show decreasing RMSE with increasing SNR (e.g. D*’s RMSE dropped ~50% from SNR15 to 60). Bias in D* and 
𝑓
f also decreased with higher SNR, while kurtosis 
𝐾
K showed a persistent positive bias (systematic overestimation) even at high SNR. Model quality (AIC) improved at higher SNR, indicating better fits in cleaner data. Importantly, TV-regularized fitting (IDTV) produced noticeably smoother and more consistent parameter maps than standard fitting (see, e.g., reduced high-frequency noise in 
𝐷
D and 
𝑓
f maps). Quantitative ROI errors confirm that TV yields lower RMSE and variance, especially at low SNR.

Clinical Data: The IVIM-DKI pipeline was applied to patient scans (tumor vs. healthy tissue). Box plots of fitted parameters (D*, D, f, K) across organs showed visible trends but no statistically significant differences (p>0.05) between benign and malignant groups in this small cohort. This underscores the challenge of limited sample size and inter-patient variability. Nevertheless, the workflow successfully generated all parameter maps for multi-organ data, demonstrating end-to-end viability. Residual difference maps (raw vs. fitted) confirmed that the models captured the dominant diffusion decay (residuals were largely unstructured).

Tools and Libraries Used: The project was implemented in Python 3, utilizing scientific libraries for MRI data and optimization: NiBabel (reading/writing NIfTI) and SimpleITK (image processing); numerical libraries NumPy/SciPy for array operations and optimization (we used scipy.optimize.curve_fit); lmfit for advanced fitting; and Matplotlib/Seaborn for plotting. The IDTV toolbox (MATLAB) was used via Python bindings for TV-penalized fitting. High-performance computing resources (Linux cluster with Slurm) were leveraged to parallelize the large simulation batches and fit many images efficiently. (Multiple Slurm jobs ran independent ROI fits and synthetic runs, illustrating experience with cluster workflows.) All analyses were tracked via version-controlled scripts (Git on GitHub), and random seeds/pipeline steps are fixed to ensure reproducibility.

How to Run:

Clone the repository: git clone https://github.com/Artamta/Summer_Internsip_IITD.git (see the python/ directory for code).

Setup environment: Install Python 3 and dependencies (e.g. pip install numpy scipy lmfit nibabel SimpleITK matplotlib). A requirements.txt is provided.

Prepare data: Place DICOM/NIfTI files in the expected data folder; ROI masks can be defined with MRIcron or provided. Synthetic data scripts generate phantom images automatically.

Run fitting scripts: Examples:

python run_simulation.py to execute SNR simulations and produce metrics (outputs saved as CSV).

python run_fitting.py --input DIR --output DIR to fit IVIM or IVIM-DKI to actual data (produces parameter maps).

Optional: Use the provided Slurm scripts (sbatch run_jobs.sh) to distribute tasks on a compute cluster.

Review results: Plots (e.g. RMSE vs. SNR) and maps are saved as PNGs. Check log files for fit statistics. The analysis is fully documented in code comments.

Contributions:

Model Fitting Development: Coded custom voxel-wise fitting routines for IVIM and IVIM-DKI models using SciPy and lmfit. Designed a two-step initialization strategy to improve convergence (e.g. fitting simpler mono-exponential first).

Synthetic Phantom Analysis: Created a 2D concentric “bulls-eye” phantom with known parameters. Generated noisy IVIM-DKI signals and used ROI masking to evaluate parameter recovery. This testbed enabled validation of TV-regularization effects under controlled noise.

Simulation Studies: Performed extensive Monte Carlo simulations across SNR levels and summarized results in terms of RMSE, bias, and AIC. Plotted key trends (e.g. error vs. SNR) to quantify model reliability.

Real Patient Data Evaluation: Applied the pipeline to clinical IVIM-DKI scans from multiple organs. Extracted ROI statistics and produced figures (parameter maps, boxplots) for tumor vs. normal tissue. Statistical analysis (t-tests) was implemented to compare groups.

Integration and Visualization: Developed plotting routines to visualize raw vs. fitted data (Fig. 3 style), ROI intensity distributions, and fitting residuals. The IDTV toolbox was integrated to apply TV-regularization, demonstrating much smoother output maps.

Scientific Rigor: Ensured reproducibility by using seed control and documenting every step. Automated as much of the workflow as possible (scripts for preprocessing, fitting, and plotting). The complete processing pipeline, data, and results are archived in the linked GitHub repository for transparency.

Future Work: Our study establishes a solid framework but highlights several next steps. As suggested in the report, future work should include expanding patient cohorts to increase statistical power and better characterize tissue differences. Standardizing the acquisition protocol and analysis pipeline across sites will further improve consistency. Organ-specific diagnostic criteria (e.g. parameter thresholds) and correlation with biopsy/histology could enhance clinical utility. We also plan to explore advanced methods: for example, integrating machine learning to identify multi-parametric biomarkers, or accelerating fitting via learned priors. Additional model refinements (e.g. incorporating multiple b-value shells or multi-compartment models) and further HPC optimizations (GPU acceleration or more efficient Slurm workflows) are also possible extensions.

Tools & Reproducibility: All code, data processing scripts, and plotting routines are available in the GitHub repository. The repository includes example data and a README with detailed usage. Results reported here (figures, metrics) can be reproduced by following the instructions above. This ensures scientific rigor and transparency in our workflow. The project reflects strong reproducibility (fixed seeds, documented environments) and leverages HPC resources to manage computational load.

References: Key methods and background are cited in-line (Le Bihan et al. IVIM, SciPy docs, etc.). The GitHub repo contains links to any external tool documentation used (e.g. NiBabel, IDTV toolbox).
