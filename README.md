# Data Assimilation and Time-Series Forecasting using Diffusion Models and Transformers

This project explores **data assimilation techniques** using a **diffusion model** for reconstructing sparse data and **time-series forecasting** using **Transformer networks**.  

Two forecasting approaches are compared:
1. **EOF-based Forecasting** – leveraging **Empirical Orthogonal Functions (EOF)** for dimensionality reduction.  
2. **Direct Forecasting** – using the **full original dataset** without dimensionality reduction.  

The goal is to analyze the trade-off between **forecasting accuracy** and **computational efficiency** for both approaches.

---

## Key Components

### 1. Data Loading and Preprocessing
- Load and prepare the `u` and `v` velocity component data.
- Perform downsampling to manage data size.
- Combine both components into a single unified data matrix.

### 2. Empirical Orthogonal Functions (EOF) Analysis
- Apply **Truncated Singular Value Decomposition (SVD)** to extract EOFs.
- Identify and retain the dominant modes of variability for reduced representation.

### 3. Transformer-based Time-Series Forecasting
- Implement and train **Transformer models** for forecasting tasks.
- Conduct experiments in both:
  - **Reduced EOF space**, and  
  - **Original full data space.**

### 4. Diffusion Model for Data Assimilation
- Develop a **diffusion-based approach** with **mask constraints**.
- Use it for **reconstructing sparse or missing data**.
- Evaluate reconstruction quality through visual and quantitative metrics.

### 5. Evaluation
- Compare forecasting performance of both methods using:
  - **Test loss**
  - **Correlation coefficient**
- Assess **data reconstruction quality** achieved by the diffusion model.

---

## Technologies Used
- Python (NumPy, SciPy, PyTorch)
- Truncated SVD / PCA
- Transformer Networks
- Diffusion Models for Data Assimilation
- Matplotlib / Seaborn for Visualization

---

## 📈 Results Summary
- **EOF-based approach** improves computational efficiency with minimal loss in accuracy.  
- **Direct Transformer forecasting** provides slightly higher accuracy at the cost of increased computation.  
- **Diffusion model** effectively reconstructs sparse datasets, supporting robust data assimilation workflows.

---

## 🚀 Future Work
- Extend to **multivariate and spatiotemporal forecasting**.
- Incorporate **advanced diffusion-based assimilation** methods.
- Explore **hybrid EOF-Transformer architectures** for improved performance.
