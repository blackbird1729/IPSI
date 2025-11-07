# IPSI: Enhancing Structural Inference with Automatically Learned Structural Priors

This folder contains a custom model implementation designed to work within the **DoSI Benchmark Framework** —  
[Benchmarking Structural Inference Methods for Interacting Dynamical Systems](https://github.com/wang422003/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems).

---

## 📁 Usage Instructions

### 1. Copy the folder into the DoSI repository
Copy this entire folder into the following path within the DoSI benchmark repository: Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems-main\src\models\PreBootSI_NRI_based

---

### 2. Download the required datasets
Download the relevant datasets from the [DoSI project website](https://structinfer.github.io/benchmark/).

Once downloaded, place the datasets in the appropriate data directory within the DoSI project (\src\simulations)
---

### 3. Run the pipeline
After adding your model and downloading the datasets, execute the main **pipeline file** in the root directory of the DoSI project to start the full structural inference process.

Example command:
```bash
python pipeline.py \
    --SI-prior-epochs 100 \
    --SI-joint-epochs 250 \
    --SI-prior-hidden_dim 256 \
    --SI-prior-lr 0.0005 \
    --b-network-type 'gene_regulatory_networks' \           
    --b-simulation-type springs \
    --b-suffix 15r1
