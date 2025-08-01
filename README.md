# scRNAseq-GNN-multiclass-tp53

## Project Overview
This repository implements a multiclass classification pipeline for single-cell RNA sequencing (scRNAseq) data using Graph Neural Networks (GNNs), with a particular focus on the TP53 gene. The project leverages advanced graph-based machine learning techniques to classify cellular states or mutations, providing a robust framework for biological data analysis.

## Background & Motivation
This project builds directly on my previous work, where I used GNNs to perform binary classification of scRNAseq data by predicting mutation status (i.e., distinguishing between wild-type and mutant samples) ([scRNAseq-GNN-binary-tp53](https://github.com/tommasoravasio/scRNAseq-GNN-binary-tp53)). In this repository, I expand the approach to multiclass classification, predicting not just whether a mutation is present, but also identifying the specific type of mutation (such as missense, nonsense, frameshift, splice site/region, in-frame indels, intronic, or silent mutations). This enables the model to address more nuanced biological questions and demonstrates the scalability of graph-based models for genomics.

## Key Features

**Main Features:**

- **Graph Neural Networks:** Uses GCN and GAT models to learn from gene expression data by representing cells as nodes in a graph.
- **Multiclass Classification:** Predicts not just if TP53 is mutated, but also the specific mutation type (missense, nonsense, etc.).
- **End-to-End Pipeline:** Includes scripts for data preprocessing, graph construction, model training, and evaluation.
- **Flexible Data Handling:** Tools for merging expression data, annotating mutations, and working with different datasets.
- **Experiment Tracking:** Makes it easy to compare different model runs and hyperparameters.
- **Tech Stack:** Built with PyTorch, PyTorch Geometric, Scanpy, scikit-learn, XGBoost, Optuna, pandas, numpy, matplotlib, and seaborn.

*For more details on the methods (like how the gene networks are built, feature selection, batch correction, regularization, and hyperparameter tuning), see the [binary classifier repo](https://github.com/tommasoravasio/scRNAseq-GNN-binary-tp53).*

## Data Sources
- **Single-cell RNA-seq:** 
    - [Single Cell Breast Cancer Cell-line Atlas (Gambardella, 2022)](https://doi.org/10.6084/m9.figshare.15022698.v2)
    - [Pan-Cancer Cell Line scRNA-seq (Kinker et al. 2020)](https://singlecell.broadinstitute.org/single_cell/study/SCP542/pan-cancer-cell-line-heterogeneity#study-download\
\)
- **TP53 Mutation Status:** 
    - [Cancer Cell Line Encyclopedia (Broad, 2019)](https://www.cbioportal.org/study/cnSegments?id=ccle_broad_2019)
    - [Cancer Cell Line Encyclopedia (Novartis/Broad, Nature 2012)](https://www.cbioportal.org/study/summary?id=cellline_ccle_broad)
- **TP53 Target Genes:** [Fischer’s curated list of p53 targets](https://tp53.cancer.gov/target_genes)

**Note:** Data is not included in this repository. Please download the datasets from the above sources and follow the instructions below for preprocessing.


## Getting Started
### 1. Clone the Repository
```bash
git clone https://github.com/tommasoravasio/scRNAseq-GNN-multiclass-tp53.git
cd scRNAseq-GNN-multiclass-tp53
```

### 2. Set Up the Environment
It is recommended to use a virtual environment. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Download and Prepare Data
- Download the scRNA-seq, mutation, and TP53 Target genes data from the sources above.
- Place the raw data in the `data/` directory, following the structure expected by the scripts (see comments in `src/load_data.py` and `src/preprocessing.py`).

### 4. Run the Pipeline

The workflow is **hybrid**:
- **Heavy computations** (such as graph construction, model training, and hyperparameter optimization) are typically launched via scripts in the `jobs/` directory.  
  - On an HPC cluster (e.g., with SLURM), you can submit the provided SLURM job scripts (e.g., `sbatch jobs/graph_constructor.sh`).
  - On a local machine, you can run the corresponding bash scripts directly:
    ```bash
    bash jobs/preprocessing_run.sh
    bash jobs/network_contructor_run.sh
    bash jobs/model_run.sh
    ```
  *(Choose the script and submission method appropriate for your environment. The scripts are designed to be adaptable for both cluster and local execution, modify as needed for your setup.)*

- **Jupyter Notebook** (`notebooks/main_experiment.ipynb`) is used for:
  - Displaying results and visualizations
  - Interactive exploration and analysis
  - Loading and interpreting outputs generated by the scripts
  - Performing lighter computations and preprocessing steps, such as data normalization

**Typical workflow:**
1. Run the appropriate shell script(s) to generate data, graphs, or model results.
2. Use the notebook to visualize, and analyze the outputs.
3. Repeat as needed for different experiments or configurations.

## Results
🚧 **Results Coming Soon!** 🚧

I am working hard to analyze the data and prepare insightful results and visualizations.  
Stay tuned—this section will be updated as the project progresses!



*For questions or collaboration opportunities, feel free to contact me.*