## Breast Cancer Learning Methods and Active Learning

This repository contains a portfolio-ready project comparing different machine learning paradigms on real-world classification tasks, implemented in Python with `scikit-learn` and presented in a single Jupyter notebook.

The project studies:
- **Supervised learning** with a linear SVM baseline
- **Semi-supervised learning** via self-training
- **Unsupervised learning** with k-means and spectral clustering
- **Active learning with SVMs** on a separate banknote authentication dataset

All core experiments are in `notebooks/learning-methods-comparison.ipynb`.

## Datasets

- **Breast Cancer Wisconsin (Diagnostic)** (`data/wdbc.data`)  
  Binary classification of malignant vs benign tumors with 30 continuous features derived from digitized images of fine needle aspirates.

- **Banknote Authentication** (`data/data_banknote_authentication.txt`)  
  Binary classification of genuine vs forged banknotes using statistical features extracted from images.

Both datasets originate from the UCI Machine Learning Repository. Make sure the files are placed in the `data/` directory with the exact filenames above.

## Repository structure

- `notebooks/learning-methods-comparison.ipynb` – main analysis notebook
- `data/` – raw dataset files (not included here; see Datasets section)
- `README.md` – project description and usage
- `requirements.txt` – Python dependencies

## Installation

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # on macOS/Linux
# .venv\Scripts\activate   # on Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure the `data/` directory exists and contains:

- `wdbc.data`
- `data_banknote_authentication.txt`

with paths that match those used in the notebook (`../data/...` relative to `notebooks/`).

## Running the notebook

From the repository root:

```bash
jupyter notebook notebooks/learning-methods-comparison.ipynb
```

or, if you use JupyterLab:

```bash
jupyter lab notebooks/learning-methods-comparison.ipynb
```

Then run all cells from top to bottom to reproduce the results.

## Methods and experiments

### 1. Breast Cancer: Supervised vs Semi-Supervised vs Unsupervised

- **Supervised**: Linear SVM with L1 penalty and cross-validated C, evaluated under 30 Monte Carlo train/test splits.
- **Semi-supervised (self-training)**: Start from a subset of labeled points, iteratively label the most confident unlabeled point and retrain.
- **Unsupervised (k-means)**: Cluster into two groups, then assign cluster labels using nearby labeled points.
- **Spectral clustering**: RBF-kernel spectral clustering with a heuristic search over gamma to approximate the original class ratio.

Each method is evaluated using:

- Accuracy
- Precision
- Recall
- F1 score
- ROC AUC

The notebook reports **average training and test metrics** across Monte Carlo runs and summarizes them in a comparison table.

### 2. Active Learning with SVMs on Banknote Authentication

On the banknote dataset, we compare:

- **Passive learning**: Randomly sample new points to label.
- **Active learning**: Query points closest to the SVM decision boundary (hyperplane).

Both strategies are run over multiple Monte Carlo trials, growing the labeled training set in stages. The notebook plots **learning curves of average test error vs training set size**, showing that active learning reaches lower error with fewer labeled examples (better label efficiency), especially when the training set is small.

## Results (high level)

- Supervised SVM on fully labeled breast cancer data achieves the strongest overall performance.
- Semi-supervised self-training recovers most of the supervised performance despite starting from fewer labels.
- Unsupervised k-means performs reasonably well but lags behind label-using methods; spectral clustering is more sensitive to hyperparameters and less stable.
- On the banknote dataset, active learning is more sample-efficient than passive learning, achieving similar or better performance with fewer labeled examples.

This project is designed to be a clear, self-contained portfolio piece demonstrating understanding of supervised, semi-supervised, unsupervised, and active learning in practice.

# Supervised, Semi-Supervised, and Unsupervised Learning Analysis

A comprehensive machine learning project comparing different learning paradigms on medical diagnosis and banknote authentication datasets.

## Overview

This project explores and compares four different machine learning approaches:
1. **Supervised Learning** - Using fully labeled data with L1-penalized SVM
2. **Semi-Supervised Learning** - Self-training with partially labeled data
3. **Unsupervised Learning** - K-Means clustering with cluster labeling
4. **Spectral Clustering** - Graph-based clustering approach

Additionally, the project includes an analysis of **Active Learning** strategies, comparing active selection (choosing samples closest to the decision boundary) versus passive random selection.

## Datasets

1. **Breast Cancer Wisconsin (Diagnostic) Dataset**
   - 569 samples with 30 features
   - Binary classification: Malignant (1) vs Benign (0)
   - Features include radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension (each with mean, standard error, and worst values)

2. **Banknote Authentication Dataset**
   - 1,372 samples with 4 features
   - Binary classification: Authentic (0) vs Forged (1)
   - Features: variance, skewness, curtosis, and entropy of wavelet-transformed images

## Methodology

### Part 1: Learning Paradigm Comparison

- **Monte Carlo Simulation**: 30 runs with different random seeds
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, and AUC-ROC
- **Model**: L1-penalized Linear SVM with 5-fold cross-validation for hyperparameter tuning

#### Supervised Learning
- Uses 100% of labeled training data
- Grid search over C parameter (log space from 10^-3 to 10^6)

#### Semi-Supervised Learning
- Starts with 50% labeled data (balanced classes)
- Self-training loop: iteratively adds the sample farthest from the decision hyperplane
- Retrains model after each addition until all unlabeled data is incorporated

#### Unsupervised Learning (K-Means)
- K-Means clustering with 2 clusters (n_init=15)
- Cluster labeling using 30 closest points to each cluster center
- Probability estimation using softmax transformation of distances

#### Spectral Clustering
- RBF kernel with adaptive gamma selection
- Gamma chosen to balance cluster sizes with original class distribution
- KNN classifier used for probability estimation

### Part 2: Active Learning

- **50 Monte Carlo runs** comparing:
  - **Passive Learning**: Random sample selection
  - **Active Learning**: Selecting samples closest to the decision hyperplane
- **Learning Curve**: Test error vs training set size (10 to 900 samples)

## Key Findings

1. **Supervised Learning** achieves the best performance (96.7% test accuracy, 99.3% AUC)
2. **Semi-Supervised Learning** performs nearly as well (96.0% test accuracy, 99.1% AUC), demonstrating effectiveness even with 50% labeled data
3. **Unsupervised Learning (K-Means)** shows lower but reasonable performance (90.6% test accuracy, 97.2% AUC)
4. **Spectral Clustering** has variable performance depending on gamma selection (70.1% test accuracy, 64.4% AUC)
5. **Active Learning** significantly outperforms passive learning, especially with small training sets, achieving lower test error with fewer samples

## Project Structure

```
breast-cancer-learning-methods/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── wdbc.data
│   ├── wdbc.names.csv
│   └── data_banknote_authentication.txt
└── notebooks/
    └── learning-methods-comparison.ipynb
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd breast-cancer-learning-methods
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Open and run the Jupyter notebook:
```bash
jupyter notebook notebooks/learning-methods-comparison.ipynb
```

The notebook is organized into two main sections:
1. Learning paradigm comparison (Supervised, Semi-Supervised, Unsupervised, Spectral)
2. Active learning analysis

## Requirements

- Python 3.7+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

## Results Summary

### Test Set Performance (Average over 30 runs)

| Method | Accuracy | Precision | Recall | F1-Score | AUC |
|--------|----------|-----------|--------|----------|-----|
| Supervised | 0.967 | 0.972 | 0.938 | 0.954 | 0.993 |
| Semi-Supervised | 0.960 | 0.962 | 0.930 | 0.945 | 0.991 |
| Unsupervised (K-Means) | 0.906 | 0.910 | 0.828 | 0.866 | 0.972 |
| Spectral Clustering | 0.701 | 0.367 | 0.187 | 0.247 | 0.644 |

## References

- Breast Cancer Wisconsin (Diagnostic) Data Set: UCI Machine Learning Repository
- Banknote Authentication Data Set: UCI Machine Learning Repository
