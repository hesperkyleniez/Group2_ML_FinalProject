# Comparative Analysis of Machine Learning Models for Cannabis Use Risk Classification

> Logistic Regression · Naive Bayes · K-Nearest Neighbors · Random Forest · Support Vector Machine

A machine learning study that trains, tunes, and compares five classifiers on the **UCI Drug Consumption dataset** to predict whether an individual is a cannabis user or non-user. The project covers the full ML workflow — from preprocessing and hyperparameter tuning to SHAP-based interpretability and cross-model evaluation.

---

## Table of Contents

- [Dataset](#dataset)
- [Problem Framing](#problem-framing)
- [Repository Structure](#repository-structure)
- [Workflow Overview](#workflow-overview)
- [Models & Hyperparameter Tuning](#models--hyperparameter-tuning)
- [Evaluation Strategy](#evaluation-strategy)
- [Outputs](#outputs)
- [Requirements & Setup](#requirements--setup)
- [Running the Project](#running-the-project)
- [Notes](#notes)

---

## Dataset

**Source:** [UCI Drug Consumption (Quantified) — ID 373](https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified)  
**Features:** Demographic and psychological variables (e.g., age, education, NEO-FFI personality scores, impulsivity, sensation-seeking)  
**Original Target:** 7-class cannabis use frequency (Never Used → Last Day)  
**Binarized Target:**
- `0` — Non-User: Never used (`CL0`) or used over a decade ago (`CL1`)
- `1` — User: Used within the last decade through last day (`CL2`–`CL6`)

> **Class Imbalance Note:** The natural class imbalance is deliberately preserved — no oversampling (e.g., SMOTE) was applied. Weighted F1 is used as the primary scoring metric during cross-validation to reduce majority-class bias.

---

## Problem Framing

This is a **binary classification** task. The goal is to identify whether a respondent is a current cannabis user based on their personality profile and demographics. The study also investigates *which features* drive each model's predictions through SHAP interpretability analysis.

---

## Repository Structure

```
├── data/
│   └── drug_consumption.data                    # Raw UCI dataset
├── figures/
│   ├── class_distribution.png
│   ├── combined_confusion_matrix_5models.png
│   ├── combined_confusion_matrix_top2.png
│   ├── combined_regularization_analysis_5models.png
│   ├── combined_roc_curve_single_graph.png
│   ├── combined_shap_bar_5models.png
│   ├── combined_shap_beeswarm_5models.png
│   ├── comprehensive_train_test_analysis.png
│   ├── cross_model_feature_importance.png
│   ├── feature_correlation_matrix.png
│   ├── model_rank_correlation.png
│   └── user_recall_comparison.png
├── models/
│   ├── knn.pkl
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── standard_scaler.pkl
│   └── svm.pkl
├── notebook/
│   └── Comparative_Analysis_of_Machine_Learning_Models_for_Cannabis_Use_Risk_Classification_...ipynb
├── tables/
│   ├── best_hyperparameters.csv
│   ├── cross_model_feature_importance.csv
│   ├── model_performance_summary.csv
│   └── per_class_performance.csv
└── README.md
```

---

## Workflow Overview

The notebook follows a structured, reproducible ML pipeline:

| Step | Description |
|------|-------------|
| 1 | Install and import libraries |
| 2 | Load dataset from UCI repository (`ucimlrepo`) |
| 3 | Prepare binary cannabis target variable |
| 4 | Visualize class distribution (7-class → binary) |
| 5 | Train-test split (80/20, stratified) + StandardScaler |
| 6 | Hyperparameter tuning via `GridSearchCV` with 5-Fold Stratified CV |
| 7 | Final evaluation on held-out test set |
| 8 | Confusion matrix comparison (all 5 models + top-2 spotlight) |
| 9 | Combined ROC curve comparison |
| 10 | Parameter sensitivity analysis (C, kernel, K, n_estimators, var_smoothing) |
| 11–13 | SHAP feature importance — bar plots and beeswarm plots for all 5 models |
| 14 | Per-class performance comparison (User vs. Non-User recall) |
| 15 | Regularization & complexity analysis |

---

## Models & Hyperparameter Tuning

All models were tuned using `GridSearchCV` with **5-Fold Stratified Cross-Validation** and `f1_weighted` as the scoring metric.

| Model | Key Hyperparameters Searched |
|-------|------------------------------|
| Logistic Regression | `C` ∈ {0.01, 0.1, 1, 10}, `solver` ∈ {liblinear, lbfgs} |
| Naive Bayes | `var_smoothing` — log-spaced over 20 values from 1 to 1e-9 |
| K-Nearest Neighbors | `n_neighbors` ∈ {1, 3, 9, 11, 13, 15}, `metric` ∈ {euclidean, manhattan} |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `class_weight` |
| Support Vector Machine | `C` ∈ {0.1, 1, 10}, `gamma` ∈ {scale, 0.01, 0.001}, `kernel` = RBF |

Trained models and the fitted `StandardScaler` are persisted as `.pkl` files via `joblib`.

Additional sensitivity analyses were run post-tuning:
- **SVM:** Linear vs. RBF vs. Polynomial kernel comparison
- **LR:** Solver comparison (liblinear, lbfgs, newton-cg)
- **NB:** `var_smoothing` sweep
- **KNN:** K from 1 to 101 (step 2)
- **RF:** `n_estimators` ∈ {10, 50, 100, 200, 500}

---

## Evaluation Strategy

Each model is evaluated on the **held-out test set** using:

- **Accuracy**
- **Precision, Recall, F1** — reported per class (User / Non-User)
- **Weighted F1** — primary metric given class imbalance
- **AUC-ROC**
- **Generalization Gap** — `Train F1 − Test F1` (overfitting check)
- **Support Vectors** (SVM only)
- **False Negatives** — critical for public health context (missed users)

The **SVM (RBF kernel)** and **Logistic Regression** are spotlighted as the best non-linear and best linear models respectively.

---

## Outputs

All outputs are auto-saved to a timestamped folder (`cannabis_ml_results_<YYYYMMDD_HHMMSS>/`) and also archived as a `.zip`.

### Figures (`figures/`)

| File | Description |
|------|-------------|
| `class_distribution.png` | Original 7-class vs. binarized target distribution |
| `feature_correlation_matrix.png` | Heatmap of feature correlations |
| `comprehensive_train_test_analysis.png` | Train vs. Test F1 bar chart (generalization gap) |
| `combined_confusion_matrix_5models.png` | Confusion matrices for all 5 models |
| `combined_confusion_matrix_top2.png` | SVM vs. LR confusion matrix spotlight |
| `combined_roc_curve_single_graph.png` | Overlaid ROC curves for all 5 models |
| `combined_shap_bar_5models.png` | SHAP mean absolute feature importance (bar) — all models |
| `combined_shap_beeswarm_5models.png` | SHAP beeswarm plots — all models |
| `user_recall_comparison.png` | Cannabis User class recall across models |
| `combined_regularization_analysis_5models.png` | Parameter sensitivity curves (C, var_smoothing, K, n_estimators) |
| `cross_model_feature_importance.png` | Cross-model comparison of top SHAP features across all 5 models |
| `model_rank_correlation.png` | Rank correlation of feature importance rankings across models |

### Tables (`tables/`)

| File | Description |
|------|-------------|
| `best_hyperparameters.csv` | Best params from GridSearchCV per model |
| `cross_model_feature_importance.csv` | Aggregated SHAP feature importance values across all 5 models |
| `model_performance_summary.csv` | Accuracy, Precision, Recall, AUC-ROC, F1 Gap, Support Vectors |
| `per_class_performance.csv` | Per-class precision, recall, F1, and false negatives |

### Models (`models/`)

| File | Description |
|------|-------------|
| `standard_scaler.pkl` | Fitted StandardScaler (required for inference) |
| `logistic_regression.pkl` | Best LR model |
| `naive_bayes.pkl` | Best NB model |
| `knn.pkl` | Best KNN model |
| `random_forest.pkl` | Best RF model |
| `svm.pkl` | Best SVM model |

---

## Requirements & Setup

**Python 3.10+** is recommended. Install all dependencies with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap joblib ucimlrepo
```

| Library | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | Modeling, preprocessing, evaluation |
| `matplotlib`, `seaborn` | Visualization |
| `shap` | Model interpretability |
| `joblib` | Model serialization |
| `ucimlrepo` | UCI dataset loader |

---

## Running the Project

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn shap joblib ucimlrepo
   ```

3. Open the notebook:
   ```bash
   jupyter notebook notebook/Comparative_Analysis_of_Machine_Learning_Models_for_Cannabis_Use_Risk_Classification_Logistic_Regression__Naive_Bayes__K_Nearest_Neighbors__Random_Forest__and_Support_Vector_Machine.ipynb
   ```

4. Run all cells sequentially from top to bottom.

> All outputs (figures, tables, models, and a summary report) will be saved automatically to a new timestamped folder in your working directory.

---

## Notes

- **GitHub notebook rendering:** GitHub may fail to render the notebook preview due to Jupyter widget metadata or large output cells. If the preview doesn't load, download the `.ipynb` and open it locally in Jupyter Notebook, VS Code, or upload it to Google Colab.

- **SHAP compute time:** SHAP values for SVM and KNN use `KernelExplainer`, which is model-agnostic but slow. These steps may take several minutes depending on your hardware. The explainer uses a background sample of 100 training points and evaluates on 50 test samples.

- **Reproducibility:** `random_state=42` is set across all stochastic components. The dataset is fetched live from the UCI repository via `ucimlrepo` — an internet connection is required on first run.

- **Feature scaling:** `StandardScaler` is fit on the training set only and applied to both train and test sets. The fitted scaler is saved alongside the models — it must be used when loading any saved model for inference.
