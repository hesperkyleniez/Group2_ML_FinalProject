# Cannabis Use Risk Classification Using a Stacking Ensemble of Support Vector Machine, Random Forest, and Logistic Regression

A machine learning study that trains, tunes, and evaluates three baseline classifiers and a proposed stacking ensemble on the **UCI Drug Consumption dataset** to predict whether an individual is a cannabis user or non-user. The project covers the full ML workflow from preprocessing and hyperparameter tuning to SHAP-based interpretability and cross-model evaluation.

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
**Features:** Demographic and psychological variables (age, education, NEO-FFI personality scores, impulsivity, sensation-seeking)  
**Original Target:** 7-class cannabis use frequency (Never Used → Last Day)  
**Binarized Target:**
- `0` — Non-User: Never used (`CL0`) or used over a decade ago (`CL1`)
- `1` — User: Used within the last decade through last day (`CL2`–`CL6`)

> **Class Imbalance Note:** The natural class imbalance (~67% users, ~33% non-users) is deliberately preserved, no oversampling (e.g., SMOTE) was applied. Weighted F1 is used as the primary scoring metric during cross-validation to reduce majority-class bias.

---

## Problem Framing

This is a **binary classification** task. The study investigates how three baseline classifiers representing distinct learning paradigms compare against a proposed stacking ensemble that combines their complementary strengths. SHAP interpretability analysis is applied to identify which personality and demographic features drive each model's predictions.

---

## Repository Structure

```
├── data/
│   └── drug_consumption.data
├── figures/
│   ├── accuracy_precision_recall_compari...
│   ├── class_distribution.png
│   ├── confusion_matrices_all_models.png
│   ├── cross_model_feature_importance.png
│   ├── feature_correlation_matrix.png
│   ├── final_model_comparison.png
│   ├── regularization_complexity_analysi...
│   ├── roc_curve_1_baselines.png
│   ├── roc_curve_2_stacking.png
│   ├── shap_bar_4_models_including_stac...
│   ├── shap_beeswarm_4_models.png
│   └── train_vs_test_f1_comparison.png
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── standard_scaler.pkl
│   ├── standard_stacking_model.pkl
│   ├── standard_stacking_threshold.pkl
│   └── svm.pkl
├── notebook/
│   └── 3CSD_Group2_Implementation.ipynb
├── tables/
│   ├── best_hyperparameters.csv
│   ├── cross_model_feature_importance.csv
│   ├── final_model_comparison.csv
│   ├── model_performance_summary.csv
│   └── per_class_performance_all_models.csv
└── README.md
```

---

## Workflow Overview

| Step | Description |
|------|-------------|
| 1 | Install and import libraries |
| 2 | Load dataset from UCI repository (`ucimlrepo`) |
| 3 | Prepare binary cannabis target variable |
| 4 | Visualize class distribution (7-class - binary) |
| 5 | Train-test split (80/20, stratified) + StandardScaler + feature correlation analysis |
| 6 | Hyperparameter tuning via `GridSearchCV` with 5-Fold Stratified CV |
| 7 | Final evaluation of baseline models on held-out test set |
| 8 | Proposed Stacking Ensemble training with optimal decision threshold |
| 9 | Train vs. Test F1 comparison (generalization gap analysis) |
| 11–14 | SHAP analysis — cross-model feature importance, bar plots, and beeswarm plots |
| 15 | Confusion matrix analysis (baselines + stacking ensemble) |
| 16 | ROC curve analysis (baselines + stacking ensemble) |
| 17 | Per-class performance comparison (User vs. Non-User recall) |
| 18 | Regularization and complexity analysis (C, n_estimators, meta-learner trees) |

---

## Models & Hyperparameter Tuning

All models were tuned using `GridSearchCV` with **5-Fold Stratified Cross-Validation** and `f1_weighted` as the scoring metric.

| Model | Role | Key Hyperparameters Searched |
|-------|------|------------------------------|
| Logistic Regression | Baseline — linear | `C` ∈ {0.01, 0.1, 1, 10}, `solver` ∈ {liblinear, lbfgs} |
| Random Forest | Baseline — ensemble | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `class_weight` |
| Support Vector Machine | Baseline — boundary-based | `C` ∈ {0.1, 1, 10}, `gamma` ∈ {scale, 0.01, 0.001}, `kernel` = RBF |
| Stacking Ensemble | Proposed model | RF + SVM + LR as base learners, LR as meta-learner, decision threshold optimized via OOF validation |

Trained models and the fitted `StandardScaler` are persisted as `.pkl` files via `joblib`.

---

## Evaluation Strategy

Each model is evaluated on the **held-out test set** using:

- **Accuracy**
- **Precision, Recall, F1** — reported per class (User / Non-User)
- **Weighted F1** — primary metric given class imbalance
- **AUC-ROC**
- **Generalization Gap** — `Train F1 − Test F1` (overfitting check)
- **False Negatives** — critical for public health screening context

---

## Outputs

All outputs are saved to `figures/`, `tables/`, and `models/` directories.

### Figures

| File | Description |
|------|-------------|
| `class_distribution.png` | Original 7-class vs. binarized target distribution |
| `feature_correlation_matrix.png` | Pearson correlation heatmap of psychological features |
| `train_vs_test_f1_comparison.png` | Train vs. Test F1 bar chart across all models |
| `confusion_matrices_all_models.png` | Confusion matrices for all models |
| `roc_curve_1_baselines.png` | ROC curves for the three baseline models |
| `roc_curve_2_stacking.png` | ROC curve for the proposed stacking ensemble |
| `shap_bar_4_models_including_stac...` | SHAP mean absolute feature importance — all models |
| `shap_beeswarm_4_models.png` | SHAP beeswarm plots — all models |
| `cross_model_feature_importance.png` | Cross-model SHAP feature importance comparison |
| `accuracy_precision_recall_compari...` | Per-class User recall comparison across models |
| `regularization_complexity_analysi...` | C sensitivity, n_estimators, and meta-learner complexity curves |
| `final_model_comparison.png` | Baseline models vs. stacking ensemble F1 comparison |

### Tables

| File | Description |
|------|-------------|
| `best_hyperparameters.csv` | Best params from GridSearchCV per model |
| `model_performance_summary.csv` | Accuracy, Precision, Recall, AUC-ROC, F1 Gap per model |
| `per_class_performance_all_models.csv` | Per-class precision, recall, F1, and false negatives |
| `cross_model_feature_importance.csv` | Normalized SHAP importance values across all models |
| `final_model_comparison.csv` | Baselines vs. stacking ensemble full comparison table |

### Models

| File | Description |
|------|-------------|
| `standard_scaler.pkl` | Fitted StandardScaler — required for inference |
| `logistic_regression.pkl` | Best tuned LR model |
| `random_forest.pkl` | Best tuned RF model |
| `svm.pkl` | Best tuned SVM model |
| `standard_stacking_model.pkl` | Trained stacking ensemble |
| `standard_stacking_threshold.pkl` | Optimal decision threshold for the stacking ensemble |

---

## Requirements & Setup

**Python 3.10+** is recommended.

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

3. Open and run the notebook sequentially:
   ```bash
   jupyter notebook notebook/3CSD_Group2_Implementation.ipynb
   ```

> All outputs are saved automatically to `figures/`, `tables/`, and `models/` directories in your working directory.

---

## Notes

- **GitHub notebook rendering:** GitHub may fail to render the notebook preview due to large output cells or widget metadata. If the preview does not load, download the `.ipynb` and open it locally in Jupyter Notebook, VS Code, or upload it to Google Colab.

- **SHAP compute time:** SHAP values for SVM and the stacking ensemble use `KernelExplainer`, which is model-agnostic but computationally slow. These steps may take several minutes. The explainer uses a background sample of 100 training points and evaluates on 50 test samples.

- **Reproducibility:** `random_state=42` is set across all stochastic components. The dataset is fetched live from the UCI repository via `ucimlrepo`, an internet connection is required on first run.

- **Feature scaling:** `StandardScaler` is fit on the training set only and applied to both train and test sets. The fitted scaler must be used alongside any saved model for inference.

- **Decision threshold:** The stacking ensemble uses an optimized decision threshold (saved as `standard_stacking_threshold.pkl`) derived from out-of-fold validation rather than the default 0.5, to improve F1 performance on the imbalanced dataset.
