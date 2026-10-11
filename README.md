# MUJ-DS-2430010408

| Field | Details |
|-------|---------|
| **Name** | Saket Sharma |
| **Registration Number** | 2430010408 |
| **Branch** | Data Science |
| **Batch** | Batch F |
| **Project Title** | Customer Churn Prediction using Feature Engineering and a Regularized Neural Network |
| **GitHub Username** | Saket-code40 |
| **Training Program** | Machine Learning / Deep Learning Capstone |

---

## Project Overview

An introductory Deep Learning project on the **Telco Customer Churn** dataset
(IBM sample dataset, 7,043 customers, 21 columns) demonstrating:

- Bias / Variance analysis (underfitting, overfitting, good fit)
- 5-Fold Cross-Validation
- Dropout regularization experiment
- L1 / L2 regularization experiments
- Ridge / Lasso (classical linear model comparison)
- Early stopping based on validation loss
- Classification metrics on imbalanced data (Precision, Recall, F1)
- No data leakage (scaler/encoder fit on training data only)

---

## Repository Structure

```
├── README.md                  ← This file
├── INSTALL.md                 ← Installation & setup guide
├── REPORT.md                  ← Full analysis + presentation script + viva Q&A
├── assignments/               ← Weekly assignment submissions
├── notebooks/                 ← Jupyter notebooks
│   ├── churn_prediction.ipynb         (fully executed, all outputs)
│   └── churn_prediction_colab.ipynb   (Colab-ready, auto-downloads dataset)
├── code/                      ← Python source scripts
│   ├── churn_prediction.py            (standalone script, 19 sections)
│   └── build_notebook.py              (notebook generator helper)
├── resources/                 ← Data, results, logs
│   ├── Telco-Customer-Churn.csv       (dataset)
│   ├── results.json                   (all computed metrics)
│   ├── run_output.log                 (full console output)
│   └── churn_prediction.html          (HTML export of executed notebook)
├── figures/                   ← All generated graphs (9 plots)
├── presentations/             ← Presentation slides
│   └── churn_presentation.md
└── capstone/                  ← Capstone-specific deliverables
```

---

## Key Results (seed = 42, fully reproducible)

| Model/Experiment | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression (baseline) | 0.7877 | 0.6167 | 0.5286 | **0.5692** |
| Neural Network (no dropout) | 0.7791 | 0.6146 | 0.4500 | 0.5196 |
| Neural Network + Dropout 0.3 | 0.7896 | 0.6343 | 0.4893 | 0.5524 |
| Neural Network + L1 | 0.7877 | 0.6573 | 0.4179 | 0.5109 |
| Neural Network + L2 | 0.7886 | 0.6338 | 0.4821 | 0.5477 |
| Ridge (linear) | 0.7791 | 0.6026 | 0.4929 | 0.5422 |
| Lasso (linear) | 0.7839 | 0.6262 | 0.4607 | 0.5309 |

**5-Fold CV:** F1 = 0.571 ± 0.021 (stable)
**Early stopping:** halted at epoch 14, best weights from epoch 6 (patience = 8)
**Best regularization:** Dropout

---

## How to Run

### Local
```bash
/usr/bin/python3.12 code/churn_prediction.py
```

### Google Colab
Open `notebooks/churn_prediction_colab.ipynb` in Colab → Runtime → Run All.
The first cell auto-downloads the dataset.

### Jupyter Notebook
```bash
jupyter notebook notebooks/churn_prediction.ipynb
```

See [INSTALL.md](INSTALL.md) for full setup instructions.

---

## Graphs

All 9 figures are in `figures/`:
1. `graph1_loss_curves.png` — Training vs Validation loss with early stopping marker
2. `graph2_accuracy_curves.png` — Training vs Validation accuracy
3. `graph3_cv_f1.png` — 5-Fold CV F1 scores
4. `graph4_confusion_matrix.png` — Labelled TN/FP/FN/TP
5. `graph5_regularization_f1.png` — Regularization comparison
6. `graph6_precision_recall_f1.png` — P/R/F1 across models
7. `graph7_feature_importance.png` — Top logistic regression coefficients
8. `graph8_bias_variance.png` — Oversized model overfitting curves
9. `class_distribution.png` — Target class distribution

---

## Evaluation Criteria Addressed

| Criteria | Evidence |
|----------|----------|
| Project Implementation & Functionality | Working code, real results, 9 graphs |
| Individual Contribution | Multiple commits, issues, PRs |
| Repository Organization | Structured folders, README, INSTALL.md |
| Presentation & Demonstration | `presentations/churn_presentation.md`, REPORT.md |
| Innovation & Problem Solving | Bias/variance demo, comprehensive regularization comparison |

</script>