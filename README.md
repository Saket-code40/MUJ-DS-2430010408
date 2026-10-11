# Customer Churn Prediction using Feature Engineering and a Regularized Neural Network

An introductory Deep Learning project on the **Telco Customer Churn** dataset
(IBM sample dataset, 7,043 customers, 21 columns).

Goal: demonstrate the fundamental concepts correctly — bias/variance,
cross-validation, regularization (dropout, L1, L2), Ridge/Lasso, early
stopping, loss curves and classification metrics on imbalanced data —
without making the model unnecessarily complex.

## Contents

| File | Description |
|---|---|
| `churn_prediction.ipynb` | Fully **executed** Jupyter notebook (all outputs + 9 graphs inline) |
| `churn_prediction_colab.ipynb` | Clean Colab-ready copy (outputs stripped, auto-downloads the dataset) |
| `churn_prediction.py` | The same code as a standalone script (19 commented sections) |
| `build_notebook.py` | Helper that generates the notebook from the script |
| `REPORT.md` | Methodology, full analysis, teacher-ready presentation script, viva Q&A |
| `results.json` | Every computed metric, machine-readable |
| `figures/` | All graphs: loss/accuracy curves, CV folds, confusion matrix, regularization comparison, bias/variance, feature importance |
| `Telco-Customer-Churn.csv` | Dataset (IBM sample) |

## How to run

**Google Colab:** upload `churn_prediction_colab.ipynb` → *Runtime → Run all*.
The first cell downloads the dataset automatically; all libraries are pre-installed.

**Local:**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
python churn_prediction.py
```

Everything is seeded (`SEED = 42`), so re-running reproduces the exact same numbers.

## Headline results (held-out test set, 15%)

| Model / Experiment | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression (baseline) | 0.7877 | 0.6167 | 0.5286 | 0.5692 |
| Neural Network (no dropout) | 0.7791 | 0.6146 | 0.4500 | 0.5196 |
| Neural Network + Dropout 0.3 | 0.7896 | 0.6343 | 0.4893 | 0.5524 |
| Neural Network + L1 | 0.7877 | 0.6573 | 0.4179 | 0.5109 |
| Neural Network + L2 | 0.7886 | 0.6338 | 0.4821 | 0.5477 |
| Ridge (linear) | 0.7791 | 0.6026 | 0.4929 | 0.5422 |
| Lasso (linear) | 0.7839 | 0.6262 | 0.4607 | 0.5309 |

- **Imbalanced data:** 26.54% churned → accuracy alone is misleading (an
  "always stay" dummy model scores ~73%).
- **Early stopping:** training stopped at epoch 14 because validation loss
  stopped improving (best weights = epoch 6, patience = 8).
- **5-fold cross-validation** (NN + dropout): F1 = 0.571 ± 0.021 → stable model.
- **Bias/variance demo:** a 1-neuron net underfits; an oversized unregularized
  net memorizes (train acc 0.967 vs val acc 0.751); the regularized model sits
  in between with a ~0 train/val gap.
- **No data leakage:** scaler and one-hot encoder are fit on training data only;
  the test set is touched exactly once, at the very end.

See `REPORT.md` for the complete analysis, a short presentation script for the
class, and 17 likely viva questions with concise answers.
