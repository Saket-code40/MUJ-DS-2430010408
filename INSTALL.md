# Installation & Setup Guide

## Prerequisites

- Python 3.10+ (tested with 3.12)
- pip package manager

## Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

Or install all at once:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
```

## Running the Project

### Option 1: Standalone Script

```bash
cd customer_chrum
python code/churn_prediction.py
```

Outputs:
- Console: all metrics, comparisons, conclusions
- `figures/`: 9 PNG graphs
- `results.json`: machine-readable results

### Option 2: Jupyter Notebook

```bash
jupyter notebook notebooks/churn_prediction.ipynb
```

All cells are pre-executed. Use **Kernel → Restart & Run All** to re-run from scratch.

### Option 3: Google Colab

1. Open [colab.research.google.com](https://colab.research.google.com)
2. **File → Upload notebook → Upload** → select `notebooks/churn_prediction_colab.ipynb`
3. **Runtime → Run All**

No installation needed — all libraries are pre-installed in Colab.
The first cell auto-downloads the dataset from IBM's GitHub repository.

## Reproducibility

All experiments use `SEED = 42` (Python `random`, NumPy, TensorFlow).
Re-running produces identical results.

## Dataset

The Telco Customer Churn dataset (`Telco-Customer-Churn.csv`) is included in `resources/`.
It can also be downloaded from:
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

## Troubleshooting

| Issue | Solution |
|-------|----------|
| TensorFlow import error | Ensure Python ≤ 3.12; TF doesn't support 3.13+ yet |
| Matplotlib backend error | The script uses `Agg` backend (headless-safe) |
| Colab "Unable to read file" | Use `churn_prediction_colab.ipynb` (outputs stripped, smaller file) |

</script>