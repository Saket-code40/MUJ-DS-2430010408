# Customer Churn Prediction — Presentation Slides

## Slide 1: Title

**Customer Churn Prediction using Feature Engineering and a Regularized Neural Network**

- Student: Saket Sharma
- Registration: 2430010408
- Batch F — Data Science
- Dataset: Telco Customer Churn (IBM, 7,043 customers)

---

## Slide 2: Problem Statement

- Telecom companies lose ~26.5% of customers annually
- Predicting churn allows targeted retention offers
- Goal: build a simple neural network that identifies at-risk customers
- Educational focus: demonstrate fundamental DL concepts correctly

---

## Slide 3: Dataset Overview

- 7,043 samples, 21 features
- Target: Churn (Yes = 1, No = 0)
- Imbalanced: 73.5% stayed, 26.5% churned
- 11 rows with blank TotalCharges (handled explicitly)
- Features: demographics, services, contract, billing

---

## Slide 4: Why Accuracy Alone Is Misleading

- A model predicting "No churn" for everyone → 73.5% accuracy
- But catches ZERO actual churners!
- Solution: report Precision, Recall, F1-score alongside accuracy

---

## Slide 5: Preprocessing Pipeline

1. Drop `customerID` (non-predictive)
2. Handle blank TotalCharges → drop 11 rows (all tenure=0)
3. One-hot encode 16 categorical columns → 46 features total
4. Scale numeric columns (tenure, MonthlyCharges, TotalCharges) with StandardScaler
5. **Fit scaler/encoder on training data ONLY** (no data leakage)

---

## Slide 6: Train / Validation / Test Split

- 70% Training (4,922 samples) — model learns from this
- 15% Validation (1,055 samples) — early stopping monitors this
- 15% Test (1,055 samples) — final evaluation, never seen during training
- Stratified splitting preserves class ratio

---

## Slide 7: Neural Network Architecture

```
Input (46 features)
  → Dense(64, ReLU)      — learns feature combinations
  → Dropout(0.3)         — prevents co-adaptation
  → Dense(32, ReLU)      — refines patterns
  → Dense(1, Sigmoid)    — outputs churn probability
```

- Loss: Binary Cross-Entropy
- Optimizer: Adam (lr = 0.001)
- Early stopping: patience = 8, restore best weights

---

## Slide 8: Early Stopping Result

- **Training stopped at epoch 14** (best epoch = 6)
- Validation loss stopped improving after epoch 6
- Prevents overfitting by halting before the model memorizes noise

---

## Slide 9: Bias-Variance Demonstration

| Model | Train Acc | Val Acc | Gap | Diagnosis |
|-------|-----------|---------|-----|-----------|
| 1 neuron (underfit) | 0.806 | 0.822 | ~0 | HIGH BIAS |
| Oversized (no reg.) | 0.967 | 0.751 | +0.216 | HIGH VARIANCE |
| Main model (dropout) | 0.812 | 0.818 | -0.006 | GOOD FIT |

---

## Slide 10: 5-Fold Cross-Validation

- F1 scores: 0.571, 0.539, 0.579, 0.569, 0.597
- Mean F1 = 0.571 ± 0.021
- Low standard deviation → model is stable across different data subsets
- Scaler refit inside each fold (no leakage)

---

## Slide 11: Regularization Experiments

| Method | Test F1 | Effect |
|--------|---------|--------|
| No dropout | 0.5196 | Baseline NN |
| Dropout 0.3 | 0.5524 | ✅ Best NN variant |
| L1 (1e-4) | 0.5109 | High precision, low recall |
| L2 (1e-4) | 0.5477 | Nearly matches dropout |

---

## Slide 12: Ridge & Lasso (Linear Models)

- Ridge (L2 penalty on linear model): F1 = 0.5422
- Lasso (L1 penalty on linear model): F1 = 0.5309
- Lasso zeroed 27/46 coefficients → automatic feature selection
- Discarded: gender, PhoneService, several streaming features
- Business insight: contract type and tenure matter most

---

## Slide 13: Final Comparison Table

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Logistic Regression | 0.7877 | 0.6167 | 0.5286 | **0.5692** |
| NN + Dropout | 0.7896 | 0.6343 | 0.4893 | 0.5524 |
| NN + L2 | 0.7886 | 0.6338 | 0.4821 | 0.5477 |

**Key finding:** Simple logistic regression beats the small NN.
This is expected on small tabular data — complexity isn't always better.

---

## Slide 14: Confusion Matrix (Final NN)

|  | Predicted Stayed | Predicted Churn |
|--|-----------------|-----------------|
| **True Stayed** | TN = 696 | FP = 79 |
| **True Churn** | FN = 143 | TP = 137 |

- FP cost: wasted retention offer to loyal customer
- FN cost: lost customer we never tried to save (more expensive!)

---

## Slide 15: Conclusions

1. Dataset is imbalanced → F1 is the right primary metric
2. Early stopping prevented overfitting (stopped epoch 14, best epoch 6)
3. Dropout was the best regularization for this architecture
4. 5-fold CV confirmed model stability (F1 std = 0.021)
5. Logistic regression remains competitive — NN adds complexity without clear gain here
6. Top churn drivers: tenure, contract type, TotalCharges

---

## Slide 16: Limitations & Future Work

- Small dataset (~7K rows) limits NN advantage
- No class-weight tuning (could improve recall)
- No hyperparameter search (learning rate, layers)
- Single train/val/test split for final evaluation
- Future: threshold tuning, SMOTE, feature embeddings

---

## Slide 17: Thank You / Questions?

</script>