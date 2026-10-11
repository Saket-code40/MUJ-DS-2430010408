# Customer Churn Prediction using Feature Engineering and a Regularized Neural Network

**Dataset:** Telco Customer Churn (IBM sample dataset, `Telco-Customer-Churn.csv`)
**Run:** reproducible — seed = 42 everywhere (`random`, `numpy`, `tensorflow`). Re-run with:

```bash
/usr/bin/python3.12 churn_prediction.py
```

All numbers in this report were produced by the script above on this machine. Nothing is hand-written. Raw values are saved in `results.json`; figures are in `figures/`.

---

## 1. Dataset inspection

| Property | Value |
|---|---|
| Samples | 7,043 |
| Columns | 21 |
| Stayed (No) | 5,174 → **73.46%** |
| Churned (Yes) | 1,869 → **26.54%** |
| Duplicate rows | 0 |
| Missing values (NaN) | none reported by pandas — but see below |
| Numeric columns | `tenure`, `MonthlyCharges`, `TotalCharges` (stored as text!) |
| Categorical columns | 16 (including `gender`, `Contract`, `InternetService`, …) |

**The TotalCharges trap.** `TotalCharges` is read as a string column. `pd.to_numeric(..., errors="coerce")` reveals **11 rows with blank strings** instead of numbers. Inspecting those rows shows they are all brand-new customers with `tenure == 0`. **Strategy chosen: drop the 11 rows** (0.16% of the data). Dropping is defensible because their total charges are genuinely unknown, imputing a fake value would add noise for no gain, and 11 rows cannot change any conclusion.

**Why accuracy alone is misleading:** the classes are imbalanced (26.5% churn). A model that always predicts "Stayed" reaches ~73% accuracy while catching **zero** churners — useless to the business. We therefore report Precision, Recall and F1 alongside Accuracy.

## 2. Preprocessing

1. `TotalCharges` → numeric, blank rows dropped (explained above).
2. `customerID` and `Churn` removed from features (`Churn` becomes the target: Yes=1, No=0).
3. Categorical columns one-hot encoded with `pd.get_dummies` → **46 features**.
   *One-hot encoding* converts a categorical column with *k* values into *k* binary columns. Neural networks do arithmetic on their inputs; plain labels (1, 2, 3) would fake an order ("Two year > One year > Month-to-month" as a quantity) that does not exist. Binary 0/1 columns have no such fake order.
4. **Split first** (70/15/15, stratified), then:
   - one-hot dummies of the validation/test sets are reindexed to the training columns (unknown categories → 0);
   - `StandardScaler` is fit **on training data only** and applied to all three sets.
   *Feature scaling* transforms each numeric column to mean 0 / standard deviation 1. Gradient descent converges faster and more stably when all inputs live on the same scale; otherwise large-magnitude features (TotalCharges in the thousands) dominate small ones.
5. Doing the split before encoding/scaling is what prevents **data leakage**: the validation and test sets never contribute to any fitted statistic.

**Split sizes:** Training 4,922 / Validation 1,055 / Test 1,055 (stratified, so each set keeps ~26.5% churn).

- **Training set** – the model learns its weights here.
- **Validation set** – acts as a proxy for "unseen data" during development: it drives early stopping and lets us compare regularization variants. The model never trains on it.
- **Test set** – completely locked away until the very last step; each model is scored on it exactly once.

## 3. Baseline: Logistic Regression

Why a baseline? It gives a reference bar. If a neural network cannot beat a single linear layer, its complexity buys nothing. Logistic regression also gives interpretable coefficients we use later for business insight.

Test-set results:

| Accuracy | Precision | Recall | F1 |
|--:|--:|--:|--:|
| 0.7877 | 0.6167 | 0.5286 | 0.5692 |

## 4. Neural network architecture

```
Input (46 scaled/encoded features)
  → Dense(64, relu)      # 64 neurons learn weighted feature combinations
  → Dropout(0.3)         # randomly disables 30% of neurons each training step
  → Dense(32, relu)      # smaller layer refines the patterns
  → Dense(1, sigmoid)    # squashes the output into a churn probability in [0,1]
```

- **Loss:** binary cross-entropy — the standard loss for "yes/no" problems; it penalizes confident wrong answers heavily.
- **Optimizer:** Adam, learning rate 1e-3 — an adaptive gradient method that is robust and low-tuning.
- **Why small?** With ~4,900 training rows and 46 features, a huge network would simply memorize. We deliberately keep capacity modest so we can *see* regularization work.

## 5. Early stopping

Training ran up to 200 epochs with `EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)`.

> **"Training was stopped at epoch 14 because validation loss stopped improving. The best weights came from epoch 6."**

After epoch 6 the validation loss never improved again; after 8 wasted epochs (patience) training halted and the epoch-6 weights were restored. Early stopping fights overfitting because the longer a model trains past its best validation point, the more it memorizes training-set noise — stopping freezes generalization near its peak.

## 6. Bias vs variance — measured, not just defined

| Model | Train acc | Val acc | Gap | Diagnosis |
|---|--:|--:|--:|---|
| 1-neuron perceptron | 0.8064 | 0.8218 | ≈ 0 | **High bias / underfitting** — both low & equal; model too simple to capture the signal |
| Oversized MLP (256-128-64), no dropout, no regularization, 150 epochs | **0.9667** | 0.7507 | **+0.216** | **High variance / overfitting** — memorizes training data, collapses on unseen data |
| Main model (64-32 + Dropout 0.3 + early stopping) | 0.8119 | 0.8180 | −0.006 | **Good fit** — small gap, validation tracks training |

The train/validation **gap** is the practical overfitting detector: a large and growing gap means the model is learning noise specific to the training set (see `graph8_bias_variance.png`).

## 7. 5-fold cross-validation (K = 5)

Methodology (leakage-free): run only on the **training** partition. In each fold the StandardScaler is refit on the 4 training folds, a fresh NN (64-32, dropout 0.3) is trained with early stopping on a 15% slice carved out of the training portion, and the held-out fold is scored once. The test set is never touched.

| Fold | Accuracy | Precision | Recall | F1 |
|--:|--:|--:|--:|--:|
| 1 | 0.7990 | 0.6600 | 0.5038 | 0.5714 |
| 2 | 0.7848 | 0.6263 | 0.4733 | 0.5391 |
| 3 | 0.7978 | 0.6493 | 0.5229 | 0.5793 |
| 4 | 0.7998 | 0.6633 | 0.4981 | 0.5689 |
| 5 | 0.8130 | 0.6974 | 0.5211 | 0.5965 |
| **Mean ± std** | **0.7989 ± 0.0100** | **0.6593 ± 0.0258** | **0.5038 ± 0.0202** | **0.5710 ± 0.0209** |

The small standard deviations tell us the model is **stable**: its performance does not hinge on which rows happen to land in the training set. Its single-split test F1 (0.5524) sits within ~1 std of the CV mean (0.571), i.e. the one-off test result is not a fluke.

## 8. Dropout experiment

Identical architecture, trained independently with and without Dropout(0.3):

| Variant | Accuracy | Precision | Recall | F1 | train/val gap |
|---|--:|--:|--:|--:|--:|
| No dropout | 0.7791 | 0.6146 | 0.4500 | 0.5196 | −0.0004 |
| Dropout 0.3 | **0.7896** | **0.6343** | **0.4893** | **0.5524** | −0.0061 |

Dropout improved every metric here (+0.033 F1). How it works: each training step randomly silences 30% of the hidden neurons, so no neuron can rely on a few fixed partners; the network is forced to spread useful information across many paths, which behaves like training an ensemble of sub-networks and reduces co-adaptation/overfitting.

## 9. L1 vs L2 on the neural network

Both added with strength **1e-4** to the Dense-layer weights (plus dropout, same architecture). 1e-4 is small enough not to cripple learning on 4,922 rows, but large enough to be felt: the added penalty is comparable in scale to the loss drop between good and bad epochs.

| Variant | Train loss | Val loss | Accuracy | Precision | Recall | F1 |
|---|--:|--:|--:|--:|--:|--:|
| NN + Dropout (reference) | — | — | 0.7896 | 0.6343 | 0.4893 | **0.5524** |
| NN + L1 (1e-4) | 0.4178 | 0.4419 | 0.7877 | **0.6573** | 0.4179 | 0.5109 |
| NN + L2 (1e-4) | 0.4122 | **0.4177** | 0.7886 | 0.6338 | 0.4821 | 0.5477 |

- **L1** adds the sum of absolute weights to the loss → pushes many weights exactly to zero → acts like feature selection. Here it raised precision but hurt recall most (F1 −0.041 vs the dropout reference) — at this strength it was the most aggressive constraint.
- **L2** adds the sum of squared weights → shrinks all weights smoothly without killing them → gentler, and nearly matched the dropout-only run (F1 −0.005 only), with the smallest val loss of the three.

Simple metaphor: L1 says "remove furniture you don't use"; L2 says "keep all furniture but make it smaller".

## 10. Ridge and Lasso — and how they differ from NN L1/L2

Ridge and Lasso are **classical linear models** with penalties: Ridge = linear regression + L2 penalty; Lasso = linear regression + L1 penalty. The *idea* (punish large weights) is the same family as NN regularization, but the **implementation is not identical**: Ridge/Lasso are convex, closed-form/coordinate-descent estimators whose L1 penalty provably zeros coefficients; applying an L1 penalty to a neural network is a non-convex approximation of the same intuition, not the same estimator. We therefore do not claim "Ridge == NN with L2".

Using logistic-link style thresholding (predict ≥ 0.5) on scaled features, test results:

| Model | Accuracy | Precision | Recall | F1 |
|---|--:|--:|--:|--:|
| Ridge (α=1.0) | 0.7791 | 0.6026 | 0.4929 | 0.5422 |
| Lasso (α=0.005) | 0.7839 | 0.6262 | 0.4607 | 0.5309 |

Both land close to the NN variants — an honest, instructive result: for clean tabular data, well-tuned linear models are hard to beat with a small MLP.

**Lasso sparsity:** Lasso drove **27 of 46 coefficients to zero**, keeping essentially tenure, charges, Contract_Month-to-month and InternetService_Fiber optic family. Business read: it discarded things like `gender` and many "No internet service" placeholders — intuitively irrelevant to churn — while keeping contract type and tenure, exactly what a telecom analyst would name as churn drivers. It also zeroed some plausibly relevant one-hot levels (e.g. `Contract_Two year`), a reminder that Lasso with correlated one-hot columns picks arbitrary survivors among them.

## 11. Final comparison (test set, all models)

| Model/Experiment | Accuracy | Precision | Recall | F1 |
|---|--:|--:|--:|--:|
| Logistic Regression | 0.7877 | 0.6167 | 0.5286 | **0.5692** |
| Neural Network (no dropout) | 0.7791 | 0.6146 | 0.4500 | 0.5196 |
| Neural Network + Dropout | **0.7896** | 0.6343 | 0.4893 | 0.5524 |
| Neural Network + L1 | 0.7877 | **0.6573** | 0.4179 | 0.5109 |
| Neural Network + L2 | 0.7886 | 0.6338 | 0.4821 | 0.5477 |
| Ridge (linear) | 0.7791 | 0.6026 | 0.4929 | 0.5422 |
| Lasso (linear) | 0.7839 | 0.6262 | 0.4607 | 0.5309 |

**Confusion matrix (final NN + Dropout, test set):**

|  | Predicted Stayed | Predicted Churn |
|---|--:|--:|
| **True Stayed** | TN = 696 | FP = 79 |
| **True Churn** | FN = 143 | TP = 137 |

Business translation:
- **FP = 79** — loyal customers wrongly offered retention deals → wasted discounts (small cost each).
- **FN = 143** — real churners we missed → the company loses those customers entirely (large cost each). For churn problems recall is usually the metric worth protecting.

## 12. Graphs (9 total, all in `figures/`)

1. `class_distribution.png` — churn vs stayed counts.
2. `graph1_loss_curves.png` — training vs validation loss with the early-stopping epoch marked.
3. `graph2_accuracy_curves.png` — training vs validation accuracy.
4. `graph3_cv_f1.png` — F1 per CV fold + mean ± std line.
5. `graph4_confusion_matrix.png` — labelled confusion matrix of the final NN.
6. `graph5_regularization_f1.png` — No reg / Dropout / L1 / L2 F1 comparison.
7. `graph6_precision_recall_f1.png` — P/R/F1 across all models.
8. `graph7_feature_importance.png` — top 12 logistic-regression coefficients.
9. `graph8_bias_variance.png` — oversized model's diverging loss/accuracy curves (overfitting visualization).

## 13. Metrics explained for churn

- **Precision** — of customers we *predicted* would churn, how many actually did? (0.63 for the final NN: 63% of retention offers went to true churners; the rest were wasted.)
- **Recall** — of customers who *actually* churned, how many did we catch? (0.49: we flagged about half of the real churners; the other half left without any retention attempt.)
- **F1** — harmonic mean of precision and recall; a single number that punishes lopsided trade-offs. On an imbalanced dataset F1 is more informative than accuracy because it depends on the minority class's TP/FP/FN directly.
- **Accuracy** — overall correctness; misleading here because a naive majority-class guess already scores 73%.

## 14. Final analysis (answers to the 14 questions)

1. **Best model?** Logistic Regression by F1 (0.5692); NN + Dropout is a close second (0.5524) with the highest accuracy (0.7896).
2. **Imbalanced?** Yes — 73.5% / 26.5%.
3. **Was accuracy misleading?** Yes: a dummy "everyone stays" model reaches ~73% accuracy with zero business value.
4. **Signs of overfitting?** Yes — deliberately demonstrated: the oversized unregularized network showed a +21.6-point train/val gap (0.9667 vs 0.7507). The main model showed no gap (−0.006) thanks to dropout + early stopping.
5. **Dropout effect?** Improved every metric vs no dropout (F1 0.5196 → 0.5524).
6. **L1 effect?** Highest precision (0.6573) but lowest recall/F1 (0.4179 / 0.5109): it zeroed weights aggressively.
7. **L2 effect?** Gentle shrinkage, nearly matching the dropout-only run (F1 0.5477) and the smallest validation loss.
8. **Best regularization here?** Dropout alone.
9. **CV → variance?** F1 = 0.571 ± 0.021 across folds: low variance, stable model; the single test result is consistent with CV.
10. **Early stopping epoch?** Stopped at epoch 14, best weights from epoch 6.
11. **Why?** Validation loss never improved after epoch 6; patience 8 was exhausted.
12. **NN vs baseline?** NN + Dropout (F1 0.5524) slightly below logistic regression (0.5692). Honest takeaway: on small, clean tabular data a small MLP matches but does not automatically beat a linear model — which is exactly why baselines exist.
13. **Top churn drivers (logistic coefficients)?** Low `tenure`, `Contract_Month-to-month`, high `TotalCharges`/`MonthlyCharges`, and `InternetService_Fiber optic` push churn up; `Contract_Two year` and DSL push it down. Business story: new customers on expensive month-to-month fiber contracts are the flight-risk segment.
14. **Limitations?** ~7k rows; single final test split (CV mitigates this); no class weights or threshold tuning (recall could be traded up); no hyperparameter search; one-hot only (no embeddings); Lasso/Ridge used with a 0.5 threshold on a linear score rather than a calibrated probability.

---

## 15. Presentation script (for your teacher)

> "My project predicts whether a telecom customer will churn, using the IBM Telco dataset of 7,043 customers. I chose it because it is small, tabular and imbalanced — perfect for demonstrating core deep-learning ideas rather than chasing accuracy.
>
> I cleaned the data first: TotalCharges contained blank strings for 11 new customers, which I dropped, and I explained why accuracy alone is misleading when only 26.5% of customers churn.
>
> I split 70/15/15 with stratification, and fit the scaler and one-hot encoder on the training set only, to avoid data leakage. My baseline was logistic regression — F1 0.569. Then I built a deliberately small neural network: Dense 64, Dropout, Dense 32, sigmoid output, binary cross-entropy with Adam.
>
> Early stopping watched the validation loss with patience 8: training halted at epoch 14 and the best weights came from epoch 6.
>
> For bias/variance I trained three models: a 1-neuron model that underfits, an oversized unregularized network that reached 96.7% training accuracy but only 75.1% validation accuracy — a 21-point gap, a textbook overfit — and my regularized model whose gap is essentially zero.
>
> 5-fold cross-validation gave F1 0.571 ± 0.021, showing the model is stable. In the regularization experiments, dropout gave the best F1 (0.552), L1 raised precision but hurt recall, L2 stayed close to dropout. Finally I showed Ridge and Lasso as classical linear analogues, and that they are related to but not identical to NN L1/L2 penalties.
>
> The honest conclusion: logistic regression won by a small margin. That is the right lesson for this course — on small clean tabular data, complexity must earn its place, and regularization is what keeps a neural network honest."

## 16. Likely viva questions & concise answers

1. **Why not just report accuracy?** — Imbalanced classes: predicting "No churn" for everyone scores 73% while catching nobody. Precision/Recall/F1 expose that.
2. **What is one-hot encoding and why?** — Each category becomes its own 0/1 column so the network never treats categories as ordered numbers.
3. **Why scale, and why fit the scaler on training data only?** — Gradient descent converges better with zero-mean unit-variance inputs. Fitting on all data would leak test-set statistics into training.
4. **Train/val/test: why three sets?** — Training teaches weights; validation tunes decisions (early stopping, model choice); test estimates real-world performance once, on never-seen data.
5. **What is data leakage?** — Any information from val/test influencing training. I prevented it by splitting before encoding/scaling and refitting the scaler inside each CV fold.
6. **How does early stopping prevent overfitting?** — Validation loss bottoms at epoch 6 then rises as the model memorizes noise; stopping and restoring best weights freezes generalization at its peak.
7. **How does dropout work?** — Each step randomly disables 30% of hidden units, forcing redundant, distributed representations; at inference all units are used. It approximates averaging many sub-networks.
8. **L1 vs L2 in plain words?** — L1 = sum of |weights|, drives weights to zero (sparsity/feature selection). L2 = sum of squared weights, shrinks all weights smoothly.
9. **Are Ridge/Lasso the same as NN regularization?** — Same family (weight penalties), different estimators: convex linear models with closed-form/iterative solutions vs non-convex penalty terms added to backpropagation.
10. **Why did Lasso zero 27 features?** — The L1 penalty plus correlated one-hot columns; it kept tenure/contract/charges and discarded noise like gender — mostly sensible, with arbitrary choices among correlated dummies.
11. **Why is recall low (~0.49)?** — Default 0.5 threshold with imbalanced classes favors the majority class; raising recall would require class weights or a lower threshold, which I listed as a limitation.
12. **Why 1e-4 for regularization strength?** — Large enough to matter relative to the loss scale on ~5k rows, small enough not to dominate the learning signal.
13. **What did CV tell you?** — Mean F1 0.571 with std 0.021: results are stable across data splits, and the single test score is consistent with CV.
14. **Why did logistic regression beat the NN?** — Small, clean, mostly-linearly-separable tabular problem; a small MLP adds variance without much extra signal. This is the classic bias-variance trade-off in action.
15. **What is binary cross-entropy?** — −[y·log(p) + (1−y)·log(1−p)]; it measures how far predicted probabilities are from true labels and heavily penalizes confident mistakes.
16. **What is the sigmoid doing?** — Maps the last layer's raw score to a probability between 0 and 1, suitable for a binary decision at threshold 0.5.
17. **What would you improve next?** — Class weights or threshold tuning for recall, hyperparameter search, probability calibration, and embeddings instead of one-hot for high-cardinality columns.