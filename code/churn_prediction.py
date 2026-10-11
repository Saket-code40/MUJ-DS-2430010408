# =====================================================================
# Customer Churn Prediction using Feature Engineering and a
# Regularized Neural Network
#
# Dataset : Telco Customer Churn (IBM sample dataset)
# Goal    : Educational Deep Learning project demonstrating bias/variance,
#           cross-validation, dropout, L1/L2 regularization, early
#           stopping, loss curves and classification metrics.
#
# Run with:  /usr/bin/python3.12 churn_prediction.py
# =====================================================================

# ---------------------------------------------------------------------
# SECTION 1: Import libraries
# ---------------------------------------------------------------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # silence TF info logs

import random
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # save figures without a window
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers, callbacks

# Reproducible random seed (same numbers every time we run the script)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:  # running inside a Jupyter notebook
    BASE_DIR = os.getcwd()
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 70)
print("CUSTOMER CHURN PREDICTION — Feature Engineering + Regularized NN")
print("=" * 70)

# ---------------------------------------------------------------------
# SECTION 2: Load dataset
# ---------------------------------------------------------------------
DATA_PATH = os.path.join(BASE_DIR, "Telco-Customer-Churn.csv")
df = pd.read_csv(DATA_PATH)

print("\n--- SECTION 2: DATASET LOADED ---")
print("Shape:", df.shape)

# ---------------------------------------------------------------------
# SECTION 3: Explore dataset (EDA)
# ---------------------------------------------------------------------
print("\n--- SECTION 3: DATASET EXPLORATION ---")
print("Number of samples :", len(df))
print("Number of columns :", df.shape[1])

print("\nClass distribution (Churn column):")
print(df["Churn"].value_counts())

counts = df["Churn"].value_counts(normalize=True) * 100
print("\nPercentage of customers:")
print(f"  Stayed (No)  : {counts['No']:.2f}%")
print(f"  Churned (Yes): {counts['Yes']:.2f}%")

print("\nMissing values per column:")
print(df.isna().sum()[df.isna().sum() > 0])

print("\nDuplicate rows:", df.duplicated().sum())

print("\nData types:")
print(df.dtypes)

# ---------------------------------------------------------------------
# SECTION 4: Visualize class distribution
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
df["Churn"].value_counts().plot.bar(ax=ax, color=["#4c72b0", "#dd8452"])
ax.set_title("Class Distribution — Churn vs Stayed")
ax.set_xlabel("Churn")
ax.set_ylabel("Number of customers")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for i, v in enumerate(df["Churn"].value_counts()):
    ax.text(i, v + 40, str(v), ha="center")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "class_distribution.png"), dpi=150)
plt.close(fig)

print("\nWHY ACCURACY ALONE IS MISLEADING HERE:")
print(f"  The dataset is imbalanced: only {counts['Yes']:.1f}% churned.")
print("  A lazy model that predicts 'No churn' for everyone would score")
print(f"  ~{counts['No']:.0f}% accuracy while catching ZERO churners.")
print("  So we must also report Precision, Recall and F1-score.")

# ---------------------------------------------------------------------
# SECTION 5: Clean / preprocess data
# ---------------------------------------------------------------------
print("\n--- SECTION 5: DATA CLEANING & PREPROCESSING ---")

# 5.1 Handle TotalCharges --------------------------------------------
# TotalCharges is stored as TEXT. Some rows contain a blank string " "
# instead of a number, so pandas read them as strings.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
blank_tc = df["TotalCharges"].isna()
print(f"TotalCharges blank/invalid rows found: {blank_tc.sum()}")
print("Their tenure values:", df.loc[blank_tc, "tenure"].unique().tolist())

# Chosen strategy: DROP these rows.
# Reason: all of them are brand-new customers (tenure == 0), so their
# TotalCharges is genuinely unknown/0. Dropping 11 of 7043 rows removes
# only 0.16% of the data and avoids injecting fake values.
df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
print(f"Rows after dropping blanks: {len(df)}")

# 5.2 Target: Churn Yes -> 1, No -> 0 --------------------------------
y = (df["Churn"] == "Yes").astype(int)

# 5.3 Drop the identifier and the target itself from the features -----
X = df.drop(columns=["customerID", "Churn"])

# 5.4 Identify numeric vs categorical columns -------------------------
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
binary_cols = [c for c in X.columns
               if X[c].dtype == object and X[c].nunique() == 2]
categorical_cols = [c for c in X.columns
                    if c not in binary_cols and c not in numeric_cols]

print("\nNumeric columns    :", numeric_cols)
print("Binary columns     :", binary_cols)
print("Categorical columns:", categorical_cols)

# 5.5 Binary encode: Yes -> 1, No -> 0 --------------------------------
# (SeniorCitizen is already 0/1.)
X = X.copy()
for c in binary_cols:
    X[c] = X[c].map({"Yes": 1, "No": 0})

# ---------------------------------------------------------------------
# SECTION 6: Split dataset (70% train / 15% val / 15% test, stratified)
# ---------------------------------------------------------------------
# TRAIN  -> fit the model (learn weights)
# VAL    -> monitor generalization, early stopping, pick hyperparameters
# TEST   -> final, completely unseen evaluation (used ONCE at the end)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)

print("\n--- SECTION 6: DATASET SPLIT ---")
print(f"Training samples : {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples      : {len(X_test)}")

# ---------------------------------------------------------------------
# One-hot encoding + scaling (NO LEAKAGE: fit on TRAIN only)
# ---------------------------------------------------------------------
# One-hot encoding: turn a categorical column (e.g. Contract with 3
# values) into 3 binary columns. A model cannot do math on words, and
# plain label numbers would fake an order that does not exist.

X_train_enc = pd.get_dummies(X_train, columns=categorical_cols)
# Reindex val/test to exactly the train columns -> same feature order,
# no leakage of categories that appear only in val/test.
X_val_enc = pd.get_dummies(X_val, columns=categorical_cols).reindex(
    columns=X_train_enc.columns, fill_value=0)
X_test_enc = pd.get_dummies(X_test, columns=categorical_cols).reindex(
    columns=X_train_enc.columns, fill_value=0)

# Feature scaling: StandardScaler transforms each numeric column to
# mean 0 / std 1. Neural networks train much faster and more stably
# when all inputs live on a similar scale. Fitted on TRAIN ONLY.
scaler = StandardScaler()
X_train_enc[numeric_cols] = scaler.fit_transform(X_train_enc[numeric_cols])
X_val_enc[numeric_cols] = scaler.transform(X_val_enc[numeric_cols])
X_test_enc[numeric_cols] = scaler.transform(X_test_enc[numeric_cols])

feature_names = list(X_train_enc.columns)
X_train_np = X_train_enc.values.astype("float32")
X_val_np = X_val_enc.values.astype("float32")
X_test_np = X_test_enc.values.astype("float32")

print(f"Number of features after one-hot encoding: {len(feature_names)}")

# ---------------------------------------------------------------------
# Helper: metrics dictionary + model builder
# ---------------------------------------------------------------------
def metrics_dict(y_true, y_pred):
    """Compute the four classification metrics for one set of predictions."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
    }

def build_model(dropout=0.0, l1=0.0, l2=0.0, seed=SEED):
    """Small MLP: 64 -> dropout -> 32 -> 1 (sigmoid).

    dropout : dropout rate after the first layer
    l1, l2  : regularization strengths applied to the weights
    """
    reg = None
    if l1 > 0 and l2 > 0:
        reg = regularizers.l1_l2(l1=l1, l2=l2)
    elif l1 > 0:
        reg = regularizers.l1(l1)
    elif l2 > 0:
        reg = regularizers.l2(l2)

    keras.utils.set_random_seed(seed)
    model = keras.Sequential([
        layers.Input(shape=(X_train_np.shape[1],)),
        layers.Dense(64, activation="relu", kernel_regularizer=reg),
        layers.Dropout(dropout),
        layers.Dense(32, activation="relu", kernel_regularizer=reg),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

def train_model(model, epochs=200, patience=8, batch_size=32):
    """Train with early stopping on validation loss; restore best weights."""
    es = callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                 restore_best_weights=True)
    hist = model.fit(X_train_np, y_train,
                     validation_data=(X_val_np, y_val),
                     epochs=epochs, batch_size=batch_size,
                     callbacks=[es], verbose=0)
    best_epoch = int(np.argmin(hist.history["val_loss"])) + 1
    return hist, best_epoch, len(hist.history["val_loss"])

def evaluate_model(model, X, y_true, threshold=0.5):
    """Predict classes (probability > threshold) and compute metrics."""
    probs = model.predict(X, verbose=0).ravel()
    preds = (probs >= threshold).astype(int)
    return metrics_dict(y_true, preds), preds

results = {}   # collects every number for the final report

# ---------------------------------------------------------------------
# SECTION 7: Baseline model — Logistic Regression
# ---------------------------------------------------------------------
print("\n--- SECTION 7: BASELINE MODEL (LOGISTIC REGRESSION) ---")
print("A baseline is a simple reference model. If our neural network")
print("cannot beat logistic Regression, the extra complexity is not justified.")

logreg = LogisticRegression(max_iter=2000, random_state=SEED)
logreg.fit(X_train_np, y_train)
lg_preds_test = logreg.predict(X_test_np)
lg_metrics = metrics_dict(y_test, lg_preds_test)

print("Logistic Regression TEST metrics:", lg_metrics)
print(classification_report(y_test, lg_preds_test,
                            target_names=["Stayed", "Churned"]))
results["logistic_regression"] = lg_metrics

# ---------------------------------------------------------------------
# SECTION 8 + 9 + 10: Main Neural Network with Dropout + Early stopping
# ---------------------------------------------------------------------
print("\n--- SECTION 8/9/10: NEURAL NETWORK (Dropout 0.3, Early Stopping) ---")
print("Architecture:")
print("  Input (scaled/encoded features)")
print("  -> Dense(64, relu)   : learns 64 weighted combinations of features")
print("  -> Dropout(0.3)      : randomly silences 30% of neurons each step")
print("  -> Dense(32, relu)   : a smaller second layer refines the patterns")
print("  -> Dense(1, sigmoid) : squashes output to a churn probability [0,1]")

main_model = build_model(dropout=0.3)
hist_main, best_epoch_main, stopped_epoch_main = train_model(main_model)
print(f"\nEARLY STOPPING REPORT:")
print(f"Training was stopped at epoch {stopped_epoch_main} because validation "
      f"loss stopped improving (best epoch = {best_epoch_main}, "
      f"patience = 8).")

nn_metrics, nn_preds_test = evaluate_model(main_model, X_test_np, y_test)
print("Neural Network (Dropout 0.3) TEST metrics:", nn_metrics)
results["nn_dropout"] = nn_metrics
results["early_stopping"] = {
    "best_epoch": best_epoch_main,
    "stopped_epoch": stopped_epoch_main,
    "patience": 8,
}

# Graph 1 — Training vs Validation LOSS -------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(hist_main.history["loss"], label="Training loss")
ax.plot(hist_main.history["val_loss"], label="Validation loss")
ax.axvline(best_epoch_main, color="red", linestyle="--", linewidth=1,
           label=f"Best epoch ({best_epoch_main}) — early stopping")
ax.set_title("Training Loss vs Validation Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Binary cross-entropy loss")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "graph1_loss_curves.png"), dpi=150)
plt.close(fig)

# Graph 2 — Training vs Validation ACCURACY ----------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(hist_main.history["accuracy"], label="Training accuracy")
ax.plot(hist_main.history["val_accuracy"], label="Validation accuracy")
ax.axvline(best_epoch_main, color="red", linestyle="--", linewidth=1,
           label=f"Best epoch ({best_epoch_main})")
ax.set_title("Training Accuracy vs Validation Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "graph2_accuracy_curves.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------
# SECTION 11 (part): Bias vs Variance demonstration
# ---------------------------------------------------------------------
print("\n--- SECTION 11: BIAS / VARIANCE DEMO ---")
m_train = metrics_dict(y_train, (main_model.predict(X_train_np, verbose=0).ravel() >= 0.5).astype(int))
m_val = metrics_dict(y_val, (main_model.predict(X_val_np, verbose=0).ravel() >= 0.5).astype(int))
print(f"Main model — train accuracy: {m_train['accuracy']:.4f}, "
      f"val accuracy: {m_val['accuracy']:.4f}, "
      f"gap: {m_train['accuracy']-m_val['accuracy']:.4f}")
print("Small gap  -> balanced bias/variance (good fit).")
print("Big gap    -> high variance (overfitting).")
results["bias_variance_main"] = {"train_accuracy": m_train["accuracy"],
                                 "val_accuracy": m_val["accuracy"]}

# High-bias demo: single perceptron (too simple -> underfits)
keras.utils.set_random_seed(SEED)
underfit = keras.Sequential([
    layers.Input(shape=(X_train_np.shape[1],)),
    layers.Dense(1, activation="sigmoid"),
])
underfit.compile(optimizer=keras.optimizers.Adam(1e-2),
                 loss="binary_crossentropy", metrics=["accuracy"])
hist_under = underfit.fit(X_train_np, y_train,
                          validation_data=(X_val_np, y_val),
                          epochs=30, batch_size=32, verbose=0)
u_train = metrics_dict(y_train, (underfit.predict(X_train_np, verbose=0).ravel() >= 0.5).astype(int))
u_val = metrics_dict(y_val, (underfit.predict(X_val_np, verbose=0).ravel() >= 0.5).astype(int))
print(f"\nUnderfitting model (1 neuron) — train acc {u_train['accuracy']:.4f}, "
      f"val acc {u_val['accuracy']:.4f}, val F1 {u_val['f1']:.4f}")
print("Both plateau at the same level with ~0 gap -> HIGH BIAS")
print("(too simple to capture more of the signal).")
results["bias_variance_underfit"] = {"train_accuracy": u_train["accuracy"],
                                     "val_accuracy": u_val["accuracy"]}

# High-variance demo: a DELIBERATELY oversized network, no dropout,
# no regularization, no early stopping, trained for a long time.
# This is NOT a model we would deploy — it exists to visualize what
# overfitting looks like on real curves.
keras.utils.set_random_seed(SEED)
oversized = keras.Sequential([
    layers.Input(shape=(X_train_np.shape[1],)),
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
oversized.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
hist_over = oversized.fit(X_train_np, y_train,
                          validation_data=(X_val_np, y_val),
                          epochs=150, batch_size=32, verbose=0)
o_train = metrics_dict(y_train, (oversized.predict(X_train_np, verbose=0).ravel() >= 0.5).astype(int))
o_val = metrics_dict(y_val, (oversized.predict(X_val_np, verbose=0).ravel() >= 0.5).astype(int))
gap_over = o_train["accuracy"] - o_val["accuracy"]
gap_main = m_train["accuracy"] - m_val["accuracy"]
print(f"Oversized model (no reg., 150 epochs) — train acc "
      f"{o_train['accuracy']:.4f}, val acc {o_val['accuracy']:.4f}, "
      f"gap {gap_over:.4f}")
print(f"Main model (dropout + early stopping) — train acc "
      f"{m_train['accuracy']:.4f}, val acc {m_val['accuracy']:.4f}, "
      f"gap {gap_main:.4f}")
print("The oversized model memorizes the training set: the large train/val")
print("gap is the fingerprint of HIGH VARIANCE (overfitting).")
results["bias_variance_overfit"] = {"train_accuracy": o_train["accuracy"],
                                    "val_accuracy": o_val["accuracy"],
                                    "gap": round(float(gap_over), 4)}

# ---------------------------------------------------------------------
# SECTION 12: K-Fold Cross Validation (K = 5)
# ---------------------------------------------------------------------
print("\n--- SECTION 12: 5-FOLD CROSS-VALIDATION ---")
print("Methodology: the TRAINING set is split into 5 folds. For each fold,")
print("the scaler is refit on the 4 training folds only (no leakage) and")
print("the NN is trained with early stopping using the held-out fold.")
print("The TEST set is never touched here.")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_records = []
num_positions = [X_train_enc.columns.get_loc(c) for c in numeric_cols]

for fold, (tr_idx, fo_idx) in enumerate(skf.split(X_train_np, y_train), 1):
    # Re-scale inside the fold to avoid any leakage between folds.
    # Only the numeric columns are scaled, exactly like the main pipeline.
    fold_scaler = StandardScaler()
    Xtr, Xfo = X_train_np[tr_idx].copy(), X_train_np[fo_idx].copy()
    Xtr[:, num_positions] = fold_scaler.fit_transform(Xtr[:, num_positions])
    Xfo[:, num_positions] = fold_scaler.transform(Xfo[:, num_positions])

    # Carve a small validation slice OUT OF the training portion for
    # early stopping, so the held-out fold stays completely unseen.
    Xtr_fit, Xtr_val, ytr_fit, ytr_val = train_test_split(
        Xtr, y_train.values[tr_idx], test_size=0.15,
        random_state=SEED, stratify=y_train.values[tr_idx])

    keras.utils.set_random_seed(SEED)
    fold_model = build_model(dropout=0.3)
    es = callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                 restore_best_weights=True)
    fold_model.fit(Xtr_fit, ytr_fit,
                   validation_data=(Xtr_val, ytr_val),
                   epochs=150, batch_size=32, callbacks=[es], verbose=0)
    fold_pred = (fold_model.predict(Xfo, verbose=0).ravel() >= 0.5).astype(int)
    rec = metrics_dict(y_train.values[fo_idx], fold_pred)
    rec["fold"] = fold
    cv_records.append(rec)
    print(f"Fold {fold}: accuracy={rec['accuracy']:.4f} "
          f"precision={rec['precision']:.4f} recall={rec['recall']:.4f} "
          f"f1={rec['f1']:.4f}")
    keras.backend.clear_session()

cv_df = pd.DataFrame(cv_records)
cv_summary = {m: {"mean": round(float(cv_df[m].mean()), 4),
                  "std": round(float(cv_df[m].std()), 4)}
              for m in ["accuracy", "precision", "recall", "f1"]}
print("\nCross-validation summary (mean +/- std over 5 folds):")
for m, s in cv_summary.items():
    print(f"  {m:>9}: {s['mean']:.4f} +/- {s['std']:.4f}")
results["cross_validation"] = {"folds": cv_records, "summary": cv_summary}

# Graph 3 — K=5 CV F1 per fold -----------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(cv_df["fold"], cv_df["f1"], color="#4c72b0")
mean_f1, std_f1 = cv_df["f1"].mean(), cv_df["f1"].std()
ax.axhline(mean_f1, color="red", linestyle="--",
           label=f"Mean F1 = {mean_f1:.3f} +/- {std_f1:.3f}")
ax.set_xticks(cv_df["fold"])
ax.set_title("5-Fold Cross-Validation — F1-score per Fold")
ax.set_xlabel("Fold")
ax.set_ylabel("F1-score")
ax.set_ylim(0, 1)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "graph3_cv_f1.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------
# SECTION 13: Dropout experiment (no dropout vs dropout 0.3)
# ---------------------------------------------------------------------
print("\n--- SECTION 13: DROPOUT EXPERIMENT ---")
# Same architecture as the main model, but with dropout switched OFF,
# so the ONLY difference between the two runs is the dropout layer.
nodrop_model = build_model(dropout=0.0)
hist_nodrop, _, _ = train_model(nodrop_model)
nodrop_metrics, _ = evaluate_model(nodrop_model, X_test_np, y_test)
nodrop_train = metrics_dict(y_train, (nodrop_model.predict(X_train_np, verbose=0).ravel() >= 0.5).astype(int))
nodrop_val = metrics_dict(y_val, (nodrop_model.predict(X_val_np, verbose=0).ravel() >= 0.5).astype(int))
gap_nodrop = nodrop_train["accuracy"] - nodrop_val["accuracy"]
print("No dropout     TEST:", nodrop_metrics, f"| train/val gap {gap_nodrop:+.4f}")
print("Dropout 0.3    TEST:", nn_metrics, f"| train/val gap {gap_main:+.4f}")
results["nn_no_dropout"] = nodrop_metrics

# ---------------------------------------------------------------------
# SECTION 14 + 15: L1 and L2 regularization experiments
# ---------------------------------------------------------------------
print("\n--- SECTION 14/15: L1 & L2 REGULARIZATION EXPERIMENTS ---")
print("Regularization strength 1e-4 is small enough not to cripple learning")
print("but large enough to penalize large weights on 46 features / 4900 rows.")

reg_records = {}
for name, kwargs in [("nn_l1", dict(l1=1e-4)), ("nn_l2", dict(l2=1e-4))]:
    keras.backend.clear_session()
    model = build_model(dropout=0.3, **kwargs)
    hist, be, se = train_model(model)
    tr_loss = model.evaluate(X_train_np, y_train, verbose=0)[0]
    va_loss = model.evaluate(X_val_np, y_val, verbose=0)[0]
    mets, _ = evaluate_model(model, X_test_np, y_test)
    mets["train_loss"] = round(float(tr_loss), 4)
    mets["val_loss"] = round(float(va_loss), 4)
    reg_records[name] = mets
    results[name] = mets
    print(f"{name}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
          f"TEST={ {k: mets[k] for k in ['accuracy','precision','recall','f1']} }")

# ---------------------------------------------------------------------
# SECTION 16: Ridge and Lasso (classical linear models)
# ---------------------------------------------------------------------
print("\n--- SECTION 16: RIDGE & LASSO (CLASSICAL LINEAR MODELS) ---")
print("Ridge = linear model + L2 penalty. Lasso = linear model + L1 penalty.")
print("These are NOT the same thing as NN L1/L2 penalties, but the idea")
print("(punish large weights) is the same family of techniques.")

ridge = Ridge(alpha=1.0, random_state=SEED)
ridge.fit(X_train_np, y_train)
ridge_pred = (ridge.predict(X_test_np) >= 0.5).astype(int)
ridge_metrics = metrics_dict(y_test, ridge_pred)
print("Ridge  TEST:", ridge_metrics)
results["ridge"] = ridge_metrics

lasso = Lasso(alpha=0.005, max_iter=10000, random_state=SEED)
lasso.fit(X_train_np, y_train)
lasso_pred = (lasso.predict(X_test_np) >= 0.5).astype(int)
lasso_metrics = metrics_dict(y_test, lasso_pred)
print("Lasso  TEST:", lasso_metrics)
results["lasso"] = lasso_metrics

zero_feats = [f for f, c in zip(feature_names, lasso.coef_) if abs(c) < 1e-8]
print(f"\nLasso drove {len(zero_feats)} of {len(feature_names)} feature "
      f"coefficients to zero:")
for f in zero_feats:
    print("   -", f)
results["lasso_zeroed_features"] = zero_feats

# ---------------------------------------------------------------------
# SECTION 17: Graphs (4 and 5; 1-3 already saved)
# ---------------------------------------------------------------------
print("\n--- SECTION 17: REMAINING GRAPHS ---")

# Graph 4 — Confusion matrix of the final NN on the TEST set ----------
cm = confusion_matrix(y_test, nn_preds_test)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Predicted Stayed", "Predicted Churn"],
            yticklabels=["True Stayed", "True Churn"])
ax.set_title("Confusion Matrix — Neural Network (test set)")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "graph4_confusion_matrix.png"), dpi=150)
plt.close(fig)

# Graph 5 — Regularization comparison (F1) -----------------------------
reg_compare = {
    "No reg.": nodrop_metrics["f1"],
    "Dropout": nn_metrics["f1"],
    "L1": reg_records["nn_l1"]["f1"],
    "L2": reg_records["nn_l2"]["f1"],
}
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(reg_compare.keys(), reg_compare.values(),
       color=["#4c72b0", "#55a868", "#c44e52", "#8172b3"])
for i, v in enumerate(reg_compare.values()):
    ax.text(i, v + 0.005, f"{v:.3f}", ha="center")
ax.set_title("Regularization Comparison — Test F1-score")
ax.set_ylabel("F1-score")
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "graph5_regularization_f1.png"), dpi=150)
plt.close(fig)

# Graph 6 (bonus) — Precision / Recall / F1 for every model -----------
table_models = ["logistic_regression", "nn_no_dropout", "nn_dropout",
                "nn_l1", "nn_l2"]
table_labels = ["Logistic Reg.", "NN (no dropout)", "NN + Dropout",
                "NN + L1", "NN + L2"]
fig, ax = plt.subplots(figsize=(9, 4.5))
xs = np.arange(len(table_models))
w = 0.26
ax.bar(xs - w, [results[m]["precision"] for m in table_models], w, label="Precision")
ax.bar(xs,       [results[m]["recall"] for m in table_models],    w, label="Recall")
ax.bar(xs + w,   [results[m]["f1"] for m in table_models],        w, label="F1")
ax.set_xticks(xs, table_labels, rotation=15)
ax.set_title("Precision / Recall / F1 by Model (test set)")
ax.set_ylim(0, 1)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "graph6_precision_recall_f1.png"), dpi=150)
plt.close(fig)

# Graph 7 (bonus) — Top logistic-regression coefficients --------------
coef = pd.Series(logreg.coef_.ravel(), index=feature_names)
top = coef.reindex(coef.abs().sort_values(ascending=False).index).head(12)
fig, ax = plt.subplots(figsize=(8, 5))
top.plot.bar(ax=ax, color=["#c44e52" if v > 0 else "#4c72b0" for v in top])
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Top 12 Logistic Regression Coefficients (business drivers of churn)")
ax.set_ylabel("Coefficient (positive -> more churn)")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "graph7_feature_importance.png"), dpi=150)
plt.close(fig)

# Graph 8 (bonus) — Bias/Variance: the oversized model overfits -------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(hist_over.history["loss"], label="Training loss")
axes[0].plot(hist_over.history["val_loss"], label="Validation loss")
axes[0].set_title("Oversized model (no regularization): loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Binary cross-entropy loss")
axes[0].legend()
axes[1].plot(hist_over.history["accuracy"], label="Training accuracy")
axes[1].plot(hist_over.history["val_accuracy"], label="Validation accuracy")
axes[1].set_title("Oversized model (no regularization): accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "graph8_bias_variance.png"), dpi=150)
plt.close(fig)

print("Saved figures:", sorted(os.listdir(FIG_DIR)))

# ---------------------------------------------------------------------
# SECTION 18: Final comparison table
# ---------------------------------------------------------------------
print("\n--- SECTION 18: FINAL COMPARISON (TEST SET) ---")
final_table = pd.DataFrame({
    "Model/Experiment": ["Logistic Regression",
                         "Neural Network (no dropout)",
                         "Neural Network + Dropout",
                         "Neural Network + L1",
                         "Neural Network + L2",
                         "Ridge (linear)",
                         "Lasso (linear)"],
    "Accuracy":  [results["logistic_regression"]["accuracy"],
                  results["nn_no_dropout"]["accuracy"],
                  results["nn_dropout"]["accuracy"],
                  results["nn_l1"]["accuracy"],
                  results["nn_l2"]["accuracy"],
                  results["ridge"]["accuracy"],
                  results["lasso"]["accuracy"]],
    "Precision": [results["logistic_regression"]["precision"],
                  results["nn_no_dropout"]["precision"],
                  results["nn_dropout"]["precision"],
                  results["nn_l1"]["precision"],
                  results["nn_l2"]["precision"],
                  results["ridge"]["precision"],
                  results["lasso"]["precision"]],
    "Recall":    [results["logistic_regression"]["recall"],
                  results["nn_no_dropout"]["recall"],
                  results["nn_dropout"]["recall"],
                  results["nn_l1"]["recall"],
                  results["nn_l2"]["recall"],
                  results["ridge"]["recall"],
                  results["lasso"]["recall"]],
    "F1":        [results["logistic_regression"]["f1"],
                  results["nn_no_dropout"]["f1"],
                  results["nn_dropout"]["f1"],
                  results["nn_l1"]["f1"],
                  results["nn_l2"]["f1"],
                  results["ridge"]["f1"],
                  results["lasso"]["f1"]],
})
print(final_table.to_string(index=False))

print("\nConfusion matrix (final NN):")
print("                Predicted Stayed   Predicted Churn")
print(f"  True Stayed   TN = {cm[0,0]:<17} FP = {cm[0,1]}")
print(f"  True Churn    FN = {cm[1,0]:<17} TP = {cm[1,1]}")
results["confusion_matrix"] = {"TN": int(cm[0, 0]), "FP": int(cm[0, 1]),
                               "FN": int(cm[1, 0]), "TP": int(cm[1, 1])}
results["final_table"] = final_table.to_dict(orient="records")
results["feature_names"] = feature_names
results["top_logreg_coefficients"] = {k: round(float(v), 4)
                                      for k, v in top.items()}

# ---------------------------------------------------------------------
# SECTION 19: Conclusions
# ---------------------------------------------------------------------
print("\n--- SECTION 19: CONCLUSIONS ---")
best_row = final_table.loc[final_table["F1"].idxmax()]
print(f"1. Best model by F1          : {best_row['Model/Experiment']} "
      f"(F1={best_row['F1']})")
print("2. Dataset imbalanced        : Yes (~26.5% churn)")
print(f"3. Accuracy misleading?      : Yes — a dummy 'always stay' model "
      f"would reach ~{counts['No']:.0f}% accuracy.")
print(f"4. Overfitting observed?     : Yes — oversized unregularized model "
      f"gap +{gap_over:.3f} vs main model gap {gap_main:+.3f}")
print(f"5. Dropout effect            : F1 {nodrop_metrics['f1']} -> "
      f"{nn_metrics['f1']}")
print(f"6. L1 effect                 : F1 {nn_metrics['f1']} -> "
      f"{reg_records['nn_l1']['f1']}")
print(f"7. L2 effect                 : F1 {nn_metrics['f1']} -> "
      f"{reg_records['nn_l2']['f1']}")
best_reg = max(reg_compare, key=reg_compare.get)
print(f"8. Best regularization       : {best_reg}")
print(f"9. CV said about variance    : F1 = {mean_f1:.3f} +/- {std_f1:.3f}")
print(f"10. Early stopping epoch     : stopped at {stopped_epoch_main}, "
      f"best = {best_epoch_main}")
print("11. Why that epoch           : val loss did not improve for 8 epochs")
print(f"12. NN vs baseline           : logistic F1 "
      f"{lg_metrics['f1']} vs NN F1 {nn_metrics['f1']}")
print("13. Top churn drivers        :",
      ", ".join(list(results['top_logreg_coefficients'])[:5]))
print("14. Limitations              : small dataset, single split for final")
print("                               evaluation, no class-weight tuning,")
print("                               no hyperparameter search.")

with open(os.path.join(FIG_DIR, "..", "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nAll results saved to results.json and figures to ./figures/")