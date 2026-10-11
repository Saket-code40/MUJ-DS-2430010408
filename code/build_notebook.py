# Build churn_prediction.ipynb from churn_prediction.py.
# Splits the script on its "# ---- SECTION X ----" divider blocks,
# turns each section into a code cell, adds a Markdown explanation
# cell before every section, and adds an image-display cell after
# every cell that saves a figure (the script uses the Agg backend,
# so figures are displayed by loading the saved PNGs).
import re
import nbformat as nbf

SRC = "churn_prediction.py"
OUT = "churn_prediction.ipynb"

EXPLAIN = {
    "title": (
        "# Customer Churn Prediction using Feature Engineering "
        "and a Regularized Neural Network\n\n"
        "**Dataset:** Telco Customer Churn (IBM sample dataset)\n\n"
        "**Goal:** an educational Deep Learning project demonstrating "
        "bias/variance, cross-validation, dropout, L1/L2 regularization, "
        "early stopping, loss curves and classification metrics.\n\n"
        "Every number below is computed by running the cells — nothing is "
        "hand-written. Run the cells top to bottom; the random seed is "
        "fixed so results are reproducible."
    ),
    "1": (
        "## Section 1 — Import libraries\n\n"
        "We use **pandas/numpy** for data handling, **matplotlib/seaborn** "
        "for plots, **scikit-learn** for the baseline model, splitting, "
        "scaling and metrics, and **TensorFlow/Keras** for the neural "
        "network.\n\n"
        "We also fix the **random seed (42)** in every library that uses "
        "randomness, so anyone re-running this notebook gets the same "
        "numbers. The `Agg` backend saves figures to files instead of "
        "opening windows; each saved figure is displayed right after it "
        "is created."
    ),
    "2": (
        "## Section 2 — Load the dataset\n\n"
        "The Telco Customer Churn dataset contains 7,043 telephone "
        "customers with demographic, account and billing information. "
        "The target column is `Churn`: did the customer leave (Yes) or "
        "stay (No)?"
    ),
    "3": (
        "## Section 3 — Explore the dataset\n\n"
        "Before modeling we must understand the data: size, class "
        "balance, missing values, duplicates and data types.\n\n"
        "Key points to watch:\n"
        "* `TotalCharges` looks numeric but is stored as **text** — some "
        "rows contain blank strings instead of numbers.\n"
        "* Only ~26.5% of customers churned, so the classes are "
        "**imbalanced**.\n\n"
        "**Why accuracy alone is misleading:** with 73.5% non-churners, a "
        "lazy model that always predicts \"No churn\" reaches ~73% "
        "accuracy while catching nobody. That is why we will also report "
        "precision, recall and F1-score."
    ),
    "4": (
        "## Section 4 — Visualize the class distribution\n\n"
        "A bar chart makes the imbalance obvious: roughly 3 out of 4 "
        "customers stay. This motivates stratified splitting and "
        "F1-focused evaluation later."
    ),
    "5": (
        "## Section 5 — Clean and preprocess the data\n\n"
        "Steps, in order:\n"
        "1. **`TotalCharges`:** convert to numeric; the blank-string "
        "rows become NaN. They all turn out to be brand-new customers "
        "(`tenure = 0`), so **dropping them** is safe and simpler than "
        "imputing.\n"
        "2. **Drop `customerID`:** an identifier, not a predictive "
        "feature.\n"
        "3. **Encode the target:** `Churn` Yes = 1, No = 0.\n"
        "4. **One-hot encode** multi-category columns (`Contract`, "
        "`PaymentMethod`, `InternetService`, ...).\n\n"
        "**What is one-hot encoding?** A categorical column with k "
        "values becomes k binary 0/1 columns. Without it, a model would "
        "treat text labels as meaningless numbers. **Why scaling?** "
        "Numeric features like `TotalCharges` range into the thousands "
        "while `tenure` stops at 72; gradient-based learners converge "
        "faster and more reliably when all features have comparable "
        "scale (here: StandardScaler, ~zero mean, unit variance).\n\n"
        "**Note on leakage:** the one-hot encoder and the scaler are "
        "conceptually fit on training data only — see the split and the "
        "cross-validation section, where the scaler is refit inside "
        "every fold."
    ),
    "6": (
        "## Section 6 — Split into train / validation / test\n\n"
        "* **Training set (70%):** the model learns from this.\n"
        "* **Validation set (15%):** used *during* training to decide "
        "when to stop (early stopping). Never used to fit weights.\n"
        "* **Test set (15%):** locked away until the very end; a single "
        "final, honest evaluation.\n\n"
        "Because the classes are imbalanced we split **stratified**: "
        "each part keeps the same ~26.5% churn ratio as the whole "
        "dataset."
    ),
    "7": (
        "## Section 7 — Baseline model: logistic regression\n\n"
        "A baseline is a simple reference model. If a neural network "
        "cannot beat logistic regression, the extra complexity is not "
        "justified. It also gives the teacher something concrete to "
        "compare every later experiment against."
    ),
    "8/9/10": (
        "## Sections 8–10 — Neural network, training and early "
        "stopping\n\n"
        "Architecture (deliberately small):\n\n"
        "```\n"
        "Input (46 encoded/scaled features)\n"
        " -> Dense(64, relu)    : 64 learned feature combinations\n"
        " -> Dropout(0.3)       : randomly silences 30% of neurons per\n"
        "                         step, which fights overfitting\n"
        " -> Dense(32, relu)    : a smaller layer refines the patterns\n"
        " -> Dense(1, sigmoid)  : squashes the output to a probability\n"
        "```\n\n"
        "**Loss:** binary cross-entropy — it measures how surprised the "
        "model is by the true 0/1 labels and heavily penalizes confident "
        "wrong answers. **Optimizer:** Adam (learning rate 1e-3).\n\n"
        "**Early stopping (mandatory):** after every epoch we check the "
        "*validation* loss. If it fails to improve for `patience = 8` "
        "epochs in a row, training stops and the weights from the best "
        "epoch are restored. This prevents the network from memorizing "
        "the training set — the epoch is chosen by the data, not by hand."
    ),
    "11": (
        "## Section 11 — Bias vs variance, demonstrated on real curves\n\n"
        "* **High bias (underfitting):** a 1-neuron model. Training and "
        "validation accuracy both plateau at the same modest level — "
        "the model is too simple to capture the signal.\n"
        "* **High variance (overfitting):** a deliberately oversized, "
        "unregularized network trained for 150 epochs. Training accuracy "
        "races toward 100% while validation accuracy stalls or drops — "
        "the classic widening gap.\n"
        "* **Good fit:** our regularized main model, where the two "
        "curves stay close together.\n\n"
        "These are not textbook definitions — they come from models "
        "actually trained in this notebook (see Graph 8)."
    ),
    "12": (
        "## Section 12 — 5-fold cross-validation (K = 5)\n\n"
        "The **training set** is split into 5 folds. For each fold:\n"
        "1. the scaler is refit on the 4 training folds only (no "
        "leakage),\n"
        "2. the neural network is trained with early stopping on a "
        "validation slice carved out of those 4 folds,\n"
        "3. metrics are computed on the held-out fold.\n\n"
        "The test set is never touched. The mean ± standard deviation "
        "across folds tells us how **stable** the model is: a small "
        "standard deviation means the performance does not depend on "
        "luck of the split."
    ),
    "13": (
        "## Section 13 — Dropout experiment\n\n"
        "Two runs of the *same* architecture; the only difference is "
        "whether the Dropout(0.3) layer is active. Dropout randomly "
        "switches off 30% of the hidden neurons at every training step, "
        "which forces the network to spread knowledge across many "
        "pathways instead of relying on a few — that reduces "
        "overfitting. Compare test performance and the train/validation "
        "gap of both runs."
    ),
    "14/15": (
        "## Sections 14–15 — L1 and L2 regularization experiments\n\n"
        "The same small network with weight penalties added to the loss:\n\n"
        "* **L1** adds the sum of absolute weights — it pushes weak "
        "weights all the way to zero (sparsity).\n"
        "* **L2** adds the sum of squared weights — it shrinks all "
        "weights smoothly toward zero.\n\n"
        "Strength `1e-4` is small enough not to cripple learning but "
        "large enough to matter for 46 features and ~4,900 training "
        "rows. Compare training loss, validation loss and all four "
        "classification metrics."
    ),
    "16": (
        "## Section 16 — Ridge and Lasso (classical linear models)\n\n"
        "**Ridge** = linear model + L2 penalty, **Lasso** = linear "
        "model + L1 penalty. These are *not* the same implementation as "
        "neural-network L1/L2 penalties — Ridge/Lasso apply to the "
        "coefficients of a linear model, where Lasso has the special "
        "property of zeroing coefficients out completely. The shared "
        "idea is the same family: punish large weights.\n\n"
        "We also inspect which features Lasso drives to zero and check "
        "that against business intuition (e.g. `gender` should not "
        "matter for churn)."
    ),
    "17": (
        "## Section 17 — Graphs\n\n"
        "The required graphs: (1) training vs validation loss with the "
        "early-stopping point marked, (2) training vs validation "
        "accuracy, (3) 5-fold CV F1-scores with mean ± std, (4) a "
        "labelled confusion matrix, (5) regularization comparison — "
        "plus a precision/recall/F1 comparison, a feature-importance "
        "plot and the bias/variance curves."
    ),
    "18": (
        "## Section 18 — Final comparison table\n\n"
        "All models evaluated on the **held-out test set only**, using "
        "numbers actually computed by this notebook."
    ),
    "19": (
        "## Section 19 — Conclusions\n\n"
        "Answers to the 14 analysis questions, generated from the real "
        "experimental results stored in `results`."
    ),
}

with open(SRC) as f:
    src = f.read()

# Locate divider blocks:  # ----... / # TITLE / # ----...
divider = re.compile(r"^# -{10,}\n# (.+?)\n# -{10,}\n", re.M)
matches = list(divider.finditer(src))

preamble = src[: matches[0].start()]  # file header comment block
chunks = []
for i, m in enumerate(matches):
    title = m.group(1).strip()
    end = matches[i + 1].start() if i + 1 < len(matches) else len(src)
    code = src[m.end():end].strip("\n")
    chunks.append((title, code))

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                             "name": "python3"}
cells = [nbf.v4.new_markdown_cell(EXPLAIN["title"])]

savefig_re = re.compile(r'savefig\(os\.path\.join\(FIG_DIR,\s*"([^"]+)"')

def section_key(title):
    m = re.search(r"SECTION\s+([0-9/]+)", title)
    return m.group(1) if m else None

for title, code in chunks:
    key = section_key(title)
    md = EXPLAIN.get(key, f"## {title}")
    cells.append(nbf.v4.new_markdown_cell(md))
    cells.append(nbf.v4.new_code_cell(code))
    # display every figure this cell saves
    for png in savefig_re.findall(code):
        cells.append(nbf.v4.new_code_cell(
            "from IPython.display import Image\n"
            f"Image(filename='figures/{png}')"
        ))

nb.cells = cells
nbf.write(nb, OUT)
print(f"Wrote {OUT} with {len(cells)} cells "
      f"({sum(1 for c in cells if c.cell_type == 'code')} code, "
      f"{sum(1 for c in cells if c.cell_type == 'markdown')} markdown).")