"""
# Wine Quality Classification — End-to-End Project

**Author:** _Your Name_

**Dataset:** `winequality-red.csv`

**Pipeline:** EDA → Baselines → Feature Engineering → Re-evaluation → Final Report

> Tip: Run cells top-to-bottom. All plots use matplotlib only (no seaborn),
and code is organized to be easy to grade and replicate.

"""

"""
## 0. Setup & Data Loading
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# For consistent plots
plt.rcParams['figure.figsize'] = (6,4)
plt.rcParams['axes.grid'] = False

# Load data
path = "/mnt/data/winequality-red.csv"
df = pd.read_csv(path, sep=',')
if df.shape[1] == 1:
    df = pd.read_csv(path, sep=';')  # common for wine dataset

print(df.shape)
df.head()


"""
## 1. EDA — Exploratory Data Analysis
"""


# Basic info
display(df.info())
display(df.describe())

# Target distribution
quality_counts = df['quality'].value_counts().sort_index()
plt.figure()
plt.bar(quality_counts.index.astype(str), quality_counts.values)
plt.title("Quality Label Distribution")
plt.xlabel("Quality")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

num_cols = [c for c in df.columns if c != 'quality']

# Histograms
for col in num_cols:
    plt.figure()
    df[col].hist(bins=30)
    plt.title(f"Histogram: {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# Boxplots by quality
qualities = sorted(df['quality'].unique())
for col in num_cols:
    plt.figure()
    data_by_q = [df[df['quality']==q][col].values for q in qualities]
    plt.boxplot(data_by_q, labels=qualities)
    plt.title(f"Boxplot of {col} by Quality")
    plt.xlabel("Quality")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

# Correlation heatmap
corr = df[num_cols + ['quality']].corr()
plt.figure(figsize=(8,6))
im = plt.imshow(corr.values, aspect='auto')
plt.xticks(ticks=np.arange(len(corr.columns)), labels=corr.columns, rotation=90)
plt.yticks(ticks=np.arange(len(corr.index)), labels=corr.index)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.title("Correlation Heatmap (features + quality)")
plt.tight_layout()
plt.show()

# Skewness/Kurtosis
skewness = df[num_cols].skew()
kurtosis = df[num_cols].kurt()
sk_table = pd.DataFrame({'skewness': skewness, 'kurtosis': kurtosis}).sort_values('skewness', ascending=False)
sk_table


"""
### 1.1 EDA Summary (Write-up)
"""

"""

- **Target imbalance:** Mid-quality classes dominate; extremes are rare → use macro-F1.
- **Notable associations:** Alcohol (+), volatile acidity (−), sulphates (+), density (−), citric acid (+) vs quality.
- **Skewness:** Chlorides, sulphates, residual sugar, and SO₂ variables are right-skewed → consider power transforms.
- **Collinearity:** alcohol ↔ density; free ↔ total SO₂; linear models may need scaling/regularization.

"""

"""
## 2. Baseline Models
"""


X = df.drop(columns=['quality']).copy()
y = df['quality'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Baseline 1: Multinomial Logistic Regression (scaled)
logreg_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000, multi_class='multinomial'))
])
logreg_pipe.fit(X_train, y_train)
y_pred_lr = logreg_pipe.predict(X_test)

# Baseline 2: Random Forest
rf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

from collections import OrderedDict
baseline_metrics = OrderedDict({
    'LR_accuracy': accuracy_score(y_test, y_pred_lr),
    'LR_macroF1': f1_score(y_test, y_pred_lr, average='macro'),
    'RF_accuracy': accuracy_score(y_test, y_pred_rf),
    'RF_macroF1': f1_score(y_test, y_pred_rf, average='macro'),
})
baseline_metrics



# Confusion matrices (LR and RF)
def plot_cm(cm, labels, title):
    plt.figure(figsize=(6,5))
    im = plt.imshow(cm, interpolation='nearest', aspect='auto')
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

labels = sorted(y.unique())
cm_lr = confusion_matrix(y_test, y_pred_lr, labels=labels)
cm_rf = confusion_matrix(y_test, y_pred_rf, labels=labels)
plot_cm(cm_lr, labels, "Confusion Matrix — Logistic Regression (Baseline)")
plot_cm(cm_rf, labels, "Confusion Matrix — Random Forest (Baseline)")

# RF feature importances
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
importances


"""
## 3. Feature Engineering — Selection & Methods
"""

"""

**Selected features (from EDA + RF importances):** alcohol, volatile acidity, sulphates, density, citric acid, residual sugar, free/total SO₂.

**Method A (Variance Stabilization):** Yeo–Johnson power transform on skewed variables (|skew| > 1).

**Method B (Domain Features):**

- `sulfur_ratio = free SO₂ / total SO₂`

- `total_acidity = fixed + volatile + citric`

- `acid_sugar_balance = total_acidity / (residual sugar + 1)`

- `alcohol_density = alcohol / density`

Rationales: sensory balance (acid vs sugar), oxidation protection (SO₂ chemistry), disentangling alcohol vs density.

We'll show visuals/statistics via importances, confusion matrices, and metric deltas.

"""


def add_domain_features(frame):
    f = frame.copy()
    eps = 1e-6
    if 'free sulfur dioxide' in f.columns and 'total sulfur dioxide' in f.columns:
        f['sulfur_ratio'] = f['free sulfur dioxide'] / (f['total sulfur dioxide'] + eps)
    if set(['fixed acidity','volatile acidity','citric acid']).issubset(f.columns):
        f['total_acidity'] = f['fixed acidity'] + f['volatile acidity'] + f['citric acid']
        if 'residual sugar' in f.columns:
            f['acid_sugar_balance'] = f['total_acidity'] / (f['residual sugar'] + 1.0)
    if 'alcohol' in f.columns and 'density' in f.columns:
        f['alcohol_density'] = f['alcohol'] / (f['density'] + eps)
    return f

X_eng = add_domain_features(X)
engineered_new = [c for c in X_eng.columns if c not in X.columns]

# Identify skewed
num_cols = X.columns.tolist()
skewness = df[num_cols].skew()
skewed_feats = skewness[skewness.abs() > 1].index.tolist()

# ColumnTransformer: Yeo-Johnson for skewed features only
pt = ColumnTransformer(
    transformers=[('yeojohnson', PowerTransformer(method='yeo-johnson'), skewed_feats)],
    remainder='passthrough'
)

# Train/test split (same as before for fairness)
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    X_eng, y, test_size=0.2, random_state=42, stratify=y
)

# Fit transform on train; transform test
X_train_e_t = pt.fit_transform(X_train_e)
X_test_e_t = pt.transform(X_test_e)

# Rebuild column order for readability (skewed first, passthrough after)
yeo_cols = skewed_feats
passthrough_cols = [c for c in X_eng.columns if c not in yeo_cols]
eng_cols = yeo_cols + passthrough_cols
X_train_e_t = pd.DataFrame(X_train_e_t, columns=eng_cols, index=X_train_e.index)
X_test_e_t = pd.DataFrame(X_test_e_t, columns=eng_cols, index=X_test_e.index)

engineered_new, skewed_feats[:10], X_train_e_t.head()


"""
## 4. Re-Training with Engineered Features
"""


# Logistic Regression: scale AFTER Yeo–Johnson
scaler = StandardScaler()
X_train_lr = scaler.fit_transform(X_train_e_t)
X_test_lr = scaler.transform(X_test_e_t)

lr_eng = LogisticRegression(max_iter=2000, multi_class='multinomial')
lr_eng.fit(X_train_lr, y_train_e)
y_pred_lr_e = lr_eng.predict(X_test_lr)

# Random Forest: trees work fine on unscaled engineered features
rf_eng = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
rf_eng.fit(X_train_e_t, y_train_e)
y_pred_rf_e = rf_eng.predict(X_test_e_t)

from collections import OrderedDict
eng_metrics = OrderedDict({
    'LR_eng_accuracy': accuracy_score(y_test_e, y_pred_lr_e),
    'LR_eng_macroF1': f1_score(y_test_e, y_pred_lr_e, average='macro'),
    'RF_eng_accuracy': accuracy_score(y_test_e, y_pred_rf_e),
    'RF_eng_macroF1': f1_score(y_test_e, y_pred_rf_e, average='macro'),
})
eng_metrics



# Confusion matrices (engineered)
labels = sorted(y.unique())
cm_lr_e = confusion_matrix(y_test_e, y_pred_lr_e, labels=labels)
cm_rf_e = confusion_matrix(y_test_e, y_pred_rf_e, labels=labels)

def plot_cm(cm, labels, title):
    plt.figure(figsize=(6,5))
    im = plt.imshow(cm, interpolation='nearest', aspect='auto')
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

plot_cm(cm_lr_e, labels, "Confusion Matrix — Logistic Regression (Engineered)")
plot_cm(cm_rf_e, labels, "Confusion Matrix — Random Forest (Engineered)")

# RF importances after engineering
rf_eng_importances = pd.Series(rf_eng.feature_importances_, index=eng_cols).sort_values(ascending=False)
rf_eng_importances.head(20)



# Visual comparison: baseline vs engineered
labels_models = [
    "LR Base (acc)","LR Base (macroF1)",
    "RF Base (acc)","RF Base (macroF1)",
    "LR Eng (acc)","LR Eng (macroF1)",
    "RF Eng (acc)","RF Eng (macroF1)"
]
values = [
    baseline_metrics['LR_accuracy'], baseline_metrics['LR_macroF1'],
    baseline_metrics['RF_accuracy'], baseline_metrics['RF_macroF1'],
    eng_metrics['LR_eng_accuracy'], eng_metrics['LR_eng_macroF1'],
    eng_metrics['RF_eng_accuracy'], eng_metrics['RF_eng_macroF1'],
]

plt.figure(figsize=(10,4))
plt.bar(labels_models, values)
plt.xticks(rotation=45, ha='right')
plt.title("Baseline vs Engineered Features: Accuracy & Macro-F1")
plt.ylabel("Score")
plt.tight_layout()
plt.show()

pd.DataFrame({
    'model': ['LogReg Baseline','RandomForest Baseline','LogReg Engineered','RandomForest Engineered'],
    'accuracy': [baseline_metrics['LR_accuracy'], baseline_metrics['RF_accuracy'],
                 eng_metrics['LR_eng_accuracy'], eng_metrics['RF_eng_accuracy']],
    'macro_f1': [baseline_metrics['LR_macroF1'], baseline_metrics['RF_macroF1'],
                 eng_metrics['LR_eng_macroF1'], eng_metrics['RF_eng_macroF1']]
}).sort_values('macro_f1', ascending=False)


"""
## 5. Final Summary (Project-Style)
"""

"""

**Key EDA insights.** Alcohol ↑ and density ↓ correlate with higher quality; volatile acidity ↓ in higher-quality wines; sulphates and citric acid trend ↑. Several variables are right-skewed (chlorides, sulphates, residual sugar, SO₂).

**Baselines.** Random forest outperforms logistic regression on both accuracy and macro-F1; LR collapses toward majority classes.

**Feature engineering.** (A) Yeo–Johnson on skewed inputs to stabilize variance (helps linear separability). (B) Domain ratios/interactions (sulfur_ratio, total_acidity, acid_sugar_balance, alcohol_density) to model sensory balance and chemical mechanisms.

**Re-evaluation.** LR gains a bit in macro-F1; RF gains slightly in accuracy (macro-F1 roughly stable). Confusion matrices indicate modest improvements in mid-to-high quality separation.

**Conclusion.** Chemistry-informed features add incremental value. For larger gains: tune class weights/hyperparameters, try gradient boosting (XGBoost/LightGBM), consider quality banding, and use cross-validation with cost-sensitive metrics.


"""

