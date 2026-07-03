"""
Credit Scoring Model — CodeAlpha ML Internship (Task 1)
Author: Imran Ali Memon
Description: Predict creditworthiness using classification algorithms
             on financial data with Logistic Regression, Decision Trees,
             Random Forest, and XGBoost.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve, f1_score
)
import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")
RANDOM_STATE = 42
FIGURE_DIR = "figures"

import os
os.makedirs(FIGURE_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# 1. GENERATE CREDIT SCORING DATASET
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("CREDIT SCORING MODEL")
print("=" * 60)

np.random.seed(RANDOM_STATE)
n = 1000

age = np.random.randint(21, 65, n)
income = np.random.lognormal(10.5, 0.5, n).astype(int)
num_credit_cards = np.random.poisson(2, n) + 1
num_loans = np.random.poisson(1.5, n)
credit_history_months = np.random.randint(6, 300, n)
debt_to_income = np.random.uniform(0.05, 0.85, n).round(2)
num_late_payments = np.random.poisson(1.2, n)
employment_years = np.random.randint(0, 40, n)
has_mortgage = np.random.choice([0, 1], n, p=[0.6, 0.4])
monthly_balance = np.random.normal(5000, 3000, n).clip(0).astype(int)

# Credit score target (Good=1, Bad=0)
score = (
    0.3 * (income / income.max())
    + 0.2 * (credit_history_months / 300)
    - 0.25 * debt_to_income
    - 0.15 * (num_late_payments / num_late_payments.max())
    + 0.1 * (employment_years / 40)
    + np.random.normal(0, 0.1, n)
)
creditworthy = (score > np.median(score)).astype(int)

df = pd.DataFrame({
    "Age": age, "Income": income, "Num_Credit_Cards": num_credit_cards,
    "Num_Loans": num_loans, "Credit_History_Months": credit_history_months,
    "Debt_To_Income": debt_to_income, "Num_Late_Payments": num_late_payments,
    "Employment_Years": employment_years, "Has_Mortgage": has_mortgage,
    "Monthly_Balance": monthly_balance, "Creditworthy": creditworthy,
})

df.to_csv("credit_scoring_dataset.csv", index=False)

print(f"\n📊 Dataset: {df.shape}")
print(f"\n🔍 First 5 Rows:")
print(df.head())
print(f"\n🏷️ Target Distribution:")
print(df["Creditworthy"].value_counts().rename({0: "Bad Credit", 1: "Good Credit"}))
print(f"\n📈 Summary:")
print(df.describe().round(2))

# ═══════════════════════════════════════════════════════════
# 2. EDA
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# 2a. Target Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df["Creditworthy"].value_counts().plot(kind="pie", ax=axes[0],
    labels=["Bad Credit", "Good Credit"], autopct="%1.1f%%",
    colors=["#e74c3c", "#2ecc71"], startangle=90)
axes[0].set_title("Credit Score Distribution", fontweight="bold")
axes[0].set_ylabel("")
sns.countplot(x="Creditworthy", data=df, ax=axes[1], palette=["#e74c3c", "#2ecc71"])
axes[1].set_xticklabels(["Bad Credit", "Good Credit"])
axes[1].set_title("Class Balance", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/01_target_distribution.png", dpi=150)
plt.close()
print("✅ Saved: 01_target_distribution.png")

# 2b. Feature Distribution by Target
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
features = ["Income", "Debt_To_Income", "Credit_History_Months",
            "Num_Late_Payments", "Employment_Years", "Monthly_Balance"]
for i, feat in enumerate(features):
    ax = axes[i // 3, i % 3]
    for label, color in zip([0, 1], ["#e74c3c", "#2ecc71"]):
        ax.hist(df[df["Creditworthy"] == label][feat], bins=25, alpha=0.6,
                color=color, label="Bad" if label == 0 else "Good", edgecolor="white")
    ax.set_title(feat, fontweight="bold")
    ax.legend()
plt.suptitle("Feature Distribution by Credit Score", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/02_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: 02_feature_distributions.png")

# 2c. Correlation
plt.figure(figsize=(12, 10))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, cmap="RdYlBu_r", fmt=".2f",
            linewidths=0.5, square=True)
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/03_correlation_heatmap.png", dpi=150)
plt.close()
print("✅ Saved: 03_correlation_heatmap.png")

# ═══════════════════════════════════════════════════════════
# 3. PREPROCESSING
# ═══════════════════════════════════════════════════════════
X = df.drop("Creditworthy", axis=1)
y = df["Creditworthy"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTraining: {X_train.shape[0]} | Testing: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ═══════════════════════════════════════════════════════════
# 4. MODEL TRAINING
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL TRAINING & EVALUATION")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
}

results = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cv = cross_val_score(model, X_train_s, y_train, cv=5, scoring="roc_auc")
    
    results[name] = {"acc": acc, "f1": f1, "auc": auc,
                     "cv_mean": cv.mean(), "y_pred": y_pred, "y_prob": y_prob}
    
    print(f"\n{'─' * 50}")
    print(f"📌 {name}")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   ROC-AUC:   {auc:.4f}")
    print(f"   CV AUC:    {cv.mean():.4f} ± {cv.std():.4f}")

# ═══════════════════════════════════════════════════════════
# 5. VISUALIZATIONS
# ═══════════════════════════════════════════════════════════

# 5a. ROC Curves
plt.figure(figsize=(8, 8))
for name, data in results.items():
    fpr, tpr, _ = roc_curve(y_test, data["y_prob"])
    plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={data['auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curves — All Models", fontsize=14, fontweight="bold")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/04_roc_curves.png", dpi=150)
plt.close()
print("\n✅ Saved: 04_roc_curves.png")

# 5b. Metrics Comparison
model_names = list(results.keys())
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metrics = [("acc", "Accuracy"), ("f1", "F1-Score"), ("auc", "ROC-AUC")]
colors = ["#2196F3", "#4CAF50", "#FF9800", "#e74c3c"]
for i, (key, label) in enumerate(metrics):
    vals = [results[m][key] for m in model_names]
    bars = axes[i].barh(model_names, vals, color=colors)
    axes[i].set_title(label, fontweight="bold")
    axes[i].set_xlim(0.5, 1.0)
    for bar, val in zip(bars, vals):
        axes[i].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontsize=9)
plt.suptitle("Model Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/05_metrics_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: 05_metrics_comparison.png")

# 5c. Confusion Matrix (Best Model)
best_name = max(results, key=lambda k: results[k]["auc"])
best_pred = results[best_name]["y_pred"]

plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
plt.title(f"Confusion Matrix — {best_name}", fontweight="bold")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/06_confusion_matrix.png", dpi=150)
plt.close()
print("✅ Saved: 06_confusion_matrix.png")

# 5d. Feature Importance
if hasattr(models[best_name], "feature_importances_"):
    imp = pd.Series(models[best_name].feature_importances_, index=X.columns).sort_values()
    plt.figure(figsize=(8, 6))
    imp.plot(kind="barh", color="#3498db")
    plt.title(f"Feature Importance — {best_name}", fontweight="bold")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/07_feature_importance.png", dpi=150)
    plt.close()
    print("✅ Saved: 07_feature_importance.png")

print(f"\n🏆 Best Model: {best_name}")
print(f"\n📋 Classification Report:")
print(classification_report(y_test, best_pred, target_names=["Bad Credit", "Good Credit"]))
print("✅ Credit Scoring Model Complete!")
