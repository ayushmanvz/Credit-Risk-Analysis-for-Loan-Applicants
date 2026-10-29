"""
credit_risk_analysis.py
-----------------------
Project : Credit Risk Analysis for Loan Applicants
Group 6 : Ayushman Barik (1ETBTD20251011), Ashwin Kumar Nayak (1ETBTD20251015)
College : Sri Sri University

What this script does
  1. Loads loan_applicants.csv
  2. Explores the data (default rate by credit score, income, employment, ...)
  3. Trains a Logistic Regression model to predict loan default
  4. Evaluates it (accuracy, precision, recall, ROC-AUC)
  5. Classifies every applicant as Low / Medium / High risk
  6. Saves scored_applicants.csv and charts in the 'charts' folder

Run:  python credit_risk_analysis.py [path_to_csv]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CSV = sys.argv[1] if len(sys.argv) > 1 else "loan_applicants.csv"
os.makedirs("charts", exist_ok=True)

# ---------------------------------------------------------------- 1. load
df = pd.read_csv(CSV)
print(f"Loaded {len(df)} applicants, {df.shape[1]} columns")
print("Missing values:", int(df.isna().sum().sum()))
df = df.dropna()
df["loan_to_income"] = df["loan_amount"] / df["annual_income"]

# ---------------------------------------------------------------- 2. explore
print(f"\nOverall default rate: {df['loan_status'].mean():.1%}")

df["credit_band"] = pd.cut(df["credit_score"], [0, 580, 650, 720, 900],
                           labels=["Poor <580", "Fair 580-649", "Good 650-719", "Excellent 720+"])
df["income_band"] = pd.qcut(df["annual_income"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])

for col in ["credit_band", "income_band", "employment_status", "existing_loans", "missed_payments"]:
    rate = df.groupby(col, observed=True)["loan_status"].mean().mul(100).round(1)
    print(f"\nDefault rate (%) by {col}:\n{rate.to_string()}")
    ax = rate.plot(kind="bar", color="#1f6feb", figsize=(6, 4))
    ax.set_ylabel("Default rate (%)")
    ax.set_title(f"Default rate by {col}")
    plt.tight_layout()
    plt.savefig(f"charts/default_by_{col}.png", dpi=120)
    plt.close()

numeric_cols = ["age", "annual_income", "loan_amount", "loan_term_months", "interest_rate",
                "credit_score", "credit_history_years", "existing_loans",
                "debt_to_income", "missed_payments", "loan_to_income"]
corr = df[numeric_cols + ["loan_status"]].corr()["loan_status"].drop("loan_status").sort_values()
print("\nCorrelation of each factor with default:\n", corr.round(3).to_string())
corr.plot(kind="barh", color="#d1495b", figsize=(6, 5), title="Correlation with default")
plt.tight_layout()
plt.savefig("charts/correlation_with_default.png", dpi=120)
plt.close()

# ---------------------------------------------------------------- 3. model
cat_cols = ["employment_status", "education", "loan_purpose"]
X = df[numeric_cols + cat_cols]
y = df["loan_status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pre = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])
model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
model.fit(X_train, y_train)

# ---------------------------------------------------------------- 4. evaluate
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
print("\n===== Model performance (test set) =====")
print(f"Accuracy : {accuracy_score(y_test, pred):.3f}")
print(f"Precision: {precision_score(y_test, pred):.3f}")
print(f"Recall   : {recall_score(y_test, pred):.3f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, proba):.3f}")
print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=["Repaid", "Default"]))

names = model.named_steps["pre"].get_feature_names_out()
coefs = pd.Series(model.named_steps["clf"].coef_[0], index=names).sort_values()
print("Top risk-increasing factors:\n", coefs.tail(5).round(3).to_string())
print("Top risk-reducing factors:\n", coefs.head(5).round(3).to_string())
coefs.plot(kind="barh", figsize=(7, 8), color="#1f6feb", title="Model coefficients (+ = higher default risk)")
plt.tight_layout()
plt.savefig("charts/model_coefficients.png", dpi=120)
plt.close()

# ---------------------------------------------------------------- 5. score everyone
df["default_probability"] = model.predict_proba(X)[:, 1].round(4)
df["risk_level"] = pd.cut(df["default_probability"], [-0.01, 0.30, 0.60, 1.01],
                          labels=["Low", "Medium", "High"])
print("\nRisk level counts:\n", df["risk_level"].value_counts().to_string())
print("Actual default rate per risk level (%):\n",
      df.groupby("risk_level", observed=True)["loan_status"].mean().mul(100).round(1).to_string())

df.drop(columns=["credit_band", "income_band"]).to_csv("scored_applicants.csv", index=False)
print("\nSaved scored_applicants.csv and charts/ folder.")
