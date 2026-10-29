"""
generate_dataset.py
-------------------
Creates loan_applicants.csv - a synthetic (practice) dataset of loan applicants
for the project "Credit Risk Analysis for Loan Applicants".

Columns
  applicant_id, age, gender, education, employment_status, annual_income,
  loan_amount, loan_term_months, interest_rate, credit_score,
  credit_history_years, existing_loans, debt_to_income, missed_payments,
  loan_purpose, loan_status   (1 = defaulted, 0 = repaid)

Note: the data is generated, not real bank data. The default probability is
built from realistic rules (low credit score, high debt, missed payments and
unemployment raise risk) so the analysis shows sensible patterns.

Run:  python generate_dataset.py
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1500

age = rng.integers(21, 66, N)
gender = rng.choice(["Male", "Female", "Other"], N, p=[0.52, 0.47, 0.01])
education = rng.choice(["High School", "Graduate", "Post Graduate", "Doctorate"], N, p=[0.25, 0.45, 0.26, 0.04])
employment = rng.choice(["Salaried", "Self-Employed", "Business Owner", "Unemployed"], N, p=[0.55, 0.2, 0.15, 0.10])

# Income depends on employment and education
base = {"Salaried": 600000, "Self-Employed": 700000, "Business Owner": 1100000, "Unemployed": 150000}
edu_mult = {"High School": 0.75, "Graduate": 1.0, "Post Graduate": 1.3, "Doctorate": 1.6}
income = np.array([base[e] * edu_mult[d] for e, d in zip(employment, education)])
income = (income * rng.lognormal(0, 0.35, N)).round(-3).clip(60000, 5000000)

loan_amount = (income * rng.uniform(0.2, 2.2, N)).round(-3).clip(50000, 5000000)
loan_term = rng.choice([12, 24, 36, 48, 60, 84, 120], N, p=[0.08, 0.15, 0.25, 0.2, 0.17, 0.1, 0.05])
credit_history = np.clip((age - 20) * rng.uniform(0.1, 0.7, N), 0, 30).round(1)
existing_loans = rng.poisson(1.3, N).clip(0, 6)
credit_score = np.clip(rng.normal(690, 85, N) + credit_history * 1.2 - existing_loans * 8, 300, 900).round().astype(int)
missed = np.clip(rng.poisson(np.where(credit_score < 600, 2.5, 0.6)), 0, 10)
dti = np.clip(rng.normal(0.30, 0.12, N) + existing_loans * 0.03, 0.02, 0.85).round(2)
interest = np.clip(18 - (credit_score - 300) / 600 * 9 + rng.normal(0, 0.8, N), 7, 24).round(2)
purpose = rng.choice(["Home", "Education", "Vehicle", "Personal", "Business", "Medical"], N, p=[0.2, 0.15, 0.2, 0.25, 0.12, 0.08])

# Hidden risk model -> default probability
lti = loan_amount / income
z = (-1.4
     - 0.012 * (credit_score - 650)
     + 2.2 * (dti - 0.3)
     + 0.35 * missed
     + 0.15 * existing_loans
     + 0.55 * np.clip(lti - 1, 0, None)
     + np.where(employment == "Unemployed", 1.3, 0)
     + np.where(employment == "Self-Employed", 0.2, 0)
     - 0.03 * credit_history
     + 0.12 * (interest - 12))
prob = 1 / (1 + np.exp(-z))
status = (rng.random(N) < prob).astype(int)

df = pd.DataFrame({
    "applicant_id": [f"APP{str(i + 1).zfill(4)}" for i in range(N)],
    "age": age, "gender": gender, "education": education,
    "employment_status": employment, "annual_income": income.astype(int),
    "loan_amount": loan_amount.astype(int), "loan_term_months": loan_term,
    "interest_rate": interest, "credit_score": credit_score,
    "credit_history_years": credit_history, "existing_loans": existing_loans,
    "debt_to_income": dti, "missed_payments": missed,
    "loan_purpose": purpose, "loan_status": status,
})
df.to_csv("loan_applicants.csv", index=False)
print(f"Saved loan_applicants.csv  rows={len(df)}  default rate={status.mean():.1%}")
