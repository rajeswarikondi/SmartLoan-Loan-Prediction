"""
Exploratory data analysis charts (Matplotlib + Seaborn) for the project report.

Creates PNG images in the folder  eda_output/  :
  missing_values, loan_status_pie, property_area_bar, income_histogram,
  loan_amount_histogram, box_plots, income_band_line, income_vs_loan_scatter,
  credit_history, correlation_heatmap

Run:   python eda_report.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")                      # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "eda_output"
OUT_DIR.mkdir(exist_ok=True)

COLORS = {"Approved": "#1f4e79", "Rejected": "#c55a11"}
CATEGORICAL = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
NUMERICAL = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]

raw = pd.read_csv(BASE_DIR / "loan-train.csv")

# Same cleaning rules as clean_data() in app.py
data = raw.drop_duplicates().copy()
for column in CATEGORICAL:
    data[column] = data[column].fillna(data[column].mode()[0])
for column in NUMERICAL:
    data[column] = data[column].fillna(data[column].median())
data["Status"] = data["Loan_Status"].map({"Y": "Approved", "N": "Rejected"})

sns.set_style("whitegrid")


def save(name):
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}.png", dpi=200)
    plt.close()
    print("saved", OUT_DIR / f"{name}.png")


# 1. Missing values of the raw data
missing = raw.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
plt.figure(figsize=(7, 4))
ax = sns.barplot(x=missing.index, y=missing.values, color=COLORS["Approved"])
ax.bar_label(ax.containers[0])
plt.xticks(rotation=25, ha="right")
plt.ylabel("Missing values")
plt.title("Missing Values per Column (Raw Dataset)")
save("missing_values")

# 2. Loan status pie chart
counts = data["Status"].value_counts()
plt.figure(figsize=(5, 4))
plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90,
        colors=[COLORS[name] for name in counts.index], wedgeprops={"width": 0.45, "edgecolor": "white"})
plt.title("Loan Status Distribution")
save("loan_status_pie")

# 3. Applications by property area
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=data, x="Property_Area", color=COLORS["Approved"])
ax.bar_label(ax.containers[0])
plt.title("Applications by Property Area")
save("property_area_bar")

# 4 and 5. Histograms
for column, limit, name in [("ApplicantIncome", 20000, "income_histogram"), ("LoanAmount", None, "loan_amount_histogram")]:
    plt.figure(figsize=(7, 4))
    shown = data if limit is None else data[data[column] <= limit]
    sns.histplot(data=shown, x=column, hue="Status", bins=40, palette=COLORS, alpha=0.75)
    plt.title(f"{column} Distribution by Loan Status")
    save(name)

# 6. Box plots
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
for axis, column in zip(axes, ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]):
    sns.boxplot(data=data, x="Status", y=column, order=["Approved", "Rejected"],
                hue="Status", palette=COLORS, legend=False, ax=axis)
    axis.set_title(column)
    axis.set_xlabel("")
    axis.set_ylabel("")
save("box_plots")

# 7. Approval rate across income bands (line chart)
bands = pd.qcut(data["ApplicantIncome"], q=6, duplicates="drop")
rate = data.groupby(bands, observed=True)["Loan_Status"].apply(lambda s: (s == "Y").mean() * 100)
labels = [f"{int(interval.left)}-{int(interval.right)}" for interval in rate.index]
plt.figure(figsize=(7, 4))
plt.plot(labels, rate.values, marker="o", color=COLORS["Approved"], linewidth=2)
for position, value in enumerate(rate.values):
    plt.text(position, value + 1, f"{value:.1f}%", ha="center")
plt.xticks(rotation=20, ha="right")
plt.ylim(0, 100)
plt.ylabel("Approval rate (%)")
plt.xlabel("Applicant income band")
plt.title("Approval Rate Across Applicant Income Bands")
save("income_band_line")

# 8. Scatter plot
plt.figure(figsize=(7, 4.5))
sns.scatterplot(data=data, x="ApplicantIncome", y="LoanAmount", hue="Status", palette=COLORS, s=25)
plt.xlim(0, 25000)
plt.title("Applicant Income vs Loan Amount")
save("income_vs_loan_scatter")

# 9. Credit history
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
sns.countplot(data=data, x="Credit_History", hue="Status", palette=COLORS, ax=axes[0])
axes[0].set_title("Credit History vs Loan Status")
credit_rate = data.groupby("Credit_History")["Loan_Status"].apply(lambda s: (s == "Y").mean() * 100)
sns.barplot(x=credit_rate.index.astype(int).astype(str), y=credit_rate.values, color=COLORS["Approved"], ax=axes[1])
axes[1].bar_label(axes[1].containers[0], fmt="%.1f%%")
axes[1].set_title("Approval Rate by Credit History")
axes[1].set_ylim(0, 100)
save("credit_history")

# 10. Correlation heatmap
corr_data = data[["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]].copy()
corr_data["Loan_Status (Y=1)"] = (data["Loan_Status"] == "Y").astype(int)
plt.figure(figsize=(7, 5.5))
sns.heatmap(corr_data.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1)
plt.xticks(rotation=35, ha="right")
plt.title("Correlation Heatmap of Numerical Variables")
save("correlation_heatmap")

print("\nDone. All images are in:", OUT_DIR)
