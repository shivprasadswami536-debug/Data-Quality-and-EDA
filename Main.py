import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load Dataset
df = pd.read_csv("data/sample_sales_data.csv")

# Create Output Folder
os.makedirs("outputs", exist_ok=True)

# 1. Display Dataset
print("First Five Rows:")
print(df.head())

# 2. Dataset Information
print("\nDataset Information:")
df.info()

# 3. Dataset Shape
print("\nDataset Shape:")
print(df.shape)

# 4. Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# 5. Duplicate Records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# 6. Descriptive Statistics
print("\nDescriptive Statistics:")
print(df.describe(include="all"))

# 7. Missing Value Percentage
print("\nMissing Value Percentage:")
print((df.isnull().mean() * 100).round(2))

# 8. Outlier Detection Using IQR
print("\nPotential Outliers:")

numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower) |
        (df[column] > upper)
    ]

    print(column, ":", len(outliers))

# 9. Remove Duplicate Rows
cleaned_df = df.drop_duplicates()

cleaned_df.to_csv(
    "outputs/cleaned_data.csv",
    index=False
)

# 10. Missing Values Visualization
missing = df.isnull().sum()
missing = missing[missing > 0]

if not missing.empty:
    missing.plot(kind="bar")
    plt.title("Missing Values by Column")
    plt.xlabel("Columns")
    plt.ylabel("Missing Count")
    plt.tight_layout()
    plt.savefig("outputs/missing_values.png")
    plt.close()

# 11. Numerical Data Distribution
if len(numeric_columns) > 0:
    df[numeric_columns].hist(
        figsize=(12, 8),
        bins=15
    )

    plt.suptitle("Numerical Data Distribution")
    plt.tight_layout()
    plt.savefig("outputs/numeric_distributions.png")
    plt.close()

# 12. Correlation Heatmap
if len(numeric_columns) >= 2:
    plt.figure(figsize=(9, 6))

    sns.heatmap(
        df[numeric_columns].corr(),
        annot=True,
        cmap="coolwarm"
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("outputs/correlation_heatmap.png")
    plt.close()

# 13. Save Data Quality Report
with open(
    "outputs/data_quality_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write("DATA QUALITY AND EDA REPORT\n")
    file.write("===========================\n\n")

    file.write(f"Rows: {df.shape[0]}\n")
    file.write(f"Columns: {df.shape[1]}\n")
    file.write(
        f"Duplicate Rows: {df.duplicated().sum()}\n\n"
    )

    file.write("Missing Values:\n")
    file.write(df.isnull().sum().to_string())

    file.write("\n\nDescriptive Statistics:\n")
    file.write(
        df.describe(include="all").to_string()
    )

print("\nAnalysis Completed Successfully!")
print("Check the outputs folder for results.")
