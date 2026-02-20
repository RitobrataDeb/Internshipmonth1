import pandas as pd
import numpy as np

np.random.seed(42)

raw_data = {
    "employee_id": [101, 102, 103, 104, 105, 106, 107, 102, 108, 109, 110],
    "name":        ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", None,
                    "Bob", "Heidi", "Ivan", "Judy"],
    "department":  ["Engineering", "Marketing", "Engineering", "HR",
                    "Marketing", "Engineering", "HR", "Marketing",
                    "Engineering", None, "HR"],
    "salary":      [95000, 72000, 88000, 61000, 74000, 99000, 58000,
                    72000, 85000, 67000, None],
    "age":         [30, 45, 38, 29, 52, 41, 35, 45, 33, 28, 400],
    "years_exp":   [5, 20, 12, 4, 25, 15, 8, 20, 9, 3, None],
    "rating":      [4.5, 3.8, 4.2, None, 4.0, 4.7, 3.5, 3.8, 4.1, 3.9, 4.3],
}

df = pd.DataFrame(raw_data)

print("=" * 60)
print("STEP 1 - RAW DATA")
print("=" * 60)
print(df.to_string(index=False))
print("\nShape: {} rows x {} cols".format(df.shape[0], df.shape[1]))
print("\nMissing values per column:")
print(df.isnull().sum().to_string())

print("\n" + "=" * 60)
print("STEP 2 - REMOVE DUPLICATES")
print("=" * 60)

before = len(df)
df = df.drop_duplicates(subset="employee_id", keep="first")
print("Removed {} duplicate row(s). Rows remaining: {}".format(before - len(df), len(df)))

print("\n" + "=" * 60)
print("STEP 3 - HANDLE MISSING VALUES")
print("=" * 60)

before = len(df)
df = df.dropna(subset=["name", "department"])
print("Dropped {} row(s) with missing name or department.".format(before - len(df)))

for col in ["salary", "years_exp", "rating"]:
    median_val = df[col].median()
    missing = df[col].isnull().sum()
    df[col] = df[col].fillna(median_val)
    print("  {}: filled {} missing value(s) with median ({:.2f})".format(col, missing, median_val))

print("\n" + "=" * 60)
print("STEP 4 - FIX DATA TYPES")
print("=" * 60)

df["salary"]    = df["salary"].astype(int)
df["years_exp"] = df["years_exp"].astype(int)
df["rating"]    = df["rating"].round(2)
print("Converted salary to int, years_exp to int, rating to float (2 dp)")

print("\n" + "=" * 60)
print("STEP 5 - REMOVE OUTLIERS (IQR method on age)")
print("=" * 60)

Q1  = df["age"].quantile(0.25)
Q3  = df["age"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

before = len(df)
df = df[df["age"].between(lower, upper)]
print("Age bounds: [{:.1f}, {:.1f}]".format(lower, upper))
print("Removed {} outlier row(s). Rows remaining: {}".format(before - len(df), len(df)))

print("\n" + "=" * 60)
print("STEP 6 - CLEANED DATASET")
print("=" * 60)
print(df.to_string(index=False))

print("\n" + "=" * 60)
print("STEP 7 - AGGREGATIONS BY DEPARTMENT")
print("=" * 60)

dept_stats = df.groupby("department").agg(
    headcount=("employee_id", "count"),
    avg_salary=("salary", "mean"),
    max_salary=("salary", "max"),
    min_salary=("salary", "min"),
    avg_age=("age", "mean"),
    avg_exp_yrs=("years_exp", "mean"),
    avg_rating=("rating", "mean"),
)
dept_stats = dept_stats.round(2).sort_values("avg_salary", ascending=False).reset_index()
print(dept_stats.to_string(index=False))

print("\n" + "=" * 60)
print("STEP 8 - COMPANY-WIDE SUMMARY STATISTICS")
print("=" * 60)

summary = df[["salary", "age", "years_exp", "rating"]].describe().round(2)
print(summary.to_string())

print("\n" + "=" * 60)
print("STEP 9 - TOP 3 EARNERS")
print("=" * 60)

top3 = df.nlargest(3, "salary")[["name", "department", "salary", "rating"]]
print(top3.to_string(index=False))

print("\nPipeline complete.\n")
