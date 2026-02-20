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

df = df.drop_duplicates(subset="employee_id", keep="first")


df = df.dropna(subset=["name", "department"])

for col in ["salary", "years_exp", "rating"]:
    df[col] = df[col].fillna(df[col].median())

df["salary"]    = df["salary"].astype(int)
df["years_exp"] = df["years_exp"].astype(int)
df["rating"]    = df["rating"].round(2)

Q1, Q3 = df["age"].quantile(0.25), df["age"].quantile(0.75)
IQR    = Q3 - Q1
df     = df[df["age"].between(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)]

dept_stats = df.groupby("department").agg(
    headcount   = ("employee_id", "count"),
    avg_salary  = ("salary",      "mean"),
    max_salary  = ("salary",      "max"),
    min_salary  = ("salary",      "min"),
    avg_age     = ("age",         "mean"),
    avg_exp_yrs = ("years_exp",   "mean"),
    avg_rating  = ("rating",      "mean"),
).round(2).reset_index()

summary = df[["salary", "age", "years_exp", "rating"]].describe().round(2)

top_earners = df.nlargest(3, "salary")[["name", "department", "salary", "rating"]]

print("Cleaned Dataset:")
print(df.to_string(index=False))

print("\nDepartment Aggregations:")
print(dept_stats.to_string(index=False))

print("\nSummary Statistics:")
print(summary.to_string())

print("\nTop 3 Earners:")
print(top_earners.to_string(index=False))
