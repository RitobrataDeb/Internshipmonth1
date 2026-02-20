
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

PALETTE   = "mako"
ACCENT    = "#4C8EDA"
BG_COLOR  = "#F8F9FA"
TEXT_COLOR = "#1A1A2E"

plt.rcParams.update({
    "figure.facecolor":  BG_COLOR,
    "axes.facecolor":    BG_COLOR,
    "axes.edgecolor":    "#CCCCCC",
    "axes.labelcolor":   TEXT_COLOR,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlepad":     10,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "font.family":       "DejaVu Sans",
    "grid.color":        "#EEEEEE",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.7,
    "legend.framealpha": 0.9,
})

np.random.seed(42)
N = 400

df = pd.DataFrame({
    "age":      np.random.normal(35, 10, N).clip(18, 70).astype(int),
    "salary":   np.random.lognormal(10.8, 0.4, N).astype(int),
    "score":    np.random.beta(5, 2, N) * 100,
    "hours_wk": np.random.normal(40, 8, N).clip(20, 70).astype(int),
    "dept":     np.random.choice(["Engineering", "Marketing", "Sales", "HR", "Finance"], N,
                                 p=[0.35, 0.2, 0.2, 0.1, 0.15]),
    "gender":   np.random.choice(["Male", "Female", "Non-binary"], N, p=[0.48, 0.47, 0.05]),
    "promoted": np.random.choice([0, 1], N, p=[0.7, 0.3]),
})

def save(name: str):
    plt.tight_layout()
    plt.savefig(f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  ✔  Saved: {name}.png")

def plot_overview(df: pd.DataFrame):
    """Quick stats summary grid for all numeric columns."""
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    ncols = 3
    nrows = -(-len(num_cols) // ncols)          # ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    fig.suptitle("Dataset Overview — Distribution of Numeric Features",
                 fontsize=15, fontweight="bold", y=1.01, color=TEXT_COLOR)

    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        ax = axes[i]
        data = df[col].dropna()
        ax.hist(data, bins=30, color=ACCENT, edgecolor="white", linewidth=0.5, alpha=0.85)
        ax.axvline(data.mean(),   color="#E63946", lw=1.8, linestyle="--", label=f"Mean  {data.mean():.1f}")
        ax.axvline(data.median(), color="#2A9D8F", lw=1.8, linestyle=":",  label=f"Median {data.median():.1f}")
        ax.set_title(col)
        ax.legend(fontsize=8)
        ax.grid(axis="y")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    for j in range(i + 1, len(axes)):  
        axes[j].set_visible(False)

    save("01_overview")

def plot_correlation(df: pd.DataFrame):
    num_df = df.select_dtypes(include=np.number)
    corr   = num_df.corr()
    mask   = np.triu(np.ones_like(corr, dtype=bool))  

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.5,
        linecolor="white", square=True, ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
    save("02_correlation_heatmap")

def plot_pairplot(df: pd.DataFrame, hue: str = "dept"):
    num_cols = df.select_dtypes(include=np.number).columns[:4].tolist()  # cap at 4
    g = sns.pairplot(
        df[num_cols + [hue]], hue=hue,
        palette=PALETTE, corner=True,
        plot_kws={"alpha": 0.55, "s": 20},
        diag_kws={"fill": True, "alpha": 0.6}
    )
    g.figure.suptitle("Pair Plot — Numeric Features by Department",
                       y=1.02, fontsize=14, fontweight="bold")
    g.figure.set_facecolor(BG_COLOR)
    g.savefig("03_pairplot.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("  ✔  Saved: 03_pairplot.png")



def plot_box_violin(df: pd.DataFrame, x: str = "dept", y: str = "salary"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    order = df.groupby(x)[y].median().sort_values(ascending=False).index

    sns.boxplot(data=df, x=x, y=y, order=order,
                palette=PALETTE, linewidth=1.2, fliersize=4, ax=ax1)
    ax1.set_title(f"Box Plot — {y} by {x}")
    ax1.set_xlabel(x); ax1.set_ylabel(y)
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(axis="y")

    sns.violinplot(data=df, x=x, y=y, order=order,
                   palette=PALETTE, inner="quartile", linewidth=1, ax=ax2)
    ax2.set_title(f"Violin Plot — {y} by {x}")
    ax2.set_xlabel(x); ax2.set_ylabel("")
    ax2.tick_params(axis="x", rotation=20)
    ax2.grid(axis="y")

    fig.suptitle(f"Distribution of {y} across {x}",
                 fontsize=14, fontweight="bold")
    save("04_box_violin")

def plot_scatter_regression(df: pd.DataFrame,
                             x: str = "age", y: str = "salary",
                             hue: str = "promoted"):
    fig, ax = plt.subplots(figsize=(10, 6))

    palette = {0: "#A8DADC", 1: "#E63946"}
    for val, grp in df.groupby(hue):
        ax.scatter(grp[x], grp[y],
                   color=palette[val], alpha=0.55, s=35, edgecolors="none",
                   label=f"{hue}={val}")

    slope, intercept, r, p, _ = stats.linregress(df[x], df[y])
    xline = np.linspace(df[x].min(), df[x].max(), 200)
    ax.plot(xline, slope * xline + intercept,
            color="#1D3557", lw=2.2, label=f"Regression  r={r:.2f}, p={p:.3f}")

    ax.set_xlabel(x); ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x} with Regression Line")
    ax.legend(); ax.grid(True)
    save("05_scatter_regression")

def plot_bar_chart(df: pd.DataFrame,
                   cat: str = "dept", val: str = "salary"):
    summary = (df.groupby(cat)[val]
                 .agg(["mean", "sem", "count"])
                 .sort_values("mean", ascending=False)
                 .reset_index())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    colors = sns.color_palette(PALETTE, len(summary))

    ax1.bar(summary[cat], summary["mean"], yerr=summary["sem"],
            color=colors, edgecolor="white", linewidth=0.8,
            error_kw={"elinewidth": 1.5, "capsize": 5, "ecolor": "grey"})
    ax1.set_title(f"Mean {val} by {cat}  (± SEM)")
    ax1.set_xlabel(cat); ax1.set_ylabel(f"Mean {val}")
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(axis="y")

    ax2.bar(summary[cat], summary["count"], color=colors, edgecolor="white")
    ax2.set_title(f"Record Count by {cat}")
    ax2.set_xlabel(cat); ax2.set_ylabel("Count")
    ax2.tick_params(axis="x", rotation=20)
    ax2.grid(axis="y")

    fig.suptitle(f"Category Analysis — {cat}", fontsize=14, fontweight="bold")
    save("06_bar_chart")

def plot_grouped_bar(df: pd.DataFrame,
                     x: str = "dept", hue: str = "promoted"):
    ct = pd.crosstab(df[x], df[hue], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(11, 6))
    ct.plot(kind="bar", stacked=True, ax=ax,
            color=sns.color_palette(PALETTE, ct.shape[1]),
            edgecolor="white", linewidth=0.6)
    ax.set_title(f"Proportion of '{hue}' within each '{x}'  (%)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel(x); ax.set_ylabel("Percentage (%)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title=hue, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y")
    save("07_grouped_bar")

def plot_time_series():
    dates  = pd.date_range("2022-01-01", periods=52, freq="W")
    revenue = np.cumsum(np.random.normal(500, 80, 52)) + 10_000
    costs   = np.cumsum(np.random.normal(300, 60, 52)) +  7_000
    profit  = revenue - costs

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(dates, revenue, alpha=0.15, color="#4C8EDA")
    ax.fill_between(dates, costs,   alpha=0.15, color="#E63946")
    ax.plot(dates, revenue, color="#4C8EDA", lw=2.2, label="Revenue")
    ax.plot(dates, costs,   color="#E63946", lw=2.2, label="Costs")
    ax.plot(dates, profit,  color="#2A9D8F", lw=2,   linestyle="--", label="Profit")
    ax.set_title("Weekly Revenue, Costs & Profit", fontsize=14, fontweight="bold")
    ax.set_xlabel("Week"); ax.set_ylabel("Amount ($)")
    ax.legend(); ax.grid(True)
    fig.autofmt_xdate()
    save("08_time_series")

def plot_kde_histogram(df: pd.DataFrame,
                       col: str = "score", hue: str = "promoted"):
    fig, ax = plt.subplots(figsize=(10, 5))
    groups = df[hue].unique()
    colors = ["#4C8EDA", "#E63946"]

    for (grp, color) in zip(groups, colors):
        data = df.loc[df[hue] == grp, col]
        ax.hist(data, bins=30, alpha=0.35, color=color, edgecolor="none", density=True)
        sns.kdeplot(data, ax=ax, color=color, lw=2.5, label=f"{hue}={grp}")

    ax.set_title(f"KDE + Histogram — {col} by {hue}")
    ax.set_xlabel(col); ax.set_ylabel("Density")
    ax.legend(); ax.grid(axis="y")
    save("09_kde_histogram")

def plot_missing(df: pd.DataFrame):
    """Visualise missing data patterns. Injects NaNs for demo if data is clean."""
    df_miss = df.copy().astype(object)
    for col in df_miss.columns:
        idx = np.random.choice(df_miss.index, size=int(0.05 * len(df_miss)), replace=False)
        df_miss.loc[idx, col] = np.nan

    miss_pct = df_miss.isnull().mean() * 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                    gridspec_kw={"width_ratios": [3, 1]})

    sns.heatmap(df_miss.isnull(), cbar=False, yticklabels=False,
                cmap=["#D9D9D9", "#E63946"], ax=ax1)
    ax1.set_title("Missing Value Map  (red = missing)")
    ax1.set_xlabel("Columns")

    miss_pct.sort_values().plot(kind="barh", ax=ax2, color=ACCENT)
    ax2.set_title("Missing %")
    ax2.set_xlabel("% Missing")
    ax2.axvline(5, color="#E63946", lw=1.5, linestyle="--", label="5 % threshold")
    ax2.legend(fontsize=8)

    fig.suptitle("Missing Data Analysis", fontsize=14, fontweight="bold")
    save("10_missing_values")

if __name__ == "__main__":
    print("\n📊  Dataset Analysis Visualizations")
    print("=" * 45)
    print(f"Dataset shape : {df.shape}")
    print(f"Columns       : {', '.join(df.columns)}\n")
    print(df.describe().round(1).to_string())
    print("\nGenerating plots …\n")

    plot_overview(df)
    plot_correlation(df)
    plot_pairplot(df, hue="dept")
    plot_box_violin(df, x="dept", y="salary")
    plot_scatter_regression(df, x="age", y="salary", hue="promoted")
    plot_bar_chart(df, cat="dept", val="salary")
    plot_grouped_bar(df, x="dept", hue="promoted")
    plot_time_series()
    plot_kde_histogram(df, col="score", hue="promoted")
    plot_missing(df)

    print("\n✅  All 10 visualizations saved as PNG files.")
    print("    Swap in your own DataFrame at the top of the script and re-run!")
