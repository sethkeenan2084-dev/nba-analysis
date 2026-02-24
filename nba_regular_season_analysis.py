import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# ---------- Config ----------
DATA_PATH = pathlib.Path("players_stats_by_season_full_details.csv")  # same folder as script by default
PLOT_PATH = pathlib.Path("three_point_accuracy_best_fit.png")

# ---------- Helpers ----------
def season_to_start_year(season_str: str) -> int:
    """Convert '1999 - 2000' -> 1999 (start year)."""
    return int(str(season_str).strip()[:4])

def safe_div(numer, denom):
    """Safe division that returns NaN when denom is 0."""
    numer = np.asarray(numer, dtype=float)
    denom = np.asarray(denom, dtype=float)
    out = np.full_like(numer, np.nan, dtype=float)
    mask = denom != 0
    out[mask] = numer[mask] / denom[mask]
    return out

# ---------- Load ----------
df = pd.read_csv(DATA_PATH)

# ---------- 1) Filter to NBA regular season ----------
nba_reg = df[(df["League"] == "NBA") & (df["Stage"] == "Regular_Season")].copy()

# ---------- 2) Player with most regular seasons ----------
seasons_per_player = nba_reg.groupby("Player")["Season"].nunique().sort_values(ascending=False)
top_player = seasons_per_player.index[0]
top_season_count = int(seasons_per_player.iloc[0])
print(f"\nPlayer with the most NBA regular seasons: {top_player} ({top_season_count} seasons)")

# ---------- 3) 3P accuracy by season ----------
p = nba_reg[nba_reg["Player"] == top_player].copy()
p["year"] = p["Season"].apply(season_to_start_year)
p = p.sort_values("year")
p["3P_acc"] = safe_div(p["3PM"], p["3PA"])

print("\nThree-point accuracy by season (3PM/3PA):")
print(p[["Season", "3PM", "3PA", "3P_acc"]].to_string(index=False))

# ---------- 4) Linear regression + best fit plot ----------
fit_df = p.dropna(subset=["3P_acc"]).copy()
x = fit_df["year"].to_numpy(dtype=float)
y = fit_df["3P_acc"].to_numpy(dtype=float)

m, b = np.polyfit(x, y, 1)              # y = m*x + b
y_hat = m * x + b

ss_res = np.sum((y - y_hat) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

print(f"\nLine of best fit for 3P accuracy: y = ({m:.6f})*year + ({b:.6f})")
print(f"R^2: {r2:.4f}")

plt.figure()
plt.scatter(x, y, label="Actual 3P accuracy")
x_line = np.linspace(x.min(), x.max(), 200)
plt.plot(x_line, m * x_line + b, label="Best-fit line")
plt.title(f"{top_player} — 3P Accuracy by Season (NBA Regular Season)")
plt.xlabel("Season start year")
plt.ylabel("3P accuracy (3PM/3PA)")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=200)
plt.close()
print(f"\nSaved best-fit plot to: {PLOT_PATH.resolve()}")

# ---------- 5) Average 3P accuracy from fit line (integration average) ----------
x0, x1 = float(x.min()), float(x.max())
f0 = m * x0 + b
f1 = m * x1 + b
avg_acc_fit = (f0 + f1) / 2  # average value of a line over an interval

actual_avg_acc = float(np.nanmean(p["3P_acc"]))
actual_avg_3pm = float(np.mean(p["3PM"]))

print("\nAverage 3P accuracy comparisons:")
print(f"- Average accuracy from fit line (integrated avg): {avg_acc_fit:.4f}")
print(f"- Actual average accuracy (mean of season accuracies): {actual_avg_acc:.4f}")
print(f"- Actual average 3PM per season (count; different units): {actual_avg_3pm:.2f}")

# ---------- 6) Interpolate missing seasons (2002 and 2015 start years) ----------
full_years = np.arange(int(p["year"].min()), int(p["year"].max()) + 1)

p_full = p.set_index("year")[["Season", "3PM", "3PA", "3P_acc"]].reindex(full_years).copy()

# Build a season label series aligned to the index, then fill missing Season values
season_labels = pd.Series(
    [f"{yr} - {yr+1}" for yr in p_full.index.astype(int)],
    index=p_full.index,
    dtype="string"
)
p_full["Season"] = p_full["Season"].astype("string").fillna(season_labels)

# Interpolate numeric columns across years
for col in ["3PM", "3PA", "3P_acc"]:
    p_full[col] = p_full[col].astype(float).interpolate(method="linear")

missing_estimates = p_full.loc[[2002, 2015], ["Season", "3PM", "3PA", "3P_acc"]]
print("\nInterpolated estimates for missing seasons (start years 2002 and 2015):")
print(missing_estimates.to_string(index=True))

# ---------- 7) Stats for FGM and FGA (dataset-wide) ----------
fgm = nba_reg["FGM"].dropna().to_numpy(dtype=float)
fga = nba_reg["FGA"].dropna().to_numpy(dtype=float)

def describe_stats(arr):
    return {
        "mean": float(np.mean(arr)),
        "variance": float(np.var(arr, ddof=1)),  # sample variance
        "skew": float(sp.stats.skew(arr, bias=False)),
        "kurtosis": float(sp.stats.kurtosis(arr, bias=False, fisher=True)),  # Fisher (0 = normal)
    }

fgm_stats = describe_stats(fgm)
fga_stats = describe_stats(fga)

print("\nFGM vs FGA descriptive statistics (NBA Regular Season):")
print("FGM:", fgm_stats)
print("FGA:", fga_stats)

print(
    "\nQuick comparison:\n"
    "- FGA should have a bigger mean/variance than FGM because attempts are always >= makes.\n"
    "- Skew/kurtosis tell you about shape: skew = lopsidedness, kurtosis = tail/heaviness.\n"
)

# ---------- 8) T-tests ----------
paired_df = nba_reg[["FGM", "FGA"]].dropna()

t_rel = sp.stats.ttest_rel(paired_df["FGM"], paired_df["FGA"])  # paired / relational
t_ind = sp.stats.ttest_ind(nba_reg["FGM"].dropna(), nba_reg["FGA"].dropna(), equal_var=False)  # unpaired
t_fgm_1s = sp.stats.ttest_1samp(nba_reg["FGM"].dropna(), popmean=0)  # one-sample
t_fga_1s = sp.stats.ttest_1samp(nba_reg["FGA"].dropna(), popmean=0)

print("\nT-test results:")
print(f"- Paired t-test (FGM vs FGA): statistic={t_rel.statistic:.4f}, p-value={t_rel.pvalue:.4e}")
print(f"- Independent t-test (FGM vs FGA): statistic={t_ind.statistic:.4f}, p-value={t_ind.pvalue:.4e}")
print(f"- One-sample t-test on FGM vs 0: statistic={t_fgm_1s.statistic:.4f}, p-value={t_fgm_1s.pvalue:.4e}")
print(f"- One-sample t-test on FGA vs 0: statistic={t_fga_1s.statistic:.4f}, p-value={t_fga_1s.pvalue:.4e}")

print(
    "\nHow paired vs regular compares (human version):\n"
    "- Paired t-test answers: 'within each row, is FGM different from FGA?'\n"
    "- Independent t-test answers: 'are the two columns different if we pretend they’re unrelated samples?'\n"
    "- One-sample t-tests answer: 'is this column’s mean different from 0?' (usually yes here — sanity check).\n"
)
