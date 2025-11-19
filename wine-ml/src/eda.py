import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

RAW = Path(os.getenv("DATA_RAW", "./data/raw/winemag-data-130k-v2.csv"))
OUT = Path(os.getenv("DATA_CLEAN", "./data/processed/wine_reviews_clean.parquet"))
OUT.parent.mkdir(parents=True, exist_ok=True)
FIGS = Path("./figures"); FIGS.mkdir(exist_ok=True, parents=True)

print("Leyendo:", RAW)
df = pd.read_csv(RAW, low_memory=False, na_values=["", "NA","NaN","null"])
cols = ["description","points","price","variety","country","winery",
        "taster_name","title","province","region_1","region_2"]
df = df[[c for c in cols if c in df.columns]].copy()

# Normalización ligera
for c in df.select_dtypes(include="object"):
    df[c] = df[c].astype(str).str.strip().replace({"nan": pd.NA})

df["points"] = pd.to_numeric(df["points"], errors="coerce")
if "price" in df: df["price"] = pd.to_numeric(df["price"], errors="coerce")

# Filtros mínimos
df = df[df["description"].notna()]
df = df[df["points"].between(80, 100, inclusive="both")]
if "price" in df: df = df[df["price"].isna() | (df["price"] > 0)]
if {"title","winery"}.issubset(df.columns):
    df = df.drop_duplicates(subset=["title","winery"])

# Guardar limpio
df.to_parquet(OUT, index=False)
print("Guardado limpio en:", OUT, "| filas:", len(df))

# Resumen y figuras básicas
def p(s,q): s=s.dropna(); return np.percentile(s,q) if len(s) else np.nan
summary = pd.DataFrame({
    "count":[df.points.count(), df.price.count() if "price" in df else np.nan],
    "mean":[df.points.mean(), df.price.mean() if "price" in df else np.nan],
    "median":[df.points.median(), df.price.median() if "price" in df else np.nan],
    "std":[df.points.std(), df.price.std() if "price" in df else np.nan],
    "min":[df.points.min(), df.price.min() if "price" in df else np.nan],
    "max":[df.points.max(), df.price.max() if "price" in df else np.nan],
    "p95":[p(df.points,95), p(df.price,95) if "price" in df else np.nan],
}, index=["points","price"])
print(summary)

plt.figure(); df.points.plot(kind="hist", bins=20, edgecolor="k", title="Distribución points (80–100)")
plt.savefig(FIGS/"hist_points.png", dpi=160, bbox_inches="tight")

if "price" in df:
    plt.figure(); df.price.plot(kind="hist", bins=40, edgecolor="k", title="Distribución price")
    plt.savefig(FIGS/"hist_price.png", dpi=160, bbox_inches="tight")
    plt.figure(); np.log1p(df.price).plot(kind="hist", bins=40, edgecolor="k", title="Distribución log(1+price)")
    plt.savefig(FIGS/"hist_log_price.png", dpi=160, bbox_inches="tight")
    plt.figure(); plt.scatter(np.log1p(df.price), df.points, s=5, alpha=0.2)
    plt.xlabel("log(1+price)"); plt.ylabel("points"); plt.title("log(price) vs points")
    plt.savefig(FIGS/"scatter_logprice_points.png", dpi=160, bbox_inches="tight")

if "variety" in df:
    plt.figure(); df.variety.value_counts().head(10).plot(kind="bar", title="Top-10 variety")
    plt.savefig(FIGS/"top10_variety.png", dpi=160, bbox_inches="tight")
print("Figuras en ./figures")
