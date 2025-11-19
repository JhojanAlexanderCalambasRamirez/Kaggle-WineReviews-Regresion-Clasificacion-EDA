import os, time, joblib, numpy as np, pandas as pd
from pathlib import Path
from dotenv import load_dotenv; load_dotenv()

DATA = Path(os.getenv("DATA_CLEAN", "./data/processed/wine_reviews_clean.parquet"))
GROUP_COL = os.getenv("GROUP_COL", "winery")
RND = int(os.getenv("RANDOM_STATE", "42"))
MODELS_DIR = Path("./models"); MODELS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA)
df["label_superior"] = (df["points"] >= 90).astype(int)

TEXT_COL = "description"
NUM_COLS = [c for c in ["price"] if c in df.columns]
CAT_COLS = [c for c in ["variety","country"] if c in df.columns]
GROUPS = df[GROUP_COL].fillna("UNKNOWN") if GROUP_COL in df.columns else pd.Series(["ALL"]*len(df))

X = df[[TEXT_COL] + NUM_COLS + CAT_COLS].copy()
y = df["points"].astype(float).copy()

# Preprocesamiento
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
def impute_median_frame(Xframe): return Xframe.fillna(Xframe.median(numeric_only=True))

num_pipe = Pipeline([('impute', FunctionTransformer(impute_median_frame, validate=False)),
                     ('log',    FunctionTransformer(lambda X: np.log1p(X), validate=False)),
                     ('scale',  RobustScaler())])

ohe = OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=50, sparse=True)
tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1,2), max_features=80000, min_df=5)

pre = ColumnTransformer([
    ('tfidf', tfidf, TEXT_COL),
    ('num',   num_pipe, NUM_COLS),
    ('cat',   ohe,      CAT_COLS),
], sparse_threshold=0.3)

# Modelos
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
import lightgbm as lgb
candidatos = {
    "ridge": Ridge(alpha=2.0, random_state=RND),
    "linsvr": LinearSVR(C=1.0, epsilon=0.0, random_state=RND),
    "lgbm": lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05,
                              num_leaves=31, subsample=0.9, colsample_bytree=0.9,
                              random_state=RND)
}

# CV por grupos
from sklearn.model_selection import GroupKFold, cross_validate
cv = GroupKFold(n_splits=5)

rows = []
for nombre, est in candidatos.items():
    pipe = Pipeline([('pre', pre), ('model', est)])
    t0 = time.time()
    scores = cross_validate(
        pipe, X, y, cv=cv, groups=GROUPS,
        scoring={'MAE':'neg_mean_absolute_error','RMSE':'neg_root_mean_squared_error','R2':'r2'},
        return_train_score=False, n_jobs=-1
    )
    elapsed = time.time() - t0
    mae  = -scores['test_MAE'].mean()
    rmse = -scores['test_RMSE'].mean()
    r2   =  scores['test_R2'].mean()
    print(f"{nombre:6s} | MAE={mae:.2f} | RMSE={rmse:.2f} | R2={r2:.3f} | t={elapsed:.1f}s")
    rows.append(dict(modelo=nombre, MAE=mae, RMSE=rmse, R2=r2, tiempo_cv_s=elapsed))

# Guardar el mejor por MAE entrenado en TODO el set
res = pd.DataFrame(rows).sort_values("MAE")
best_name = res.iloc[0]["modelo"]
print("\nMejor por MAE:", best_name, "\n", res)

best = candidatos[best_name]
best_pipe = Pipeline([('pre', pre), ('model', best)])
best_pipe.fit(X, y)
joblib.dump(best_pipe, MODELS_DIR/f"reg_{best_name}.joblib")
res.to_csv(MODELS_DIR/"cv_results_regression.csv", index=False)
print("Modelo guardado en:", MODELS_DIR/f"reg_{best_name}.joblib")
