import sys, joblib, pandas as pd
from pathlib import Path

MODEL_PATH = Path("./models/reg_ridge.joblib")  # cambia si el mejor fue otro
if not MODEL_PATH.exists():
    # fallback: busca cualquier reg_*.joblib
    found = list(Path("./models").glob("reg_*.joblib"))
    if found: MODEL_PATH = found[0]

def load_model(path): return joblib.load(path)

def predict_excel(xlsx_path, model):
    df = pd.read_excel(xlsx_path) if xlsx_path.endswith(".xlsx") else pd.read_csv(xlsx_path)
    # columnas mínimas
    need = ["description","price","variety","country","winery"]
    for c in need:
        if c not in df.columns: df[c] = None
    X = df[["description","price","variety","country"]].copy()
    pred = model.predict(X)
    out = df.copy()
    out["predicted_points"] = pred.round(1)
    return out[["title" if "title" in out.columns else "description", "predicted_points"] + need]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python -m src.predict_from_excel <ruta_excel_o_csv>")
        sys.exit(1)
    inpath = sys.argv[1]
    pipe = load_model(str(MODEL_PATH))
    out = predict_excel(inpath, pipe)
    OUT_PATH = Path("./results"); OUT_PATH.mkdir(parents=True, exist_ok=True)
    out_file = OUT_PATH / "predicciones.csv"
    out.to_csv(out_file, index=False)
    print("Guardado:", out_file)
