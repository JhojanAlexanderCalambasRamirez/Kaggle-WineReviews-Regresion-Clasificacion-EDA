import joblib, numpy as np
from pathlib import Path

MODEL_PATH = list(Path("./models").glob("reg_*.joblib"))[0]
pipe = joblib.load(MODEL_PATH)

pre = pipe.named_steps["pre"]
model = pipe.named_steps["model"]

# Nombres de features del bloque TF-IDF (solo para términos)
tfidf = pre.named_transformers_["tfidf"]
vocab = np.array(tfidf.get_feature_names_out())

# ¡OJO! El vector de coeficientes mezcla texto + num + cat;
# para ilustrar, tomamos los primeros len(vocab) como proxy del bloque texto.
coefs = model.coef_[:len(vocab)]

top_pos = np.argsort(coefs)[-20:][::-1]
top_neg = np.argsort(coefs)[:20]

print("\nTérminos que más SUMAN puntos:\n", vocab[top_pos])
print("\nTérminos que más RESTAN puntos:\n", vocab[top_neg])
