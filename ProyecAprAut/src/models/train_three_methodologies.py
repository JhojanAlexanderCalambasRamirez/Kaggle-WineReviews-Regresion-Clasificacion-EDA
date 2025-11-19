"""
=============================================================================
COMPARACIÓN DE 3 METODOLOGÍAS DE PREPROCESAMIENTO DE TEXTO
=============================================================================
Script que implementa y compara EXPLÍCITAMENTE tres metodologías diferentes
de preprocesamiento de texto para predicción de calidad de vinos.

METODOLOGÍAS IMPLEMENTADAS:
1. TF-IDF Básico (sin limpieza NLP)
2. TF-IDF + Stopwords + Stemming
3. TF-IDF + Stopwords + Lemmatization + N-grams

Cada metodología se aplica de forma INDEPENDIENTE y se evalúa con el mismo
modelo (MLP) para una comparación justa.
=============================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
import warnings
from typing import Tuple, Dict

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# ML
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Configuración
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# =============================================================================
# CONFIGURACIÓN INICIAL
# =============================================================================

# Rutas dinámicas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "winemag-data-130k-v2.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "docs", "resultados")

# Crear carpeta de resultados
os.makedirs(RESULTS_DIR, exist_ok=True)

# Verificar recursos NLTK
print("="*80)
print("COMPARACIÓN DE 3 METODOLOGÍAS DE PREPROCESAMIENTO DE TEXTO")
print("="*80)
print("\n[1/7] Verificando recursos NLTK...")

try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Descargando recursos NLTK...")
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

print("✓ Recursos NLTK listos")

# Inicializar herramientas NLP
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


# =============================================================================
# METODOLOGÍA 1: TF-IDF BÁSICO (SIN LIMPIEZA NLP)
# =============================================================================
def metodologia_1_basica(texto: str) -> str:
    """
    METODOLOGÍA 1: TF-IDF Básico

    Descripción:
    - Solo convierte a minúsculas
    - NO elimina stopwords
    - NO aplica stemming ni lemmatization
    - Enfoque minimalista para baseline

    Ventaja: Preserva toda la información original
    Desventaja: Incluye ruido (palabras comunes sin significado)
    """
    if not isinstance(texto, str):
        return ""

    # Solo lowercase y limpieza básica de símbolos
    texto = texto.lower()
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)

    return texto


# =============================================================================
# METODOLOGÍA 2: TF-IDF + STOPWORDS + STEMMING
# =============================================================================
def metodologia_2_stemming(texto: str) -> str:
    """
    METODOLOGÍA 2: TF-IDF + Stopwords + Stemming

    Descripción:
    - Limpieza con regex
    - Elimina stopwords (the, is, and, etc.)
    - Aplica STEMMING (corta palabras a raíz: running → run, wines → wine)

    Ventaja: Reduce dimensionalidad, agrupa variantes de palabras
    Desventaja: Puede perder matices semánticos (stemming agresivo)

    Ejemplo:
    "This wine is complex" → "wine complex" (stopwords eliminadas + stemming)
    """
    if not isinstance(texto, str):
        return ""

    # Limpieza
    texto = texto.lower()
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)

    # Tokenización
    tokens = texto.split()

    # Aplicar: Stopwords + Stemming
    tokens_procesados = [
        stemmer.stem(token)  # STEMMING (PorterStemmer)
        for token in tokens
        if token not in stop_words  # Eliminar stopwords
    ]

    return " ".join(tokens_procesados)


# =============================================================================
# METODOLOGÍA 3: TF-IDF + STOPWORDS + LEMMATIZATION + N-GRAMS
# =============================================================================
def metodologia_3_lemmatization(texto: str) -> str:
    """
    METODOLOGÍA 3: TF-IDF + Stopwords + Lemmatization

    Descripción:
    - Limpieza con regex
    - Elimina stopwords
    - Aplica LEMMATIZATION (convierte a forma base preservando significado)
    - Usa n-grams (pares de palabras) en el vectorizador

    Ventaja: Preserva significado lingüístico, captura contexto (n-grams)
    Desventaja: Más lento computacionalmente

    Ejemplo:
    "This wine is complex" → "wine complex" (lemmatization + preserva semántica)

    Diferencia con Stemming:
    - Stemming: "running" → "run" (corte mecánico)
    - Lemmatization: "better" → "good" (análisis lingüístico)
    """
    if not isinstance(texto, str):
        return ""

    # Limpieza
    texto = texto.lower()
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)

    # Tokenización
    tokens = texto.split()

    # Aplicar: Stopwords + Lemmatization
    tokens_procesados = [
        lemmatizer.lemmatize(token)  # LEMMATIZATION (WordNetLemmatizer)
        for token in tokens
        if token not in stop_words  # Eliminar stopwords
    ]

    return " ".join(tokens_procesados)


# =============================================================================
# FUNCIÓN DE ENTRENAMIENTO Y EVALUACIÓN
# =============================================================================
def entrenar_y_evaluar(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    nombre_metodologia: str
) -> Dict:
    """
    Entrena un modelo MLP y retorna métricas de evaluación
    """
    print(f"\n   → Entrenando modelo MLP con {nombre_metodologia}...")

    start_time = time.time()

    # Modelo MLP (mismo para todas las metodologías)
    modelo = MLPRegressor(
        hidden_layer_sizes=(50, 50),
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )

    modelo.fit(X_train, y_train)

    # Predicciones
    y_pred = modelo.predict(X_test)

    # Métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    tiempo = time.time() - start_time

    print(f"   ✓ Completado en {tiempo:.2f}s - MAE: {mae:.3f}")

    return {
        'modelo': modelo,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'Tiempo (s)': tiempo,
        'y_pred': y_pred
    }


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================
def main():
    """
    Función principal que ejecuta la comparación de las 3 metodologías
    """

    # -------------------------------------------------------------------------
    # PASO 1: CARGAR DATOS
    # -------------------------------------------------------------------------
    print(f"\n[2/7] Cargando dataset...")

    try:
        df = pd.read_csv(DATA_PATH, usecols=['description', 'points'])
        df = df.dropna().drop_duplicates()
        print(f"✓ Dataset cargado: {len(df)} reseñas únicas")
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo en {DATA_PATH}")
        return

    # Limitar tamaño para pruebas rápidas (opcional)
    # df = df.sample(n=10000, random_state=42)  # Descomentar para pruebas

    # -------------------------------------------------------------------------
    # PASO 2: APLICAR LAS 3 METODOLOGÍAS DE PREPROCESAMIENTO
    # -------------------------------------------------------------------------
    print(f"\n[3/7] Aplicando las 3 metodologías de preprocesamiento...")

    print("\n   METODOLOGÍA 1: TF-IDF Básico (sin limpieza NLP)")
    t0 = time.time()
    df['metodo_1_basico'] = df['description'].apply(metodologia_1_basica)
    print(f"   ✓ Completado en {time.time()-t0:.2f}s")

    print("\n   METODOLOGÍA 2: TF-IDF + Stopwords + Stemming")
    t0 = time.time()
    df['metodo_2_stemming'] = df['description'].apply(metodologia_2_stemming)
    print(f"   ✓ Completado en {time.time()-t0:.2f}s")

    print("\n   METODOLOGÍA 3: TF-IDF + Stopwords + Lemmatization")
    t0 = time.time()
    df['metodo_3_lemmatization'] = df['description'].apply(metodologia_3_lemmatization)
    print(f"   ✓ Completado en {time.time()-t0:.2f}s")

    # Mostrar ejemplos
    print("\n" + "="*80)
    print("EJEMPLO DE PREPROCESAMIENTO CON LAS 3 METODOLOGÍAS:")
    print("="*80)
    ejemplo_idx = 42
    print(f"\nRESEÑA ORIGINAL:\n'{df['description'].iloc[ejemplo_idx]}'")
    print(f"\nMETODOLOGÍA 1 (Básico):\n'{df['metodo_1_basico'].iloc[ejemplo_idx]}'")
    print(f"\nMETODOLOGÍA 2 (Stemming):\n'{df['metodo_2_stemming'].iloc[ejemplo_idx]}'")
    print(f"\nMETODOLOGÍA 3 (Lemmatization):\n'{df['metodo_3_lemmatization'].iloc[ejemplo_idx]}'")
    print("="*80)

    # -------------------------------------------------------------------------
    # PASO 3: VECTORIZACIÓN Y DIVISIÓN DE DATOS
    # -------------------------------------------------------------------------
    print(f"\n[4/7] Vectorizando con TF-IDF...")

    y = df['points'].values
    resultados = {}

    # METODOLOGÍA 1: TF-IDF básico (sin n-grams)
    print("\n   → Vectorizando Metodología 1...")
    vectorizer_1 = TfidfVectorizer(max_features=3000, ngram_range=(1, 1))
    X_1 = vectorizer_1.fit_transform(df['metodo_1_basico']).toarray()
    X_train_1, X_test_1, y_train, y_test = train_test_split(
        X_1, y, test_size=0.2, random_state=42
    )
    print(f"   ✓ Shape: {X_1.shape}")

    # METODOLOGÍA 2: TF-IDF (sin n-grams para stemming)
    print("\n   → Vectorizando Metodología 2...")
    vectorizer_2 = TfidfVectorizer(max_features=3000, ngram_range=(1, 1))
    X_2 = vectorizer_2.fit_transform(df['metodo_2_stemming']).toarray()
    X_train_2, X_test_2, _, _ = train_test_split(
        X_2, y, test_size=0.2, random_state=42
    )
    print(f"   ✓ Shape: {X_2.shape}")

    # METODOLOGÍA 3: TF-IDF + N-grams (1,2) - captura contexto
    print("\n   → Vectorizando Metodología 3 (con bigramas)...")
    vectorizer_3 = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_3 = vectorizer_3.fit_transform(df['metodo_3_lemmatization']).toarray()
    X_train_3, X_test_3, _, _ = train_test_split(
        X_3, y, test_size=0.2, random_state=42
    )
    print(f"   ✓ Shape: {X_3.shape}")

    # -------------------------------------------------------------------------
    # PASO 4: ENTRENAR MODELOS CON CADA METODOLOGÍA
    # -------------------------------------------------------------------------
    print(f"\n[5/7] Entrenando modelos MLP con cada metodología...")

    resultados['Metodología 1: TF-IDF Básico'] = entrenar_y_evaluar(
        X_train_1, X_test_1, y_train, y_test, "Metodología 1"
    )

    resultados['Metodología 2: Stemming'] = entrenar_y_evaluar(
        X_train_2, X_test_2, y_train, y_test, "Metodología 2"
    )

    resultados['Metodología 3: Lemmatization + N-grams'] = entrenar_y_evaluar(
        X_train_3, X_test_3, y_train, y_test, "Metodología 3"
    )

    # -------------------------------------------------------------------------
    # PASO 5: COMPARACIÓN DE RESULTADOS
    # -------------------------------------------------------------------------
    print(f"\n[6/7] Comparando resultados...")

    # Crear DataFrame de resultados
    df_resultados = pd.DataFrame({
        'Metodología': list(resultados.keys()),
        'MAE': [r['MAE'] for r in resultados.values()],
        'RMSE': [r['RMSE'] for r in resultados.values()],
        'R²': [r['R²'] for r in resultados.values()],
        'Tiempo (s)': [r['Tiempo (s)'] for r in resultados.values()]
    })

    # Ordenar por MAE (menor es mejor)
    df_resultados = df_resultados.sort_values('MAE')

    print("\n" + "="*80)
    print("RESULTADOS DE LA COMPARACIÓN")
    print("="*80)
    print(df_resultados.to_string(index=False))
    print("="*80)

    # Identificar mejor metodología
    mejor_metodo = df_resultados.iloc[0]['Metodología']
    mejor_mae = df_resultados.iloc[0]['MAE']

    print(f"\n🏆 MEJOR METODOLOGÍA: {mejor_metodo}")
    print(f"   MAE: {mejor_mae:.3f} puntos")
    print(f"   (Menor error promedio en la predicción)")

    # -------------------------------------------------------------------------
    # PASO 6: VISUALIZACIONES
    # -------------------------------------------------------------------------
    print(f"\n[7/7] Generando visualizaciones comparativas...")

    # Gráfico 1: Comparación de MAE
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MAE
    ax1 = axes[0]
    bars = ax1.bar(
        range(len(df_resultados)),
        df_resultados['MAE'],
        color=['#2ecc71' if i == 0 else '#3498db' for i in range(len(df_resultados))]
    )
    ax1.set_xticks(range(len(df_resultados)))
    ax1.set_xticklabels(df_resultados['Metodología'], rotation=15, ha='right')
    ax1.set_ylabel('MAE (Mean Absolute Error)')
    ax1.set_title('Comparación de Error por Metodología\n(Menor es Mejor)', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Anotar valores
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')

    # R²
    ax2 = axes[1]
    bars2 = ax2.bar(
        range(len(df_resultados)),
        df_resultados['R²'],
        color=['#2ecc71' if i == 0 else '#e74c3c' for i in range(len(df_resultados))]
    )
    ax2.set_xticks(range(len(df_resultados)))
    ax2.set_xticklabels(df_resultados['Metodología'], rotation=15, ha='right')
    ax2.set_ylabel('R² Score')
    ax2.set_title('Coeficiente de Determinación por Metodología\n(Mayor es Mejor)', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1])

    # Anotar valores
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    # Guardar gráfico
    output_path = os.path.join(RESULTS_DIR, 'comparacion_metodologias.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Gráfico guardado: {output_path}")

    plt.show()

    # -------------------------------------------------------------------------
    # RESUMEN FINAL
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("RESUMEN DE METODOLOGÍAS")
    print("="*80)
    print("""
METODOLOGÍA 1: TF-IDF Básico
├─ Preprocesamiento: Mínimo (solo lowercase)
├─ Ventaja: Rápido, simple, preserva información
└─ Desventaja: Incluye ruido (stopwords)

METODOLOGÍA 2: TF-IDF + Stopwords + Stemming
├─ Preprocesamiento: Intermedio (limpieza + stemming)
├─ Ventaja: Reduce dimensionalidad, agrupa variantes
└─ Desventaja: Stemming agresivo puede perder matices

METODOLOGÍA 3: TF-IDF + Stopwords + Lemmatization + N-grams
├─ Preprocesamiento: Avanzado (lemmatization + bigramas)
├─ Ventaja: Preserva semántica, captura contexto
└─ Desventaja: Más lento computacionalmente
    """)
    print("="*80)

    print(f"\n✓ PROCESO COMPLETADO EXITOSAMENTE")
    print(f"✓ Resultados guardados en: {RESULTS_DIR}")


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
if __name__ == "__main__":
    main()
