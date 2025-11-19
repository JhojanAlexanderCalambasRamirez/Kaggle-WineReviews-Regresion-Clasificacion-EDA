# =============================================================================
# PROYECTO: PREDICCIÓN DE CALIDAD DE VINO Y EXPLICABILIDAD (LOCAL VS CODE)
# =============================================================================
# Autor: Gemini (Asistente)
# Entorno: Local (Python 3.x)
# Descripción: 
# Script completo que carga datos de vinos, visualiza estadísticas, preprocesa 
# texto, entrena 3 modelos (Ridge, RF, MLP) y explica predicciones individuales.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
import warnings
from wordcloud import WordCloud

# Librerías de Procesamiento de Lenguaje Natural (NLP)
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Librerías de Machine Learning (Scikit-Learn)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error

# Modelos (3 Tipos para análisis de complejidad)
from sklearn.linear_model import Ridge             # Modelo 1: Lineal
from sklearn.ensemble import RandomForestRegressor # Modelo 2: Ensamble
from sklearn.neural_network import MLPRegressor    # Modelo 3: Red Neuronal

# Librería de Explicabilidad
from lime.lime_text import LimeTextExplainer

# Configuración inicial
warnings.filterwarnings('ignore') # Ignorar advertencias de versiones
sns.set(style="whitegrid")

# =============================================================================
# 1. CONFIGURACIÓN DE RECURSOS NLTK
# =============================================================================
print("--- 1. VERIFICANDO RECURSOS LINGÜÍSTICOS ---")
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Descargando recursos de NLTK necesarios (solo la primera vez)...")
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
print("Recursos listos.")

# =============================================================================
# 2. CARGA DE DATOS
# =============================================================================
def cargar_datos(ruta_archivo):
    print(f"\n--- 2. CARGANDO DATASET: {ruta_archivo} ---")
    try:
        # Intentamos cargar el archivo local
        df = pd.read_csv(ruta_archivo, usecols=['description', 'points'])
        df = df.dropna().drop_duplicates()
        print(f"Dataset cargado. Total de reseñas únicas: {len(df)}")
        return df
    except FileNotFoundError:
        print(f"AVISO: No se encontró '{ruta_archivo}'.")
        print("Generando datos sintéticos para demostración...")
        # Generador de datos dummy para que el script no falle si no tienes el archivo
        return pd.DataFrame({
            'description': [
                'This wine is complex, fruity and has a long finish.',
                'Simple, flat, acidic and lacks structure.',
                'Rich tannins, dark berry flavors and smooth texture.',
                'Watery and bitter.'
            ] * 200,
            'points': [92, 82, 90, 80] * 200
        })

# =============================================================================
# 3. ANÁLISIS EXPLORATORIO (EDA)
# =============================================================================
def realizar_eda(df):
    print("\n--- 3. GENERANDO VISUALIZACIONES (EDA) ---")
    print("(Por favor, cierra las ventanas de los gráficos para continuar la ejecución)")
    
    # Gráfico 1: Histograma
    plt.figure(figsize=(10, 5))
    sns.histplot(df['points'], bins=20, kde=True, color='darkred')
    plt.title('Distribución de la Calidad del Vino')
    plt.xlabel('Puntos')
    plt.tight_layout()
    plt.show() # Bloquea ejecución hasta cerrar ventana en local

    # Gráfico 2: Nube de Palabras
    texto = " ".join(review for review in df['description'].iloc[:5000])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texto)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Palabras frecuentes')
    plt.tight_layout()
    plt.show()

# =============================================================================
# 4. PRE-PROCESAMIENTO
# =============================================================================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocesar_texto(texto):
    # Metodología 1: Limpieza Regex
    texto = texto.lower()
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    
    tokens = texto.split()
    
    # Metodología 2 y 3: Stopwords y Lematización
    tokens_procesados = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token not in stop_words
    ]
    return " ".join(tokens_procesados)

# =============================================================================
# 5. FLUJO PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    # A. Carga
    archivo = 'winemag-data-130k-v2.csv' # Asegúrate de tener este archivo en la misma carpeta
    df = cargar_datos(archivo)
    
    # B. EDA
    realizar_eda(df)
    
    # C. Preprocesamiento
    print("\n--- 4. PRE-PROCESANDO TEXTO (3 METODOLOGÍAS) ---")
    t0 = time.time()
    df['clean_text'] = df['description'].apply(preprocesar_texto)
    print(f"Texto procesado en {time.time()-t0:.2f} segundos.")
    
    # D. Vectorización y Split
    print("\n--- 5. VECTORIZACIÓN Y DIVISIÓN ---")
    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(df['clean_text']).toarray()
    y = df['points'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # E. Modelos
    print("\n--- 6. ENTRENAMIENTO Y COMPARACIÓN DE MODELOS ---")
    resultados = {}
    
    # Modelo 1: Ridge
    start = time.time()
    model_ridge = Ridge(alpha=1.0)
    model_ridge.fit(X_train, y_train)
    time_ridge = time.time() - start
    pred_ridge = model_ridge.predict(X_test)
    resultados['Ridge'] = {'MAE': mean_absolute_error(y_test, pred_ridge), 'Tiempo': time_ridge}
    print(f"Ridge terminado (MAE: {resultados['Ridge']['MAE']:.2f})")
    
    # Modelo 2: Random Forest
    start = time.time()
    model_rf = RandomForestRegressor(n_estimators=30, max_depth=10, n_jobs=-1, random_state=42)
    model_rf.fit(X_train, y_train)
    time_rf = time.time() - start
    pred_rf = model_rf.predict(X_test)
    resultados['Random Forest'] = {'MAE': mean_absolute_error(y_test, pred_rf), 'Tiempo': time_rf}
    print(f"Random Forest terminado (MAE: {resultados['Random Forest']['MAE']:.2f})")

    # Modelo 3: MLP
    start = time.time()
    model_mlp = MLPRegressor(hidden_layer_sizes=(50,), max_iter=30, random_state=42)
    model_mlp.fit(X_train, y_train)
    time_mlp = time.time() - start
    pred_mlp = model_mlp.predict(X_test)
    resultados['MLP'] = {'MAE': mean_absolute_error(y_test, pred_mlp), 'Tiempo': time_mlp}
    print(f"MLP terminado (MAE: {resultados['MLP']['MAE']:.2f})")

    # Visualización Complejidad
    df_res = pd.DataFrame(resultados).T
    print("\nResumen Comparativo:\n", df_res)
    
    # F. Explicabilidad Final
    print("\n--- 7. EXPLICABILIDAD (LIME) ---")
    pipeline_final = make_pipeline(vectorizer, model_ridge)
    
    # 1. Función "Wrapper" para hacer compatible la Regresión con LIME
    def funcion_prediccion_lime(textos):
        # Predecimos con el pipeline normal
        preds = pipeline_final.predict(textos)
        # IMPORTANTE: Convertimos el array de forma (N,) a (N, 1)
        # Esto evita el error de índices en LIME
        return preds.reshape(-1, 1)

    def explicar_instancia(texto):
        # Limpieza inicial para mostrar en pantalla
        texto_proc = preprocesar_texto(texto)
        
        # Predicción simple para mostrar al usuario
        pred = pipeline_final.predict([texto_proc])[0]
        
        # Inicializamos LIME (Sin parámetros extraños)
        explainer = LimeTextExplainer(verbose=False, random_state=42)
        
        # Generamos la explicación
        # labels=[0] es CRUCIAL: le dice a LIME que mire la única columna que existe (el puntaje)
        exp = explainer.explain_instance(
            texto_proc, 
            funcion_prediccion_lime, # Usamos nuestra función especial
            num_features=5, 
            labels=[0] 
        )
        
        print("="*60)
        print(f"RESEÑA: {texto}")
        print(f"PREDICCIÓN: {pred:.2f} Puntos")
        print("-" * 60)
        print("PALABRAS CLAVE (Impacto en el puntaje):")
        
        # Extraemos la explicación de la etiqueta 0
        lista_pesos = exp.as_list(label=0)
        
        for palabra, peso in lista_pesos:
            tipo = "(+) SUBE PUNTOS" if peso > 0 else "(-) BAJA PUNTOS"
            print(f"  * '{palabra}': {peso:.4f}  {tipo}")
        print("="*60)

    # Pruebas manuales
    try:
        explicar_instancia("This wine is elegant, balanced, and has a rich finish.")
        explicar_instancia("The wine is bitter, flat and very acidic.")
    except Exception as e:
        print(f"\nError durante la explicación: {e}")

    print("\nPROCESO FINALIZADO EXITOSAMENTE.")