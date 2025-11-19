# =============================================================================
# PROYECTO: PREDICCIÓN DE VINO (VERSIÓN CON GUARDADO LOCAL)
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
import warnings
import os  # ### NUEVO: Para crear carpetas
import joblib # ### NUEVO: Para guardar el modelo entrenado
from wordcloud import WordCloud

# Librerías de NLP y ML
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from lime.lime_text import LimeTextExplainer

# Configuración inicial
warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

# ### NUEVO: Crear carpeta para resultados si no existe ###
CARPETA_SALIDA = "resultados_proyecto"
if not os.path.exists(CARPETA_SALIDA):
    os.makedirs(CARPETA_SALIDA)
    print(f"--- Carpeta '{CARPETA_SALIDA}' creada para guardar archivos ---")

# =============================================================================
# 1. CONFIGURACIÓN Y CARGA
# =============================================================================
print("\n--- 1. VERIFICANDO RECURSOS ---")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

def cargar_datos(ruta_archivo):
    print(f"--- 2. CARGANDO DATASET: {ruta_archivo} ---")
    try:
        # Intenta cargar archivo local
        df = pd.read_csv(ruta_archivo, usecols=['description', 'points'])
        df = df.dropna().drop_duplicates()
        return df
    except FileNotFoundError:
        print(f"AVISO: No se encontró '{ruta_archivo}'. Usando datos sintéticos.")
        return pd.DataFrame({
            'description': ['Wine is complex and fruity'] * 200 + ['Acidic and flat'] * 200,
            'points': [90] * 200 + [80] * 200
        })

# =============================================================================
# 2. EDA CON GUARDADO DE GRÁFICAS
# =============================================================================
def realizar_eda(df):
    print("\n--- 3. GENERANDO Y GUARDANDO GRÁFICAS (EDA) ---")
    
    # Gráfico 1: Histograma
    plt.figure(figsize=(10, 5))
    sns.histplot(df['points'], bins=20, kde=True, color='darkred')
    plt.title('Distribución de la Calidad del Vino')
    plt.xlabel('Puntos')
    plt.tight_layout()
    
    # ### NUEVO: Guardar antes de mostrar ###
    ruta_hist = os.path.join(CARPETA_SALIDA, '1_distribucion_puntos.png')
    plt.savefig(ruta_hist)
    print(f"   -> Gráfica guardada en: {ruta_hist}")
    plt.show() 

    # Gráfico 2: Nube de Palabras
    texto = " ".join(review for review in df['description'].iloc[:5000])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texto)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Palabras frecuentes')
    plt.tight_layout()
    
    # ### NUEVO: Guardar antes de mostrar ###
    ruta_cloud = os.path.join(CARPETA_SALIDA, '2_nube_palabras.png')
    plt.savefig(ruta_cloud)
    print(f"   -> Gráfica guardada en: {ruta_cloud}")
    plt.show()

# =============================================================================
# 3. PRE-PROCESAMIENTO
# =============================================================================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocesar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    tokens = [lemmatizer.lemmatize(t) for t in texto.split() if t not in stop_words]
    return " ".join(tokens)

# =============================================================================
# 4. FLUJO PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    # A. Carga (Reemplaza con tu ruta si es necesario)
    archivo = r'winemag-data-130k-v2.csv' 
    df = cargar_datos(archivo)
    
    # B. EDA
    realizar_eda(df)
    
    # C. Procesamiento
    print("\n--- 4. PRE-PROCESANDO TEXTO ---")
    t0 = time.time()
    df['clean_text'] = df['description'].apply(preprocesar_texto)
    print(f"Texto procesado en {time.time()-t0:.2f} s.")
    
    # D. Vectorización
    print("\n--- 5. VECTORIZACIÓN ---")
    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(df['clean_text']).toarray()
    y = df['points'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # E. Entrenamiento
    print("\n--- 6. ENTRENANDO MODELOS ---")
    resultados = {}
    
    # Ridge
    start = time.time()
    model_ridge = Ridge(alpha=1.0).fit(X_train, y_train)
    resultados['Ridge'] = {'MAE': mean_absolute_error(y_test, model_ridge.predict(X_test)), 'Time': time.time()-start}
    print(f"Ridge: MAE {resultados['Ridge']['MAE']:.2f}")
    
    # MLP (Red Neuronal) - El mejor según tus pruebas
    start = time.time()
    model_mlp = MLPRegressor(hidden_layer_sizes=(50,), max_iter=30, random_state=42).fit(X_train, y_train)
    resultados['MLP'] = {'MAE': mean_absolute_error(y_test, model_mlp.predict(X_test)), 'Time': time.time()-start}
    print(f"MLP: MAE {resultados['MLP']['MAE']:.2f}")

    # ### NUEVO: Guardar Gráfica de Comparación ###
    df_res = pd.DataFrame(resultados).T
    plt.figure(figsize=(8, 5))
    df_res['MAE'].plot(kind='bar', color='salmon')
    plt.title('Comparación de Error (MAE)')
    plt.ylabel('Error MAE')
    ruta_comp = os.path.join(CARPETA_SALIDA, '3_comparacion_modelos.png')
    plt.savefig(ruta_comp)
    print(f"   -> Gráfica de comparación guardada en: {ruta_comp}")
    plt.show()

    # F. Guardado del Modelo
    print("\n--- 7. GUARDANDO EL MEJOR MODELO ---")
    # Creamos el pipeline final con el modelo más rápido/equilibrado (Ridge) o el más preciso (MLP)
    # Usaremos Ridge para LIME porque es más compatible y rápido para la demo.
    pipeline_final = make_pipeline(vectorizer, model_ridge)
    
    ruta_modelo = os.path.join(CARPETA_SALIDA, 'modelo_vino_entrenado.pkl')
    joblib.dump(pipeline_final, ruta_modelo)
    print(f"✅ ¡MODELO GUARDADO EXITOSAMENTE! Lo tienes en: {ruta_modelo}")
    print("   (Puedes cargarlo después con: modelo = joblib.load('...'))")

    # G. Explicabilidad (LIME)
    print("\n--- 8. EXPLICABILIDAD (LIME) ---")
    
    # Función auxiliar para LIME
    def funcion_prediccion_lime(textos):
        return pipeline_final.predict(textos).reshape(-1, 1)

    def explicar_instancia(texto):
        texto_proc = preprocesar_texto(texto)
        pred = pipeline_final.predict([texto_proc])[0]
        
        explainer = LimeTextExplainer(verbose=False, random_state=42)
        exp = explainer.explain_instance(texto_proc, funcion_prediccion_lime, num_features=5, labels=[0])
        
        print(f"\nRESEÑA: {texto[:50]}...")
        print(f"PREDICCIÓN: {pred:.2f}")
        print("IMPACTO PALABRAS:")
        for pal, peso in exp.as_list(label=0):
            signo = "+" if peso > 0 else "-"
            print(f"  [{signo}] {pal}: {peso:.3f}")
            
        # ### NUEVO: Guardar explicación como HTML ###
        nombre_archivo = f"explicacion_{texto.split()[0]}.html"
        ruta_html = os.path.join(CARPETA_SALIDA, nombre_archivo)
        exp.save_to_file(ruta_html)
        print(f"   -> Explicación detallada guardada en: {ruta_html}")

    explicar_instancia("This wine is elegant, balanced, and has a rich finish.")
    explicar_instancia("The wine is bitter, flat and very acidic.")

    print("\nPROCESO COMPLETADO.")