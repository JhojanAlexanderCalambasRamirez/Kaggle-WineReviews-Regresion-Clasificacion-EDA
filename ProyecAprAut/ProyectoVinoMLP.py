# =============================================================================
# APP DE CALIDAD DE VINO: ENTRENAMIENTO Y PREDICCIÓN INTERACTIVA
# =============================================================================
import pandas as pd
import numpy as np
import os
import time
import re
import joblib  # Para guardar/cargar el modelo
import warnings

# NLP y ML
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor # El mejor modelo según tus pruebas
from sklearn.metrics import mean_absolute_error
from lime.lime_text import LimeTextExplainer

# Configuración
warnings.filterwarnings('ignore')
CARPETA_SALIDA = "sistema_vino"
ARCHIVO_MODELO = os.path.join(CARPETA_SALIDA, "cerebro_vino_mlp.pkl")

# Crear carpeta si no existe
if not os.path.exists(CARPETA_SALIDA):
    os.makedirs(CARPETA_SALIDA)

# Verificar recursos NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# =============================================================================
# 1. FUNCIONES DE LIMPIEZA (COMUNES PARA ENTRENAR Y PREDECIR)
# =============================================================================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def limpiar_texto(texto):
    """Limpia, elimina stopwords y lematiza el texto."""
    if not isinstance(texto, str): return ""
    # 1. Minúsculas y eliminar símbolos raros
    texto = texto.lower()
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    
    # 2. Tokenizar, limpiar stopwords y lematizar
    tokens = texto.split()
    tokens_limpios = [
        lemmatizer.lemmatize(t) 
        for t in tokens 
        if t not in stop_words
    ]
    return " ".join(tokens_limpios)

# =============================================================================
# 2. MÓDULO DE ENTRENAMIENTO
# =============================================================================
def entrenar_nuevo_modelo():
    print("\n" + "="*40)
    print(" MODO ENTRENAMIENTO ACTIVO")
    print("="*40)
    
    # A. Cargar Datos
    ruta_csv = 'winemag-data-130k-v2.csv' # <--- VERIFICA QUE EL ARCHIVO ESTÉ AQUÍ
    print(f"--> Cargando dataset: {ruta_csv}...")
    
    try:
        df = pd.read_csv(ruta_csv, usecols=['description', 'points'])
        df = df.dropna().drop_duplicates()
    except FileNotFoundError:
        print("ERROR: No encuentro el archivo csv. Asegúrate de tenerlo en la carpeta.")
        return

    # B. Pre-procesamiento Masivo
    print(f"--> Procesando {len(df)} reseñas (esto puede tardar un poco)...")
    t0 = time.time()
    df['clean_text'] = df['description'].apply(limpiar_texto)
    print(f"    Tiempo de limpieza: {time.time()-t0:.2f} segundos.")

    # C. Preparar datos
    X = df['clean_text']
    y = df['points'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # D. Crear Pipeline (Vectorizador + Modelo)
    # Usamos MLP (Red Neuronal) porque fue tu mejor resultado (MAE 1.37)
    print("--> Entrenando Red Neuronal (MLP)...")
    pipeline = make_pipeline(
        TfidfVectorizer(max_features=3000),
        MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=50, random_state=42)
    )
    
    t_start = time.time()
    pipeline.fit(X_train, y_train)
    print(f"    Entrenamiento finalizado en {time.time()-t_start:.2f} segundos.")

    # E. Evaluación
    predicciones = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predicciones)
    print(f"--> Precisión del modelo (MAE): {mae:.4f}")

    # F. Guardar
    print(f"--> Guardando modelo en: {ARCHIVO_MODELO}")
    joblib.dump(pipeline, ARCHIVO_MODELO)
    print("¡LISTO! El modelo ha sido entrenado y guardado.")

# =============================================================================
# 3. MÓDULO DE PREDICCIÓN INTERACTIVA
# =============================================================================
def funcion_prediccion_lime(textos, modelo):
    """Auxiliar para que LIME entienda la regresión"""
    # LIME espera un array 2D, reshapeamos la salida
    return modelo.predict(textos).reshape(-1, 1)

def predecir_interactivo():
    print("\n" + "="*40)
    print(" MODO PREDICCIÓN INTERACTIVO")
    print("="*40)

    # A. Cargar modelo
    if not os.path.exists(ARCHIVO_MODELO):
        print(f"ERROR: No existe el archivo '{ARCHIVO_MODELO}'.")
        print("Debes ejecutar la opción '1. Entrenar' primero.")
        return

    print("--> Cargando cerebro digital...")
    modelo_cargado = joblib.load(ARCHIVO_MODELO)
    print("--> ¡Modelo cargado! Listo para catar vinos.")
    print("(Escribe 'salir' para volver al menú)")

    # B. Bucle de chat
    explainer = LimeTextExplainer(verbose=False, random_state=42)

    while True:
        entrada = input("\n📝 Escribe la descripción del vino (Inglés): ")
        
        if entrada.lower() in ['salir', 'exit', 'q']:
            break
        
        if len(entrada.strip()) < 5:
            print("   (Por favor escribe una frase más larga)")
            continue

        # 1. Limpiar entrada usuario
        texto_limpio = limpiar_texto(entrada)

        # 2. Predecir
        try:
            prediccion = modelo_cargado.predict([texto_limpio])[0]
            
            print("-" * 50)
            print(f"🍷 CALIDAD PREDICHA: {prediccion:.2f} / 100")
            print("-" * 50)
            print("Generando explicación (¿Por qué?)...")

            # 3. Explicar con LIME
            # Usamos lambda para pasar el modelo cargado a la función auxiliar
            exp = explainer.explain_instance(
                texto_limpio, 
                lambda x: funcion_prediccion_lime(x, modelo_cargado), 
                num_features=4, 
                labels=[0]
            )
            
            # Mostrar factores
            print("Factores Clave:")
            for palabra, peso in exp.as_list(label=0):
                signo = "⬆️ SUBE (+)" if peso > 0 else "⬇️ BAJA (-)"
                print(f"   * '{palabra}': {signo}")
            print("-" * 50)

        except Exception as e:
            print(f"Ocurrió un error al predecir: {e}")

# =============================================================================
# 4. MENÚ PRINCIPAL
# =============================================================================
def menu():
    while True:
        print("\n" + "*"*40)
        print(" SISTEMA EXPERTO DE VINOS (NLP)")
        print("*"*40)
        print("1. Entrenar modelo nuevo (Sobrescribe el anterior)")
        print("2. Usar modelo existente (Predecir)")
        print("3. Salir")
        
        opcion = input("\nSelecciona una opción (1-3): ")

        if opcion == '1':
            entrenar_nuevo_modelo()
        elif opcion == '2':
            predecir_interactivo()
        elif opcion == '3':
            print("Cerrando sistema. ¡Salud! 🍷")
            break
        else:
            print("Opción no válida.")

if __name__ == "__main__":
    menu()