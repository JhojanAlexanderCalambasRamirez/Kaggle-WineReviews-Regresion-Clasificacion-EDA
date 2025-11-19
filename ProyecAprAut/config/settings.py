"""
Configuración Central del Proyecto
===================================
Define todas las rutas, constantes y parámetros globales.
"""
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# --- RUTAS DINÁMICAS DEL PROYECTO ---
# Obtener ruta del archivo de configuración
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CONFIG_DIR)

# Rutas de datos
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_RAW_DIR = os.path.join(DATA_DIR, "raw")
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Rutas de modelos
MODELS_DIR = os.path.join(PROJECT_ROOT, "sistema_vino")
MODEL_PATH = os.path.join(MODELS_DIR, "cerebro_vino.pkl")
MODEL_MLP_PATH = os.path.join(MODELS_DIR, "cerebro_vino_mlp.pkl")

# Rutas de datasets
DATASET_130K = os.path.join(DATA_RAW_DIR, "winemag-data-130k-v2.csv")
DATASET_150K = os.path.join(DATA_RAW_DIR, "winemag-data_first150k.csv")
DATASET_JSON = os.path.join(DATA_RAW_DIR, "winemag-data-130k-v2.json")

# Rutas de resultados
RESULTS_DIR = os.path.join(PROJECT_ROOT, "docs", "resultados")

# --- PARÁMETROS DEL MODELO ---
# TF-IDF
TFIDF_MAX_FEATURES = 3000

# MLP (Red Neuronal)
MLP_HIDDEN_LAYERS = (50, 50)
MLP_MAX_ITER = 30
MLP_RANDOM_STATE = 42

# Train/Test Split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# LIME (Explicabilidad)
LIME_NUM_FEATURES = 6
LIME_RANDOM_STATE = 42

# --- CONFIGURACIÓN GUI ---
APP_TITLE = "Wine AI Prophet 🍷"
APP_GEOMETRY = "950x750"
APP_THEME = "dark-blue"
APP_MODE = "Dark"

# Colores
COLOR_EXCELENTE = "#27AE60"
COLOR_MUY_BUENO = "#F39C12"
COLOR_REGULAR = "#E74C3C"
COLOR_BOTON_ENTRENAR = "#8E44AD"
COLOR_BOTON_PREDECIR = "#27AE60"

# Umbrales de calidad
UMBRAL_EXCELENTE = 90
UMBRAL_MUY_BUENO = 85

# --- CONFIGURACIÓN DE IA (APIs) ---
# Proveedor de IA para feedback avanzado
AI_PROVIDER = os.getenv("AI_PROVIDER", "groq")  # openai, gemini, claude, groq
AI_TEMPERATURE = float(os.getenv("AI_TEMPERATURE", "0.7"))
AI_MAX_TOKENS = int(os.getenv("AI_MAX_TOKENS", "400"))

# Modo de feedback
USE_AI_FEEDBACK = os.getenv("USE_AI_FEEDBACK", "true").lower() == "true"

# --- CREAR CARPETAS SI NO EXISTEN ---
def crear_carpetas():
    """Crea las carpetas necesarias del proyecto"""
    carpetas = [
        MODELS_DIR,
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        RESULTS_DIR
    ]

    for carpeta in carpetas:
        if not os.path.exists(carpeta):
            os.makedirs(carpeta)
            print(f"Carpeta creada: {carpeta}")

# Crear carpetas al importar el módulo
crear_carpetas()
