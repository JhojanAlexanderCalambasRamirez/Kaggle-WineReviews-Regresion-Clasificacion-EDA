"""
Módulo de Procesamiento de Lenguaje Natural
============================================
Contiene funciones para limpieza, preprocesamiento y utilidades NLP.
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Verificar recursos NLTK (Descarga automática si falta)
def verificar_recursos_nltk():
    """Verifica y descarga recursos NLTK necesarios"""
    print("Verificando diccionarios de lenguaje...")
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Descargando recursos NLTK...")
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

# Inicializar componentes NLP
verificar_recursos_nltk()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def limpiar_texto(texto):
    """
    Preprocesa texto aplicando:
    1. Lowercasing
    2. Regex (solo letras)
    3. Eliminación de stopwords
    4. Lematización

    Args:
        texto (str): Texto crudo a procesar

    Returns:
        str: Texto limpio y procesado
    """
    if not isinstance(texto, str):
        return ""

    # 1. Minúsculas
    texto = texto.lower()

    # 2. Solo letras y espacios
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)

    # 3. Tokenización
    tokens = texto.split()

    # 4. Stopwords + Lematización
    tokens_limpios = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words
    ]

    return " ".join(tokens_limpios)

def lime_wrapper(textos, modelo):
    """
    Adaptador para que LIME funcione con modelos de regresión.

    Args:
        textos: Lista de textos a predecir
        modelo: Modelo de scikit-learn

    Returns:
        array: Predicciones en formato (N, 1)
    """
    return modelo.predict(textos).reshape(-1, 1)
