# Wine Quality Predictor - NLP & Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8+-green.svg)](https://www.nltk.org/)

> Sistema inteligente de predicción de calidad de vinos mediante Procesamiento de Lenguaje Natural (NLP), Redes Neuronales y Agente Sommelier con IA.

Predice la puntuación de calidad de un vino (escala 80-100) basándose en la descripción textual de sommeliers, con explicabilidad mediante LIME y feedback profesional generado por IA.

---

## Características Principales

- **Modelo MLP Neural Network** con MAE de ~1.37 puntos
- **Agente Sommelier Inteligente** con integración de múltiples APIs de IA (Groq, Gemini, OpenAI, Claude)
- **Explicabilidad con LIME** - Comprende qué palabras influyen en cada predicción
- **Interfaz Gráfica Moderna** con CustomTkinter
- **Preprocesamiento NLP Avanzado** (Regex, Stopwords, Lematización)
- **Arquitectura Modular** con configuración centralizada

---

## Instalación Rápida

### 1. Crear Entorno Virtual

**Opción A: Conda (Recomendado)**
```bash
conda create --name ProyeVino python=3.10 -y
conda activate ProyeVino
conda install pandas numpy matplotlib seaborn scikit-learn nltk -y
```

**Opción B: pip**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

### 2. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 3. Descargar Recursos NLTK

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## Configuración de IA (Opcional pero Recomendado)

Para obtener feedback profesional generado por IA en lugar de frases predefinidas:

### 1. Obtener API Key de Groq (GRATIS)

1. Ve a: https://console.groq.com/keys
2. Crea una cuenta gratis
3. Clic en "Create API Key"
4. Copia la clave (empieza con `gsk_...`)

### 2. Configurar Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto:

```bash
GROQ_API_KEY=gsk_tu_clave_aqui
AI_PROVIDER=groq
USE_AI_FEEDBACK=true
```

### 3. Instalar Cliente de Groq

```bash
pip install groq
```

### Proveedores Alternativos

| Proveedor | Costo | Velocidad | Variable en .env |
|-----------|-------|-----------|------------------|
| **Groq** (Llama 3.3) | GRATIS | Muy rápido | `GROQ_API_KEY` |
| **Gemini** (Google) | GRATIS* | Rápido | `GEMINI_API_KEY` |
| **OpenAI** (GPT-4) | De pago | Medio | `OPENAI_API_KEY` |
| **Claude** (Haiku) | De pago | Rápido | `ANTHROPIC_API_KEY` |

*Gemini gratis hasta 60 req/min

Para desactivar IA y volver al agente básico:
```bash
USE_AI_FEEDBACK=false
```

---

## Uso

### Ejecutar la Aplicación

**Windows:**
```bash
run_prophet.bat
```

**Manualmente:**
```bash
python src/gui/wine_ai_prophet.py
```

### Flujo de Trabajo

1. **Entrenar Modelo** (si no existe):
   - Ve a la pestaña "ENTRENAMIENTO"
   - Clic en "INICIAR ENTRENAMIENTO"
   - Espera a que termine (proceso automático)

2. **Analizar Vinos**:
   - Ve a la pestaña "PREDICCIÓN & SOMMELIER"
   - Escribe una reseña de vino en inglés
   - Clic en "CONSULTAR AL AGENTE SOMMELIER"
   - Obtén puntuación y análisis profesional

### Ejemplos de Reseñas

**Alta Calidad (esperado: 90+):**
```
This wine is elegant, complex and has rich tannins with a long finish.
```

**Calidad Media (esperado: 85-90):**
```
A decent wine with fruity notes and moderate acidity.
```

**Baja Calidad (esperado: 80-85):**
```
This wine is flat, watery and lacks character.
```

---

## Estructura del Proyecto

```
ProyecAprAut/
├── src/
│   ├── gui/
│   │   └── wine_ai_prophet.py      # Aplicación principal
│   ├── utils/
│   │   ├── nlp_processor.py        # Limpieza NLP
│   │   ├── sommelier_agent.py      # Agente básico
│   │   └── ai_integrations.py      # Integraciones con APIs de IA
│   └── models/
│       └── (scripts de entrenamiento legacy)
│
├── config/
│   └── settings.py                  # Configuración centralizada
│
├── data/
│   └── raw/
│       ├── winemag-data-130k-v2.csv # Dataset principal (130k reseñas)
│       └── winemag-data_first150k.csv
│
├── sistema_vino/
│   └── cerebro_vino_mlp.pkl         # Modelo entrenado
│
├── .env                             # Variables de entorno (API keys)
├── requirements.txt                 # Dependencias
└── run_prophet.bat                  # Ejecutor Windows
```

---

## Metodología

### Pipeline NLP

1. **Limpieza de Texto:**
   - Conversión a minúsculas
   - Eliminación de símbolos especiales (regex)
   - Tokenización

2. **Preprocesamiento:**
   - Eliminación de stopwords (`the`, `is`, `and`, etc.)
   - Lematización (`running` → `run`, `wines` → `wine`)

3. **Vectorización:**
   - TF-IDF con 3000 features
   - Prioriza palabras únicas sobre comunes

4. **Modelo:**
   - MLP (Multilayer Perceptron) Neural Network
   - Arquitectura: Input (3000) → Hidden (50) → Hidden (50) → Output (1)
   - MAE: ~1.37 puntos

5. **Explicabilidad:**
   - LIME identifica palabras clave positivas/negativas
   - Genera pesos de influencia por palabra

6. **Feedback con IA:**
   - Convierte datos técnicos en análisis de sommelier profesional
   - Soporta múltiples proveedores (Groq, Gemini, OpenAI, Claude)
   - Fallback al agente básico si falla la API

---

## Dataset

**Fuente:** Wine Magazine Reviews
**Tamaño:** 130,000 reseñas de vinos profesionales
**Escala:** 80-100 puntos

**Columnas:**
- `description`: Texto descriptivo del sommelier (entrada del modelo)
- `points`: Puntuación 80-100 (objetivo a predecir)

---

## Solución de Problemas

### Error: "Modelo no encontrado"
```bash
# Ve a la pestaña "ENTRENAMIENTO" y entrena el modelo
# O verifica que existe: sistema_vino/cerebro_vino_mlp.pkl
```

### Error: "API key no encontrada"
```bash
# Verifica que existe el archivo .env en la raíz
# Verifica que contiene: GROQ_API_KEY=tu_clave_aqui
```

### Error: "No module named 'groq'"
```bash
pip install groq
```

### Feedback muy genérico (con IA activada)
```bash
# En .env, aumenta la temperatura:
AI_TEMPERATURE=0.8
```

### Error: "Rate limit exceeded"
- **Groq:** Espera 1 minuto (límite: 30 req/min gratis)
- **Gemini:** Espera 1 minuto (límite: 60 req/min gratis)

---

## Tecnologías

- **Python 3.10+**
- **Machine Learning:** Scikit-Learn, MLP Neural Network
- **NLP:** NLTK, TF-IDF Vectorization
- **Explicabilidad:** LIME
- **IA:** OpenAI, Google Gemini, Anthropic Claude, Groq
- **GUI:** CustomTkinter
- **Configuración:** python-dotenv

---

## Equipo de Desarrollo

Proyecto Final - Aprendizaje Automático

| Nombre | ID Estudiante |
|--------|---------------|
| Oscar Portela | 22507314 |
| Jorge Fong | 2205016 |
| Jhojan Alexander Calambas Ramirez | 2190555 |
| Angelo Parra Cortez | 22506988 |
| Juan Sebastian Rodriguez | 2195060 |

---

## Licencia

Proyecto académico desarrollado con fines educativos.

---

## Contacto y Soporte

Para casos de prueba, consulta: [docs/test_cases.txt](docs/test_cases.txt)

---

> *"El vino es poesía embotellada. La ciencia de datos es la llave para descifrarla."*
