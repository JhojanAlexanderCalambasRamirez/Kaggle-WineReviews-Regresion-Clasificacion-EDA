# 🏗️ Arquitectura Modular - Wine AI Prophet

## 📋 Resumen

El proyecto ha sido reestructurado siguiendo principios de **ingeniería de software moderna**:
- ✅ Separación de responsabilidades
- ✅ Código reutilizable
- ✅ Configuración centralizada
- ✅ Sin redundancia

---

## 📁 Nueva Estructura

```
ProyecAprAut/
├── src/
│   ├── gui/
│   │   ├── wine_ai_prophet.py      ⭐ GUI PRINCIPAL (refactorizada)
│   │   └── wine_predictor_gui.py   (versión anterior simple)
│   ├── models/
│   │   ├── train_basic.py
│   │   ├── train_with_metrics.py
│   │   └── train_mlp_interactive.py
│   └── utils/                       🆕 NUEVO PAQUETE
│       ├── __init__.py
│       ├── nlp_processor.py         🆕 Procesamiento NLP
│       └── sommelier_agent.py       🆕 Agente Inteligente
│
├── config/                          🆕 NUEVO PAQUETE
│   ├── __init__.py
│   └── settings.py                  🆕 Configuración central
│
├── run_prophet.bat                  🆕 Ejecutar versión mejorada
└── ...
```

---

## 🔧 Módulos Creados

### 1. **`src/utils/nlp_processor.py`**

**Responsabilidad:** Procesamiento de Lenguaje Natural

**Funciones:**
- `verificar_recursos_nltk()` → Descarga automática de recursos
- `limpiar_texto(texto)` → Preprocesamiento completo
- `lime_wrapper(textos, modelo)` → Adaptador para LIME

**Ventajas:**
- Reutilizable en todos los scripts
- No se repite código de NLP
- Fácil de testear

**Uso:**
```python
from src.utils import limpiar_texto

texto_limpio = limpiar_texto("This wine is elegant and fruity.")
```

---

### 2. **`src/utils/sommelier_agent.py`**

**Responsabilidad:** Generación de narrativas expertas

**Clase:** `SommelierAgent`

**Métodos públicos:**
- `generar_narrativa(score, lime_weights)` → Narrativa completa
- `obtener_color_y_mensaje(score)` → Color y mensaje UI

**Métodos privados:**
- `_seleccionar_introduccion(score)` → Frase de apertura
- `_analizar_factores(factores, tipo)` → Análisis positivo/negativo

**Ventajas:**
- Lógica de negocio separada de UI
- Fácil de extender (añadir más frases)
- Testeable independientemente

**Uso:**
```python
from src.utils import SommelierAgent

agent = SommelierAgent()
narrativa = agent.generar_narrativa(
    score=92.5,
    lime_weights=[('elegant', 0.45), ('rich', 0.38)]
)
```

---

### 3. **`config/settings.py`**

**Responsabilidad:** Configuración global del proyecto

**Constantes definidas:**

#### Rutas (dinámicas):
```python
PROJECT_ROOT          # Raíz del proyecto
DATA_RAW_DIR         # data/raw/
MODELS_DIR           # sistema_vino/
MODEL_PATH           # cerebro_vino.pkl
DATASET_130K         # winemag-data-130k-v2.csv
RESULTS_DIR          # docs/resultados/
```

#### Parámetros ML:
```python
TFIDF_MAX_FEATURES = 3000
MLP_HIDDEN_LAYERS = (50, 50)
MLP_MAX_ITER = 30
TEST_SIZE = 0.2
LIME_NUM_FEATURES = 6
```

#### Configuración GUI:
```python
APP_TITLE = "Wine AI Prophet 🍷"
APP_GEOMETRY = "950x750"
COLOR_EXCELENTE = "#27AE60"
UMBRAL_EXCELENTE = 90
```

**Ventajas:**
- Un solo lugar para cambiar parámetros
- Rutas siempre correctas (dinámicas)
- Fácil experimentación

**Uso:**
```python
from config.settings import MODEL_PATH, TFIDF_MAX_FEATURES

modelo = joblib.load(MODEL_PATH)
vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
```

---

### 4. **`src/gui/wine_ai_prophet.py`**

**Responsabilidad:** Interfaz gráfica principal (refactorizada)

**Mejoras vs versión anterior:**

| Aspecto | Antes | Ahora |
|---------|-------|-------|
| **Líneas de código** | ~463 | ~390 |
| **Lógica NLP** | En el mismo archivo | Importada de utils |
| **Agente Sommelier** | En el mismo archivo | Importado de utils |
| **Rutas** | Hardcoded | Desde config |
| **Parámetros ML** | Hardcoded | Desde config |
| **Métodos UI** | Largos | Divididos en submétodos |
| **Legibilidad** | Baja | Alta |

**Arquitectura de la clase:**

```python
class WineAIApp(ctk.CTk):
    # Inicialización
    __init__()

    # Componentes UI (privados)
    _crear_header()
    _crear_pestanas()
    _setup_prediccion()
    _crear_area_resultados()
    _setup_entrenamiento()

    # Lógica de negocio
    _cargar_modelo_inicial()
    _proceso_entrenamiento()
    _proceso_prediccion()
    _actualizar_resultados()

    # Ventanas emergentes
    _mostrar_ayuda()
```

**Ventajas:**
- Código más limpio y organizado
- Fácil de mantener
- Sin duplicación
- Claridad en las responsabilidades

---

## 🔄 Flujo de Datos

### **Entrenamiento:**
```
Usuario → GUI
    ↓
_proceso_entrenamiento()
    ↓
DATASET_130K (config)
    ↓
limpiar_texto() (utils)
    ↓
MLPRegressor (parámetros de config)
    ↓
MODEL_PATH (config)
```

### **Predicción:**
```
Usuario → Input Text
    ↓
limpiar_texto() (utils)
    ↓
modelo.predict()
    ↓
LIME (parámetros de config)
    ↓
SommelierAgent.generar_narrativa() (utils)
    ↓
UI actualizada
```

---

## 🎯 Beneficios de la Refactorización

### ✅ **Mantenibilidad**
- Cambiar frases del agente → Solo editar `sommelier_agent.py`
- Cambiar parámetros ML → Solo editar `settings.py`
- Agregar nueva funcionalidad NLP → Solo editar `nlp_processor.py`

### ✅ **Reusabilidad**
```python
# Usar el agente en otro script
from src.utils import SommelierAgent
agent = SommelierAgent()
```

### ✅ **Testabilidad**
```python
# Test unitario del agente
def test_narrativa_alta_calidad():
    agent = SommelierAgent()
    narrativa = agent.generar_narrativa(95, [])
    assert "🏆" in narrativa
```

### ✅ **Escalabilidad**
- Fácil agregar nuevos agentes
- Fácil agregar nuevas interfaces (CLI, Web)
- Fácil agregar nuevos modelos

---

## 🚀 Cómo Usar

### **Ejecutar la app mejorada:**
```bash
# Windows:
.\run_prophet.bat

# O directamente:
python src/gui/wine_ai_prophet.py
```

### **Importar componentes en otro script:**
```python
# Importar utilidades
from src.utils import limpiar_texto, SommelierAgent

# Importar configuración
from config.settings import MODEL_PATH, DATASET_130K
```

---

## 📊 Comparación de Versiones

| Característica | V1 (wine_predictor_gui.py) | V2 (wine_ai_prophet.py) |
|----------------|---------------------------|------------------------|
| **Agente Sommelier** | ❌ | ✅ |
| **Código modular** | ❌ | ✅ |
| **Config centralizada** | ❌ | ✅ |
| **Rutas dinámicas** | ✅ | ✅ |
| **Líneas de código** | ~350 | ~390 (con más features) |
| **Explicabilidad** | Técnica | Natural |
| **Mantenibilidad** | Media | Alta |
| **Reusabilidad** | Baja | Alta |

---

## 📝 Próximos Pasos (Opcionales)

- [ ] Tests unitarios en `tests/`
- [ ] CLI usando los mismos módulos utils
- [ ] API REST usando los mismos módulos
- [ ] Dashboard web con Streamlit
- [ ] Logging centralizado

---

## 🎓 Principios Aplicados

1. **DRY (Don't Repeat Yourself)** → No duplicamos código NLP
2. **Single Responsibility** → Cada módulo tiene una función
3. **Separation of Concerns** → UI separada de lógica
4. **Configuration Management** → Configuración centralizada
5. **Clean Code** → Métodos cortos y claros

---

**🏆 Resultado: Código profesional, escalable y mantenible.**
