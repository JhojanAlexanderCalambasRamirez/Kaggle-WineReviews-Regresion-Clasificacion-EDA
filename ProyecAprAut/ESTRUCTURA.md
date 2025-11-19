# 📂 Estructura del Proyecto - Wine Quality Predictor

```
ProyecAprAut/
│
├── 📄 README.md                          # Documentación principal completa
├── 📄 QUICKSTART.md                      # Guía de inicio rápido
├── 📄 ESTRUCTURA.md                      # Este archivo
├── 📄 requirements.txt                   # Dependencias Python
├── 📄 .gitignore                        # Archivos ignorados por Git
│
├── 🚀 run_gui.bat                        # Ejecutar interfaz gráfica
├── 🚀 run_cli.bat                        # Ejecutar CLI interactivo
├── 🚀 run_training.bat                   # Ejecutar entrenamiento completo
│
├── 📂 src/                               # CÓDIGO FUENTE
│   ├── 📂 gui/                          # Interfaz Gráfica
│   │   └── 🐍 wine_predictor_gui.py    # App GUI principal (CustomTkinter)
│   │                                     # - 2 pestañas: Predicción y Entrenamiento
│   │                                     # - Semáforo de calidad visual
│   │                                     # - Explicabilidad LIME integrada
│   │                                     # - Ventana de ayuda con créditos
│   │
│   ├── 📂 models/                       # Scripts de Entrenamiento
│   │   ├── 🐍 train_basic.py           # Entrenamiento básico
│   │   │                                 # - Compara 3 modelos (Ridge, RF, MLP)
│   │   │                                 # - Visualizaciones EDA
│   │   │                                 # - Explicabilidad LIME básica
│   │   │
│   │   ├── 🐍 train_with_metrics.py    # Entrenamiento con guardado
│   │   │                                 # - Guarda gráficos PNG
│   │   │                                 # - Exporta modelo .pkl
│   │   │                                 # - Genera HTML explicativos
│   │   │
│   │   └── 🐍 train_mlp_interactive.py # CLI interactivo MLP
│   │                                     # - Menú: Entrenar/Predecir/Salir
│   │                                     # - Predicción con input del usuario
│   │                                     # - Explicaciones en consola
│   │
│   └── 📂 utils/                        # Utilidades (futuro)
│       └── (vacío - para expansión futura)
│
├── 📂 data/                              # DATASETS
│   ├── 📂 raw/                          # Datos originales
│   │   ├── 📊 winemag-data-130k-v2.csv  # 130k reseñas (principal)
│   │   ├── 📊 winemag-data-130k-v2.json # Versión JSON
│   │   └── 📊 winemag-data_first150k.csv # 150k reseñas (alternativo)
│   │
│   └── 📂 processed/                    # Datos procesados (generados)
│       └── (vacío - se generan al ejecutar)
│
├── 📂 docs/                              # DOCUMENTACIÓN
│   ├── 📄 INSTALACION.md                # Guía de instalación detallada
│   ├── 📄 test_cases.txt                # Frases de prueba clasificadas
│   │                                     # - 🟢 Alta calidad (90+)
│   │                                     # - 🟡 Calidad media (85-90)
│   │                                     # - 🔴 Baja calidad (80-85)
│   │
│   ├── 📂 images/                       # Imágenes de documentación
│   │   └── (vacío - para screenshots)
│   │
│   └── 📂 resultados/                   # Resultados de entrenamiento
│       ├── 📈 1_distribucion_puntos.png # Histograma de calidad
│       ├── ☁️  2_nube_palabras.png      # Word cloud
│       ├── 📊 3_comparacion_modelos.png # Barras MAE
│       ├── 🌐 explicacion_*.html        # Explicaciones LIME
│       └── 🧠 modelo_vino_entrenado.pkl # Modelo Ridge guardado
│
├── 📂 sistema_vino/                     # MODELOS ENTRENADOS
│   ├── 🧠 cerebro_vino.pkl              # Modelo para GUI
│   └── 🧠 cerebro_vino_mlp.pkl          # Modelo MLP para CLI
│
├── 📂 notebooks/                        # Jupyter Notebooks (futuro)
│   └── (vacío - para análisis exploratorio)
│
└── 📂 config/                           # Configuraciones (futuro)
    └── (vacío - para archivos .env, etc.)
```

---

## 🎯 Flujo de Archivos

### **Entrenamiento:**
```
data/raw/*.csv
    → src/models/train_*.py
    → sistema_vino/*.pkl
    → docs/resultados/*.png
```

### **Predicción (GUI):**
```
Usuario escribe reseña
    → src/gui/wine_predictor_gui.py
    → sistema_vino/cerebro_vino.pkl
    → Predicción + Explicación
```

### **Predicción (CLI):**
```
Usuario escribe reseña
    → src/models/train_mlp_interactive.py
    → sistema_vino/cerebro_vino_mlp.pkl
    → Predicción en consola
```

---

## 📊 Tamaños Aproximados

| Tipo | Tamaño |
|------|--------|
| **Datasets CSV** | ~50-60 MB cada uno |
| **Modelos .pkl** | ~20-30 MB cada uno |
| **Imágenes PNG** | ~100-500 KB cada una |
| **Código Python** | ~10-15 KB cada archivo |

**Total del proyecto:** ~150-200 MB (con datasets)

---

## 🔑 Archivos Clave

### **Ejecutables (Inicio Rápido):**
1. `run_gui.bat` - Mejor para usuarios finales
2. `run_cli.bat` - Para uso en consola
3. `run_training.bat` - Para generar nuevos modelos

### **Documentación (Aprendizaje):**
1. `README.md` - Referencia completa
2. `QUICKSTART.md` - Inicio en 5 minutos
3. `docs/INSTALACION.md` - Setup detallado

### **Código (Desarrollo):**
1. `src/gui/wine_predictor_gui.py` - App principal
2. `src/models/train_with_metrics.py` - Mejor para investigación
3. `src/models/train_mlp_interactive.py` - Mejor para producción

---

## 🎨 Convenciones de Nombres

- **Scripts ejecutables:** `run_*.bat`
- **Módulos de entrenamiento:** `train_*.py`
- **Resultados:** `*_*.png` (número + descripción)
- **Modelos:** `cerebro_vino*.pkl`
- **Docs Markdown:** `MAYUSCULAS.md`

---

## 🚦 Estado de Carpetas

| Carpeta | Estado | Propósito |
|---------|--------|-----------|
| `src/` | ✅ Activo | Código fuente principal |
| `data/raw/` | ✅ Activo | Datasets originales |
| `data/processed/` | 📦 Generado | Se crea al entrenar |
| `docs/resultados/` | ✅ Activo | Salidas de entrenamiento |
| `sistema_vino/` | ✅ Activo | Modelos listos |
| `notebooks/` | 🔮 Futuro | Para Jupyter |
| `config/` | 🔮 Futuro | Para configuraciones |
| `src/utils/` | 🔮 Futuro | Funciones comunes |

---

## 🔄 Ciclo de Vida

```
1. INSTALACIÓN
   └── requirements.txt → pip install

2. DESCARGA DE RECURSOS
   └── NLTK stopwords, wordnet

3. ENTRENAMIENTO (Primera vez)
   └── run_training.bat
       ├── Lee: data/raw/*.csv
       ├── Genera: docs/resultados/*.png
       └── Guarda: sistema_vino/*.pkl

4. PREDICCIÓN (Uso continuo)
   └── run_gui.bat o run_cli.bat
       ├── Carga: sistema_vino/*.pkl
       └── Predice reseñas nuevas

5. REENTRENAMIENTO (Opcional)
   └── Ejecutar nuevamente run_training.bat
       └── Sobrescribe modelos antiguos
```

---

## 📝 Notas Importantes

1. **No subir a Git:**
   - `data/raw/*.csv` (muy pesados)
   - `sistema_vino/*.pkl` (modelos binarios grandes)
   - Ver `.gitignore` para detalles

2. **Mantener versionados:**
   - Todo el código en `src/`
   - Documentación en `docs/*.md`
   - Scripts ejecutables `run_*.bat`

3. **Backup crítico:**
   - `sistema_vino/*.pkl` si tarda mucho entrenar
   - `docs/resultados/` si son resultados finales

---

**🎯 Para navegación rápida, usa tu IDE con búsqueda de archivos (Ctrl+P en VSCode)**
