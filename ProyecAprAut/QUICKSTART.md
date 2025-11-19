# 🚀 Guía de Inicio Rápido - Wine Quality Predictor

## ⚡ Ejecución Inmediata

### **Opción 1: Interfaz Gráfica (Recomendado)**
```bash
# Windows
run_gui.bat

# Linux/Mac
python src/gui/wine_predictor_gui.py
```

### **Opción 2: CLI Interactivo**
```bash
# Windows
run_cli.bat

# Linux/Mac
python src/models/train_mlp_interactive.py
```

### **Opción 3: Entrenamiento con Visualizaciones**
```bash
# Windows
run_training.bat

# Linux/Mac
python src/models/train_with_metrics.py
```

---

## 📋 Prerequisitos

### 1. **Verificar Python**
```bash
python --version
# Debe ser Python 3.10 o superior
```

### 2. **Instalar Dependencias**

**Con Conda (Recomendado):**
```bash
conda create --name ProyeVino python=3.10 -y
conda activate ProyeVino
conda install pandas numpy matplotlib seaborn scikit-learn nltk -y
pip install lime wordcloud customtkinter packaging
```

**Con pip:**
```bash
pip install -r requirements.txt
```

### 3. **Descargar Recursos NLTK**
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## 📁 Estructura Rápida

```
ProyecAprAut/
├── src/                    # Código fuente
│   ├── gui/               # Interfaz gráfica
│   └── models/            # Scripts de entrenamiento
├── data/raw/              # Datasets (CSV/JSON)
├── docs/                  # Documentación
│   ├── resultados/       # Gráficos generados
│   └── test_cases.txt    # Casos de prueba
├── sistema_vino/          # Modelos entrenados (.pkl)
└── run_*.bat             # Scripts de ejecución
```

---

## 🎯 Flujo de Trabajo Típico

### **Primera Vez:**

1. **Entrenar el modelo:**
   ```bash
   run_training.bat
   ```
   - Genera visualizaciones en `docs/resultados/`
   - Guarda modelo en `docs/resultados/modelo_vino_entrenado.pkl`

2. **Usar la GUI:**
   ```bash
   run_gui.bat
   ```
   - Pestaña "ENTRENAMIENTO" → Crear modelo
   - Pestaña "PREDICCIÓN" → Probar reseñas

### **Uso Normal:**

```bash
run_gui.bat  # Solo ejecutar la interfaz gráfica
```

---

## 🧪 Casos de Prueba Rápidos

Abre `docs/test_cases.txt` y copia/pega estas frases en la GUI:

**🟢 Alta Calidad:**
```
This is truly elegant and complex with a rich finish.
```
**Predicción esperada:** ~92 puntos

**🔴 Baja Calidad:**
```
This wine is flat, watery, and lacks character.
```
**Predicción esperada:** ~82 puntos

---

## ❓ Solución de Problemas

### **Error: "No module named 'customtkinter'"**
```bash
pip install customtkinter
```

### **Error: "FileNotFoundError: winemag-data-130k-v2.csv"**
- Verifica que el archivo CSV esté en `data/raw/`
- O descarga desde: [Kaggle Wine Reviews](https://www.kaggle.com/zynicide/wine-reviews)

### **Error: "NLTK stopwords not found"**
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

### **La GUI no aparece:**
- Windows: Ejecuta `run_gui.bat` como Administrador
- Verifica que estés en el entorno virtual correcto

---

## 📊 Resultados Esperados

**Después de entrenar:**
- ✅ 3 gráficos PNG en `docs/resultados/`
- ✅ Modelo `.pkl` guardado
- ✅ MAE (error) entre 1.37 - 1.50 puntos

**Predicciones:**
- Precisión: ±1.4 puntos en escala 80-100
- Tiempo de predicción: <1 segundo
- Explicabilidad: 4-5 palabras clave identificadas

---

## 🔗 Siguiente Paso

Una vez funcionando, consulta el [README.md](README.md) completo para entender la metodología y personalización avanzada.

---

**¿Listo? ¡Ejecuta `run_gui.bat` y comienza a predecir vinos! 🍷**
