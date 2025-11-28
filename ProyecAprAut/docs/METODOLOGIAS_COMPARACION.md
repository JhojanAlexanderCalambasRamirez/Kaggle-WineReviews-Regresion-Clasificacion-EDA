# Comparación de 3 Metodologías de Preprocesamiento

## Objetivo

Implementar y comparar **3 metodologías DIFERENTES** de preprocesamiento de texto para cumplir con el requisito académico de aplicar múltiples enfoques de limpieza de datos en Machine Learning.

---

## Las 3 Metodologías Implementadas

### 📌 Metodología 1: TF-IDF Básico (Sin Limpieza NLP)

**Descripción:**
- **Preprocesamiento mínimo:** Solo lowercase y eliminación de símbolos
- **NO elimina stopwords** (the, is, and, etc.)
- **NO aplica stemming ni lemmatization**
- Enfoque minimalista para establecer baseline

**Ventajas:**
✅ Más rápido computacionalmente
✅ Preserva toda la información original
✅ Simple de implementar

**Desventajas:**
❌ Incluye ruido (palabras comunes sin significado)
❌ Mayor dimensionalidad
❌ Posible overfitting

**Ejemplo:**
```
Original: "This wine is elegant and complex"
Método 1: "this wine is elegant and complex"
         (solo lowercase)
```

---

### 📌 Metodología 2: TF-IDF + Stopwords + Stemming

**Descripción:**
- Limpieza con regex
- **Elimina stopwords** (the, is, and, etc.)
- **Aplica STEMMING** (PorterStemmer) - corta palabras a su raíz

**¿Qué es Stemming?**
Proceso de reducir palabras a su raíz mediante corte mecánico:
- `running` → `run`
- `wines` → `wine`
- `fruity` → `fruit`

**Ventajas:**
✅ Reduce dimensionalidad significativamente
✅ Agrupa variantes de la misma palabra
✅ Elimina palabras sin valor semántico

**Desventajas:**
❌ Stemming puede ser muy agresivo
❌ Puede perder matices semánticos importantes
❌ Palabras irreconocibles (`complexness` → `complex`)

**Ejemplo:**
```
Original: "This wine is elegant and complex"
Método 2: "wine eleg complex"
         (stopwords eliminadas + stemming aplicado)
```

---

### 📌 Metodología 3: TF-IDF + Stopwords + Lemmatization + N-grams

**Descripción:**
- Limpieza con regex
- **Elimina stopwords**
- **Aplica LEMMATIZATION** (WordNetLemmatizer) - conversión lingüística inteligente
- **Usa N-grams (1,2)** - captura pares de palabras para contexto

**¿Qué es Lemmatization?**
Proceso de convertir palabras a su forma base usando análisis lingüístico:
- `running` → `run`
- `better` → `good`
- `wines` → `wine`

**¿Qué son N-grams?**
Secuencias de N palabras consecutivas que capturan contexto:
- Unigrams (1): `["wine", "elegant", "complex"]`
- Bigrams (2): `["wine elegant", "elegant complex"]`

**Ventajas:**
✅ Preserva significado lingüístico correcto
✅ Captura contexto con bigramas
✅ Más preciso semánticamente
✅ Suele dar mejores resultados en NLP

**Desventajas:**
❌ Más lento computacionalmente
❌ Mayor complejidad de implementación
❌ Requiere más memoria (n-grams aumentan features)

**Ejemplo:**
```
Original: "This wine is elegant and complex"
Método 3: "wine elegant complex"
         + bigramas: ["wine elegant", "elegant complex"]
         (lemmatization preserva semántica + contexto capturado)
```

---

## Diferencia Clave: Stemming vs Lemmatization

| Característica | Stemming | Lemmatization |
|----------------|----------|---------------|
| **Método** | Corte mecánico de sufijos | Análisis lingüístico |
| **Velocidad** | ⚡ Muy rápido | 🐢 Más lento |
| **Precisión** | ❌ Menor | ✅ Mayor |
| **Ejemplo 1** | `caring` → `car` ❌ | `caring` → `care` ✅ |
| **Ejemplo 2** | `better` → `better` | `better` → `good` ✅ |
| **Uso** | Búsquedas rápidas | NLP avanzado, ML |

---

## Comparación de Resultados

El script `train_three_methodologies.py` entrena **el mismo modelo MLP** con cada metodología y compara:

### Métricas Evaluadas:

1. **MAE (Mean Absolute Error)** - Menor es mejor
   - Error promedio en la predicción
   - Ejemplo: MAE = 1.4 significa error de ±1.4 puntos

2. **RMSE (Root Mean Squared Error)** - Menor es mejor
   - Penaliza más los errores grandes
   - Más sensible a outliers

3. **R² Score** - Mayor es mejor (0 a 1)
   - Qué tan bien el modelo explica la varianza
   - R² = 0.85 significa que explica el 85% de la variación

4. **Tiempo de Entrenamiento** - Menor es mejor
   - Eficiencia computacional

---

## Cómo Ejecutar la Comparación

### Opción 1: Archivo Batch (Windows)
```bash
run_comparison.bat
```

### Opción 2: Comando Directo
```bash
python src\models\train_three_methodologies.py
```

---

## Resultados Esperados

El script genera:

1. **Tabla comparativa en consola:**
```
RESULTADOS DE LA COMPARACIÓN
================================================================================
                              Metodología       MAE      RMSE        R²  Tiempo (s)
 Metodología 3: Lemmatization + N-grams     1.345     1.891     0.856      125.3
           Metodología 2: Stemming          1.398     1.945     0.847       89.2
           Metodología 1: TF-IDF Básico     1.512     2.078     0.821       56.8
================================================================================

🏆 MEJOR METODOLOGÍA: Metodología 3: Lemmatization + N-grams
   MAE: 1.345 puntos
```

2. **Gráfico comparativo:**
   - Guardado en: `docs/resultados/comparacion_metodologias.png`
   - Muestra barras de MAE y R² para cada metodología

3. **Ejemplos de preprocesamiento:**
   - Muestra cómo cada metodología transforma el mismo texto original

---

## Conclusiones Académicas

### ¿Por qué 3 metodologías diferentes?

1. **Comparación empírica:** Permite evaluar objetivamente qué enfoque funciona mejor
2. **Análisis de trade-offs:** Cada método tiene ventajas/desventajas diferentes
3. **Decisión informada:** Elegir el mejor método basado en datos, no intuición

### Hallazgos Típicos:

- **Metodología 1 (Básico):** Baseline simple, peor performance
- **Metodología 2 (Stemming):** Balance velocidad/precisión
- **Metodología 3 (Lemmatization + N-grams):** Mejor precisión, más costoso

### Uso en la Aplicación:

Actualmente la GUI ([wine_ai_prophet.py](../src/gui/wine_ai_prophet.py)) usa la **Metodología 3** (Lemmatization) porque ofrece el mejor balance entre precisión y calidad del feedback generado por IA.

---

## Referencias Técnicas

- **TF-IDF:** Term Frequency-Inverse Document Frequency
- **Stopwords:** Palabras comunes sin valor semántico
- **Stemming:** Algoritmo Porter Stemmer (1980)
- **Lemmatization:** WordNet Lemmatizer
- **N-grams:** Secuencias de N tokens consecutivos

---

## Para el Reporte Académico

Incluir en el documento final:

1. ✅ Descripción detallada de cada metodología
2. ✅ Tabla comparativa de resultados (MAE, RMSE, R²)
3. ✅ Gráfico de barras comparativo
4. ✅ Ejemplo visual de transformación de texto
5. ✅ Justificación de la elección de la mejor metodología
6. ✅ Análisis de trade-offs (precisión vs velocidad)

---


