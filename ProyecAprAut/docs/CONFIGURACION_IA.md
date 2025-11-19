# 🤖 Configuración de IA para Feedback Avanzado

## 📋 Resumen

Esta guía te explica cómo configurar **APIs de IA** para generar feedback profesional y natural en lugar de las frases predefinidas básicas.

---

## 🎯 ¿Por qué usar IA?

### **ANTES (Agente básico):**
```
🏆 Este vino muestra cualidades excepcionales. Con una proyección de 92.5 puntos...

✅ FORTALEZAS: Destaca positivamente por su carácter 'ELEGANT', 'RICH', 'COMPLEX'...
```

### **DESPUÉS (Con IA):**
```
Este Cabernet Sauvignon presenta características excepcionales que lo posicionan
en el rango superior (92.5/100). Su perfil aromático, descrito como "elegante" y
"rico", sugiere una elaboración cuidadosa con barricas de roble francés bien
integradas.

La presencia de taninos sedosos indica madurez fenólica óptima, mientras que los
descriptores de "frutas negras" y "chocolate" confirman la complejidad esperada
en un vino de esta categoría. La ligera nota "ácida" podría sugerir una cosecha
más fresca o menor tiempo de crianza.

Recomendación: Ideal para maridar con carnes rojas a la parrilla o quesos curados.
Temperatura de servicio: 16-18°C.
```

---

## 🚀 Opciones Disponibles

| Proveedor | Costo | Velocidad | Calidad | Recomendación |
|-----------|-------|-----------|---------|---------------|
| **Groq** (Llama 3.1) | ✅ GRATIS | ⚡ Muy rápido | ⭐⭐⭐⭐ | **🏆 RECOMENDADO** |
| **Gemini** (Google) | ✅ GRATIS* | ⚡ Rápido | ⭐⭐⭐⭐⭐ | Excelente |
| **OpenAI** (GPT-4) | 💰 $0.0001/análisis | 🐢 Medio | ⭐⭐⭐⭐⭐ | Para producción |
| **Claude** (Haiku) | 💰 $0.00025/análisis | ⚡ Rápido | ⭐⭐⭐⭐⭐ | Muy bueno |

*Gemini gratis hasta 60 req/min

---

## ⚙️ Configuración Paso a Paso

### **Opción 1: Groq (Recomendado - GRATIS y Rápido)**

#### **1. Obtener API Key:**
1. Ve a: https://console.groq.com/keys
2. Crea una cuenta (gratis)
3. Clic en "Create API Key"
4. Copia la key (empieza con `gsk_...`)

#### **2. Instalar dependencia:**
```bash
pip install groq
```

#### **3. Configurar en .env:**
```bash
# Copia el archivo de ejemplo
cp .env.example .env

# Edita .env y añade tu key:
GROQ_API_KEY=gsk_tu_api_key_aqui
AI_PROVIDER=groq
USE_AI_FEEDBACK=true
```

#### **4. ¡Listo!**
```bash
python src/gui/wine_ai_prophet.py
```

---

### **Opción 2: Google Gemini (GRATIS)**

#### **1. Obtener API Key:**
1. Ve a: https://makersuite.google.com/app/apikey
2. Inicia sesión con Google
3. Clic en "Get API key"
4. Copia la key

#### **2. Instalar dependencia:**
```bash
pip install google-generativeai
```

#### **3. Configurar en .env:**
```bash
GEMINI_API_KEY=AIzaSy_tu_api_key_aqui
AI_PROVIDER=gemini
USE_AI_FEEDBACK=true
```

---

### **Opción 3: OpenAI (GPT-4)**

#### **1. Obtener API Key:**
1. Ve a: https://platform.openai.com/api-keys
2. Crea cuenta y añade créditos ($5 mínimo)
3. Crea API key

#### **2. Instalar dependencia:**
```bash
pip install openai
```

#### **3. Configurar en .env:**
```bash
OPENAI_API_KEY=sk-proj-tu_api_key_aqui
AI_PROVIDER=openai
USE_AI_FEEDBACK=true
```

---

### **Opción 4: Anthropic Claude**

#### **1. Obtener API Key:**
1. Ve a: https://console.anthropic.com/
2. Crea cuenta y añade créditos
3. Crea API key

#### **2. Instalar dependencia:**
```bash
pip install anthropic
```

#### **3. Configurar en .env:**
```bash
ANTHROPIC_API_KEY=sk-ant-tu_api_key_aqui
AI_PROVIDER=claude
USE_AI_FEEDBACK=true
```

---

## 📝 Archivo .env Completo

Crea un archivo `.env` en la raíz del proyecto:

```bash
# =============================================================================
# CONFIGURACIÓN DE IA - Wine AI Prophet
# =============================================================================

# --- ELEGIR UN PROVEEDOR (solo uno) ---

# Opción 1: Groq (GRATIS Y RÁPIDO - RECOMENDADO)
GROQ_API_KEY=gsk_tu_clave_aqui
AI_PROVIDER=groq

# Opción 2: Gemini (GRATIS)
# GEMINI_API_KEY=AIzaSy_tu_clave_aqui
# AI_PROVIDER=gemini

# Opción 3: OpenAI (De pago)
# OPENAI_API_KEY=sk-proj_tu_clave_aqui
# AI_PROVIDER=openai

# Opción 4: Claude (De pago)
# ANTHROPIC_API_KEY=sk-ant_tu_clave_aqui
# AI_PROVIDER=claude

# --- CONFIGURACIÓN GENERAL ---
USE_AI_FEEDBACK=true        # true = usar IA, false = agente básico
AI_TEMPERATURE=0.7          # Creatividad (0.0 - 1.0)
AI_MAX_TOKENS=400           # Longitud máxima de respuesta
```

---

## 🔧 Configuración Avanzada

### **Cambiar entre modos:**

```bash
# Usar IA:
USE_AI_FEEDBACK=true

# Volver al agente básico (sin IA):
USE_AI_FEEDBACK=false
```

### **Ajustar creatividad:**

```bash
# Más técnico y preciso:
AI_TEMPERATURE=0.3

# Más creativo y variado:
AI_TEMPERATURE=0.9

# Balanceado (recomendado):
AI_TEMPERATURE=0.7
```

---

## ❓ Solución de Problemas

### **Error: "API key no encontrada"**
```bash
# Verifica que el archivo .env exista:
ls -la .env

# Verifica que la variable esté definida:
cat .env | grep API_KEY
```

### **Error: "No module named 'groq'"**
```bash
# Instala la dependencia:
pip install groq
```

### **Error: "Rate limit exceeded"**
- **Groq:** Espera 1 minuto (límite: 30 req/min gratis)
- **Gemini:** Espera 1 minuto (límite: 60 req/min gratis)
- **OpenAI/Claude:** Añade más créditos a tu cuenta

### **Feedback muy genérico**
```bash
# Aumenta la temperatura:
AI_TEMPERATURE=0.8

# O usa un modelo más potente:
# Groq: llama-3.1-70b-versatile (por defecto)
# OpenAI: gpt-4 (en lugar de gpt-4o-mini)
```

---

## 💰 Costos Estimados

### **Por 1000 análisis:**

| Proveedor | Costo | Notas |
|-----------|-------|-------|
| **Groq** | $0.00 | ✅ Totalmente gratis |
| **Gemini** | $0.00 | ✅ Gratis hasta límites |
| **OpenAI (GPT-4o-mini)** | ~$0.10 | Input + Output |
| **OpenAI (GPT-4)** | ~$0.50 | Más caro pero mejor |
| **Claude (Haiku)** | ~$0.25 | Balanceado |

---

## 🎓 Ejemplo de Uso en Código

```python
from src.utils import generar_feedback_ia

# Generar feedback con IA
feedback = generar_feedback_ia(
    score=92.5,
    lime_weights=[('elegant', 0.45), ('rich', 0.38), ('complex', 0.29)],
    texto_original="This wine is elegant, rich and complex.",
    provider="groq"  # o "gemini", "openai", "claude"
)

print(feedback)
```

---

## ✅ Checklist de Configuración

- [ ] Elegir un proveedor (recomendado: Groq)
- [ ] Obtener API key del proveedor
- [ ] Copiar `.env.example` a `.env`
- [ ] Pegar la API key en `.env`
- [ ] Instalar dependencia (`pip install groq`)
- [ ] Ejecutar la app
- [ ] Probar con una reseña de vino

---

## 🚀 Recomendación Final

**Para proyectos académicos/demo:**
```bash
# Usa Groq (gratis, rápido, buena calidad)
GROQ_API_KEY=tu_key
AI_PROVIDER=groq
```

**Para producción:**
```bash
# Usa OpenAI GPT-4 o Claude
OPENAI_API_KEY=tu_key
AI_PROVIDER=openai
```

---

## 📞 Soporte

Si tienes problemas, verifica:
1. El archivo `.env` existe y está en la raíz del proyecto
2. La API key es correcta (sin espacios extras)
3. Tienes internet (las APIs son externas)
4. Instalaste la dependencia correcta (`pip install groq`)

---

**🎉 ¡Listo! Ahora tienes feedback de nivel profesional generado por IA.**
