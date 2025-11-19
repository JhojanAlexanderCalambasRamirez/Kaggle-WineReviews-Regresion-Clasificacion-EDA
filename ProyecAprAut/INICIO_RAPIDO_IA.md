# ⚡ Inicio Rápido - Feedback con IA (3 minutos)

## 🎯 Objetivo

Configurar feedback inteligente con **Groq (GRATIS)** en 3 pasos.

---

## 📝 Paso 1: Obtener API Key (1 min)

1. Ve a: **https://console.groq.com/keys**
2. Crea cuenta (botón "Sign Up" - gratis)
3. Clic en **"Create API Key"**
4. Copia la clave (empieza con `gsk_...`)

---

## 🔧 Paso 2: Configurar (1 min)

### **Crear archivo .env:**

```bash
# En la raíz del proyecto (ProyecAprAut/), crea .env:
GROQ_API_KEY=gsk_PEGA_TU_CLAVE_AQUI
AI_PROVIDER=groq
USE_AI_FEEDBACK=true
```

### **O copiar el ejemplo:**

```bash
# Windows PowerShell:
cp .env.example .env

# Luego edita .env y pega tu clave
```

---

## 📦 Paso 3: Instalar e Iniciar (1 min)

```bash
# Instalar Groq:
pip install groq

# Ejecutar la app:
python src/gui/wine_ai_prophet.py
```

---

## ✅ ¡Listo!

Ahora cuando analices un vino, recibirás feedback profesional generado por IA:

### **Prueba con esto:**
```
This wine is elegant, complex and has rich tannins with a long finish.
```

### **Recibirás algo como:**
```
Este vino presenta características excepcionales (puntuación: 92.5/100).

Su perfil aromático, descrito como "elegante" y "complejo", sugiere una
elaboración cuidadosa. La presencia de taninos ricos indica madurez fenólica
óptima, típica de vinos de guarda con potencial de envejecimiento.

El final prolongado confirma la calidad superior, mostrando persistencia
aromática que es característica de vinos premium...

Recomendación: Temperatura de servicio 16-18°C. Ideal para carnes rojas.
```

---

## 🔄 Desactivar IA (volver al modo básico)

```bash
# En .env:
USE_AI_FEEDBACK=false
```

---

## ❓ Problemas Comunes

| Error | Solución |
|-------|----------|
| "API key no encontrada" | Verifica que `.env` existe y tiene `GROQ_API_KEY=...` |
| "No module named 'groq'" | Ejecuta `pip install groq` |
| "Rate limit exceeded" | Espera 1 minuto (límite: 30 req/min gratis) |

---

## 📚 Más Opciones

- **Gemini (Google):** También gratis → Ver [CONFIGURACION_IA.md](docs/CONFIGURACION_IA.md)
- **OpenAI (GPT-4):** Mejor calidad, de pago → Ver documentación
- **Claude (Anthropic):** Muy bueno, de pago → Ver documentación

---

**🎉 ¡Disfruta de feedback profesional con IA!**
