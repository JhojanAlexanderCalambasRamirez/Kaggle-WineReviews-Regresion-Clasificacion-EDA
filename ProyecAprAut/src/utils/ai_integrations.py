"""
Integraciones con APIs de IA para Feedback Avanzado
===================================================
Soporta múltiples proveedores: OpenAI, Gemini, Claude, Groq
"""
import os
import json
from typing import Optional, List, Tuple


class AIFeedbackGenerator:
    """
    Generador de feedback usando APIs de IA.
    Convierte datos técnicos en análisis profesional de sommelier.
    """

    def __init__(self, provider: str = "openai", api_key: Optional[str] = None):
        """
        Inicializa el generador con el proveedor especificado.

        Args:
            provider: "openai", "gemini", "claude", o "groq"
            api_key: Clave API (opcional, se lee de .env si no se provee)
        """
        self.provider = provider.lower()
        self.api_key = api_key or self._get_api_key()
        self.client = self._init_client()

    def _get_api_key(self) -> str:
        """Obtiene la API key desde variables de entorno"""
        key_map = {
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "groq": "GROQ_API_KEY"
        }

        env_var = key_map.get(self.provider)
        if not env_var:
            raise ValueError(f"Proveedor no soportado: {self.provider}")

        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(
                f"API key no encontrada. Define {env_var} en tu archivo .env"
            )

        return api_key

    def _init_client(self):
        """Inicializa el cliente según el proveedor"""
        if self.provider == "openai":
            try:
                from openai import OpenAI
                return OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("Instala openai: pip install openai")

        elif self.provider == "gemini":
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                return genai.GenerativeModel('gemini-pro')
            except ImportError:
                raise ImportError("Instala google-generativeai: pip install google-generativeai")

        elif self.provider == "claude":
            try:
                from anthropic import Anthropic
                return Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Instala anthropic: pip install anthropic")

        elif self.provider == "groq":
            try:
                from groq import Groq
                return Groq(api_key=self.api_key)
            except ImportError:
                raise ImportError("Instala groq: pip install groq")

        else:
            raise ValueError(f"Proveedor no soportado: {self.provider}")

    def _construir_prompt(
        self,
        score: float,
        lime_weights: List[Tuple[str, float]],
        texto_original: str
    ) -> str:
        """
        Construye el prompt para la IA.

        Args:
            score: Puntuación predicha (80-100)
            lime_weights: Lista de (palabra, peso)
            texto_original: Reseña original del usuario

        Returns:
            str: Prompt estructurado
        """
        # Separar factores
        positivos = [(p, w) for p, w in lime_weights if w > 0]
        negativos = [(p, w) for p, w in lime_weights if w < 0]

        # Formatear palabras clave
        pos_str = ", ".join([f"'{p}' ({w:.2f})" for p, w in positivos[:5]])
        neg_str = ", ".join([f"'{p}' ({w:.2f})" for p, w in negativos[:5]])

        prompt = f"""Eres un sommelier profesional. Analiza este vino de forma CONCISA y DIRECTA.

**DATOS:**
- Puntuación: {score:.1f}/100
- Reseña: "{texto_original}"
- Factores positivos: {pos_str if positivos else "Ninguno"}
- Factores negativos: {neg_str if negativos else "Ninguno"}

**INSTRUCCIONES ESTRICTAS:**
1. Escribe SOLO 2 párrafos cortos (máximo 120 palabras total)
2. Párrafo 1: Evaluación general + principales características identificadas
3. Párrafo 2: Recomendación breve (maridaje o momento de consumo)
4. Sé directo, sin rodeos ni información excesiva
5. NO uses emojis
6. Lenguaje profesional pero accesible

Genera el análisis CONCISO ahora:"""

        return prompt

    def generar_feedback(
        self,
        score: float,
        lime_weights: List[Tuple[str, float]],
        texto_original: str
    ) -> str:
        """
        Genera feedback profesional usando IA.

        Args:
            score: Puntuación del vino
            lime_weights: Pesos LIME
            texto_original: Reseña original

        Returns:
            str: Feedback profesional generado por IA
        """
        prompt = self._construir_prompt(score, lime_weights, texto_original)

        try:
            if self.provider == "openai":
                return self._generar_openai(prompt)
            elif self.provider == "gemini":
                return self._generar_gemini(prompt)
            elif self.provider == "claude":
                return self._generar_claude(prompt)
            elif self.provider == "groq":
                return self._generar_groq(prompt)
        except Exception as e:
            return self._feedback_fallback(score, lime_weights, str(e))

    def _generar_openai(self, prompt: str) -> str:
        """Genera con OpenAI GPT"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # Más económico que gpt-4
            messages=[
                {"role": "system", "content": "Eres un sommelier profesional experto."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()

    def _generar_gemini(self, prompt: str) -> str:
        """Genera con Google Gemini"""
        response = self.client.generate_content(prompt)
        return response.text.strip()

    def _generar_claude(self, prompt: str) -> str:
        """Genera con Anthropic Claude"""
        response = self.client.messages.create(
            model="claude-3-haiku-20240307",  # Más económico
            max_tokens=400,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text.strip()

    def _generar_groq(self, prompt: str) -> str:
        """Genera con Groq (Llama, Mixtral gratis)"""
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Modelo actualizado (Nov 2024)
            messages=[
                {"role": "system", "content": "Eres un sommelier profesional experto."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()

    def _feedback_fallback(self, score: float, lime_weights: List, error: str) -> str:
        """Feedback básico si falla la API"""
        positivos = [p for p, w in lime_weights if w > 0]
        negativos = [p for p, w in lime_weights if w < 0]

        texto = f"**Análisis del Vino (Puntuación: {score:.1f}/100)**\n\n"

        if score >= 90:
            texto += "Este vino ha sido clasificado como excepcional. "
        elif score >= 85:
            texto += "Este vino presenta características muy positivas. "
        else:
            texto += "Este vino muestra un perfil estándar. "

        if positivos:
            texto += f"Los descriptores positivos identificados ({', '.join(positivos[:3])}) "
            texto += "indican cualidades que elevan su calidad. "

        if negativos:
            texto += f"Sin embargo, ciertos aspectos ({', '.join(negativos[:2])}) "
            texto += "podrían afectar la percepción global. "

        texto += f"\n\n*Nota: Error conectando con API de IA ({self.provider}): {error[:100]}*"

        return texto


# Función auxiliar para uso simple
def generar_feedback_ia(
    score: float,
    lime_weights: List[Tuple[str, float]],
    texto_original: str,
    provider: str = "groq"  # Por defecto Groq (es gratis)
) -> str:
    """
    Función auxiliar para generar feedback rápidamente.

    Args:
        score: Puntuación del vino
        lime_weights: Pesos LIME
        texto_original: Reseña original
        provider: "openai", "gemini", "claude", "groq"

    Returns:
        str: Feedback profesional
    """
    try:
        generator = AIFeedbackGenerator(provider=provider)
        return generator.generar_feedback(score, lime_weights, texto_original)
    except Exception as e:
        # Fallback simple si no hay API configurada
        return f"""**Análisis del Vino (Puntuación: {score:.1f}/100)**

Este vino ha sido analizado mediante inteligencia artificial, obteniendo una puntuación de {score:.1f} puntos.

Para obtener análisis más detallados, configura una API key en el archivo .env

Error: {str(e)[:100]}"""
