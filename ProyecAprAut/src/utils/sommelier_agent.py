"""
Agente Sommelier Inteligente
=============================
Sistema basado en reglas que convierte análisis técnicos (LIME)
en narrativas expertas en lenguaje natural.
"""
import random


class SommelierAgent:
    """
    Agente que traduce pesos matemáticos de LIME a narrativa natural.
    Simula el estilo comunicativo de un sommelier profesional.
    """

    # Frases de introducción categorizadas por calidad
    FRASES_ALTA_CALIDAD = [
        "Este vino muestra cualidades excepcionales.",
        "Estamos ante una muestra de gran categoría.",
        "El perfil detectado sugiere un vino de alta gama.",
        "Una propuesta enológica fascinante."
    ]

    FRASES_CALIDAD_MEDIA = [
        "Es un vino sólido y correcto.",
        "Presenta un perfil estándar para su categoría.",
        "Un vino aceptable, aunque sin grandes complejidades."
    ]

    FRASES_BAJA_CALIDAD = [
        "Este vino presenta deficiencias notables.",
        "El perfil indica un vino de calidad inferior.",
        "Se detectaron características problemáticas en la cata.",
        "Carece de la estructura esperada."
    ]

    # Conectores para factores positivos y negativos
    CONECTORES_POSITIVOS = [
        "Destaca positivamente por su carácter",
        "El modelo valora la presencia de",
        "Gana puntos gracias a ser descrito como"
    ]

    CONECTORES_NEGATIVOS = [
        "Lamentablemente, se describe como",
        "Pierde puntos significativamente por ser",
        "Se penaliza la característica"
    ]

    def __init__(self):
        """Inicializa el agente sommelier"""
        pass

    def _seleccionar_introduccion(self, score):
        """
        Selecciona frase de introducción según el puntaje.

        Args:
            score (float): Puntaje del vino (80-100)

        Returns:
            str: Introducción apropiada
        """
        if score >= 90:
            frase = random.choice(self.FRASES_ALTA_CALIDAD)
            contexto = f"Con una proyección de {score:.1f} puntos, supera el umbral de la excelencia."
            emoji = "🏆"
        elif score >= 85:
            frase = random.choice(self.FRASES_CALIDAD_MEDIA)
            contexto = f"Su puntuación de {score:.1f} lo sitúa en un rango muy disfrutable."
            emoji = "👌"
        else:
            frase = random.choice(self.FRASES_BAJA_CALIDAD)
            contexto = f"Su calificación de {score:.1f} refleja carencias estructurales."
            emoji = "⚠️"

        return f"{emoji} {frase} {contexto}"

    def _analizar_factores(self, factores, tipo="positivo"):
        """
        Genera análisis de factores positivos o negativos.

        Args:
            factores (list): Lista de palabras clave
            tipo (str): "positivo" o "negativo"

        Returns:
            str: Análisis formateado
        """
        if not factores:
            return ""

        # Seleccionar top 3
        top_factores = factores[:3]

        # Formatear palabras
        palabras_fmt = ", ".join([f"'{p.upper()}'" for p in top_factores])

        # Seleccionar conector
        if tipo == "positivo":
            conector = random.choice(self.CONECTORES_POSITIVOS)
            emoji = "✅"
            titulo = "FORTALEZAS"
            explicacion = "Estos descriptores son típicos de vinos bien elaborados y elevan la percepción de calidad."
        else:
            conector = random.choice(self.CONECTORES_NEGATIVOS)
            emoji = "❌"
            titulo = "DEBILIDADES"
            explicacion = "En el lenguaje enológico, estos términos suelen asociarse con falta de balance o defectos sensoriales."

        return f"\n\n{emoji} {titulo}: {conector} {palabras_fmt}. {explicacion}"

    def generar_narrativa(self, score, lime_weights):
        """
        Genera narrativa completa a partir del score y pesos LIME.

        Args:
            score (float): Puntuación predicha
            lime_weights (list): Lista de tuplas (palabra, peso) de LIME

        Returns:
            str: Narrativa completa estilo sommelier

        Example:
            >>> agent = SommelierAgent()
            >>> weights = [('elegant', 0.45), ('rich', 0.38), ('flat', -0.32)]
            >>> narrativa = agent.generar_narrativa(92.5, weights)
        """
        # Separar factores por signo
        positivos = [palabra for palabra, peso in lime_weights if peso > 0]
        negativos = [palabra for palabra, peso in lime_weights if peso < 0]

        # Construir narrativa
        narrativa = self._seleccionar_introduccion(score)

        # Añadir análisis de fortalezas
        narrativa += self._analizar_factores(positivos, tipo="positivo")

        # Añadir análisis de debilidades
        narrativa += self._analizar_factores(negativos, tipo="negativo")

        # Conclusión si no hay factores claros
        if not positivos and not negativos:
            narrativa += ("\n\nℹ️ El modelo ha basado su decisión en la estructura global "
                         "del texto más que en palabras clave específicas aisladas.")

        return narrativa

    def obtener_color_y_mensaje(self, score):
        """
        Retorna color y mensaje según puntuación.

        Args:
            score (float): Puntaje del vino

        Returns:
            tuple: (color_hex, mensaje_str)
        """
        if score >= 90:
            return "#27AE60", "¡Excelente! 🏆"
        elif score >= 85:
            return "#F39C12", "Muy Bueno 👌"
        else:
            return "#E74C3C", "Regular/Bajo ⚠️"
