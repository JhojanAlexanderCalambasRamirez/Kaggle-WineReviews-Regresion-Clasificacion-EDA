#!/usr/bin/env python3
"""
Wine AI Prophet - Interfaz Gráfica Principal
============================================
Sistema de predicción de calidad de vinos con agente sommelier inteligente.

Autores:
    - Oscar Portela (22507314)
    - Jorge Fong (2205016)
    - Jhojan Alexander Calambas Ramirez (2190555)
    - Angelo Parra Cortez (22506988)
    - Juan Sebastian Rodriguez (2195060)
"""
import sys
import os

# Agregar paths para imports relativos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import customtkinter as ctk
import threading
import joblib
import pandas as pd
import time
import warnings

# Imports del proyecto
from config.settings import *
from src.utils import limpiar_texto, lime_wrapper, SommelierAgent, generar_feedback_ia

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from lime.lime_text import LimeTextExplainer

# Configuración
warnings.filterwarnings('ignore')
ctk.set_appearance_mode(APP_MODE)
ctk.set_default_color_theme(APP_THEME)


class WineAIApp(ctk.CTk):
    """Aplicación principal con agente sommelier integrado"""

    def __init__(self):
        super().__init__()

        # Configuración ventana
        self.title(APP_TITLE)
        self.geometry(APP_GEOMETRY)
        self.resizable(False, False)

        # Inicializar agente sommelier
        self.agent = SommelierAgent()

        # Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Crear UI
        self._crear_header()
        self._crear_pestanas()

        # Estado
        self.modelo = None
        self._cargar_modelo_inicial()

    # =========================================================================
    # COMPONENTES UI
    # =========================================================================

    def _crear_header(self):
        """Crea el header superior con título y botón de ayuda"""
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(15, 0))

        ctk.CTkLabel(
            header,
            text="Wine Quality Predictor 🍷",
            font=("Roboto", 28, "bold")
        ).pack(side="left")

        ctk.CTkButton(
            header,
            text="ℹ️ AYUDA & CRÉDITOS",
            width=160,
            fg_color="#34495E",
            hover_color="#2C3E50",
            command=self._mostrar_ayuda
        ).pack(side="right")

    def _crear_pestanas(self):
        """Crea las pestañas principales"""
        self.tabview = ctk.CTkTabview(self, width=900, height=600)
        self.tabview.grid(row=1, column=0, padx=20, pady=10)

        self.tab_prediccion = self.tabview.add("🔮 PREDICCIÓN & SOMMELIER")
        self.tab_entrenamiento = self.tabview.add("⚙️ ENTRENAMIENTO")

        self._setup_prediccion()
        self._setup_entrenamiento()

    def _setup_prediccion(self):
        """Configura la pestaña de predicción"""
        frame = self.tab_prediccion

        # Input de texto
        ctk.CTkLabel(
            frame,
            text="Escribe la reseña del vino (en inglés):",
            font=("Roboto", 14)
        ).pack(anchor="w", padx=25, pady=(15, 5))

        self.input_text = ctk.CTkTextbox(frame, height=100, font=("Roboto", 14))
        self.input_text.pack(fill="x", padx=25, pady=5)

        # Botón de predicción
        self.btn_predecir = ctk.CTkButton(
            frame,
            text="✨ CONSULTAR AL AGENTE SOMMELIER",
            height=50,
            font=("Roboto", 15, "bold"),
            fg_color=COLOR_BOTON_PREDECIR,
            hover_color="#1E8449",
            command=self._iniciar_prediccion
        )
        self.btn_predecir.pack(pady=15)

        # Área de resultados
        self._crear_area_resultados(frame)

    def _crear_area_resultados(self, parent):
        """Crea el área de visualización de resultados"""
        res_frame = ctk.CTkFrame(parent, fg_color="transparent")
        res_frame.pack(fill="both", expand=True, padx=15, pady=5)

        # Panel izquierdo: Score
        left = ctk.CTkFrame(res_frame, width=240)
        left.pack(side="left", fill="y", padx=10)

        ctk.CTkLabel(left, text="PUNTUACIÓN", font=("Roboto", 14, "bold")).pack(pady=(30, 5))
        self.lbl_score = ctk.CTkLabel(left, text="--", font=("Roboto", 80, "bold"), text_color="#7F8C8D")
        self.lbl_score.pack(pady=10)
        self.lbl_msg_score = ctk.CTkLabel(left, text="Esperando cata...", font=("Roboto", 14))
        self.lbl_msg_score.pack()

        # Panel derecho: Narrativa
        right = ctk.CTkFrame(res_frame)
        right.pack(side="right", fill="both", expand=True, padx=10)

        ctk.CTkLabel(
            right,
            text="💬 ANÁLISIS DEL AGENTE SOMMELIER",
            font=("Roboto", 13, "bold")
        ).pack(pady=10)

        self.txt_narrativa = ctk.CTkTextbox(
            right,
            font=("Segoe UI", 14),
            state="disabled",
            wrap="word"
        )
        self.txt_narrativa.pack(fill="both", expand=True, padx=10, pady=10)

    def _setup_entrenamiento(self):
        """Configura la pestaña de entrenamiento"""
        frame = self.tab_entrenamiento

        ctk.CTkLabel(
            frame,
            text="Panel de Ingeniería de Modelo (MLOps)",
            font=("Roboto", 20, "bold")
        ).pack(pady=20)

        info = (
            "Pipeline completo: Carga CSV → Limpieza NLP → Vectorización TF-IDF\n"
            "→ Entrenamiento MLP → Serialización (.pkl)"
        )
        ctk.CTkLabel(frame, text=info, justify="center", text_color="#BDC3C7").pack(pady=10)

        self.btn_entrenar = ctk.CTkButton(
            frame,
            text=" INICIAR ENTRENAMIENTO",
            height=50,
            font=("Roboto", 16, "bold"),
            fg_color=COLOR_BOTON_ENTRENAR,
            hover_color="#732D91",
            command=self._iniciar_entrenamiento
        )
        self.btn_entrenar.pack(pady=20)

        ctk.CTkLabel(frame, text="Terminal de Progreso:", anchor="w").pack(fill="x", padx=40)
        self.console = ctk.CTkTextbox(frame, height=200, font=("Consolas", 12))
        self.console.pack(fill="x", padx=40, pady=5)

    # =========================================================================
    # LÓGICA DE NEGOCIO
    # =========================================================================

    def _cargar_modelo_inicial(self):
        """Intenta cargar el modelo al iniciar"""
        if os.path.exists(MODEL_PATH):
            try:
                self.modelo = joblib.load(MODEL_PATH)
                print(f"✓ Modelo cargado: {MODEL_PATH}")
                self.btn_predecir.configure(state="normal")
            except Exception as e:
                print(f"✗ Error cargando modelo: {e}")
                self.btn_predecir.configure(state="disabled")
        else:
            self.btn_predecir.configure(
                state="disabled",
                text=" MODELO NO ENCONTRADO (Ve a Entrenar)"
            )

    def _log(self, mensaje):
        """Escribe en la consola de entrenamiento"""
        self.console.insert("end", mensaje + "\n")
        self.console.see("end")

    # =========================================================================
    # ENTRENAMIENTO
    # =========================================================================

    def _iniciar_entrenamiento(self):
        """Inicia el entrenamiento en un hilo separado"""
        self.btn_entrenar.configure(state="disabled", text="⏳ Entrenando...")
        threading.Thread(target=self._proceso_entrenamiento, daemon=True).start()

    def _proceso_entrenamiento(self):
        """Pipeline completo de entrenamiento"""
        self._log("--- INICIANDO PIPELINE ---")

        # Verificar dataset
        if not os.path.exists(DATASET_130K):
            self._log(f" ERROR: Dataset no encontrado en {DATASET_130K}")
            self.btn_entrenar.configure(state="normal", text="REINTENTAR")
            return

        try:
            # 1. Cargar datos
            self._log("1. Cargando dataset...")
            t0 = time.time()
            df = pd.read_csv(DATASET_130K, usecols=['description', 'points'])
            df = df.dropna().drop_duplicates()
            self._log(f"   -> {len(df)} reseñas cargadas ({time.time()-t0:.2f}s)")
            # 2. Preprocesar
            self._log("2. Limpieza NLP...")
            t0 = time.time()
            df['clean'] = df['description'].apply(limpiar_texto)
            self._log(f"   -> Procesado en {time.time()-t0:.2f}s")

            # 3. Split
            self._log("3. División Train/Test...")
            X_train, X_test, y_train, y_test = train_test_split(
                df['clean'], df['points'],
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE
            )

            # 4. Construir pipeline
            self._log("4. Configurando Red Neuronal MLP...")
            pipeline = make_pipeline(
                TfidfVectorizer(max_features=TFIDF_MAX_FEATURES),
                MLPRegressor(
                    hidden_layer_sizes=MLP_HIDDEN_LAYERS,
                    max_iter=MLP_MAX_ITER,
                    random_state=MLP_RANDOM_STATE
                )
            )

            # 5. Entrenar
            self._log("5. Entrenando modelo...")
            t0 = time.time()
            pipeline.fit(X_train, y_train)
            self._log(f"   → Convergido en {time.time()-t0:.2f}s")

            # 6. Evaluar
            self._log("6. Evaluando precisión...")
            preds = pipeline.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            self._log(f"   -> MAE: {mae:.4f} puntos")

            # 7. Guardar
            self._log("7.Guardando modelo...")
            joblib.dump(pipeline, MODEL_PATH)
            self.modelo = pipeline

            self._log(f"¡ÉXITO! Modelo guardado en: {MODEL_PATH}")

            # Actualizar botones
            self.btn_predecir.configure(
                state="normal",
                text="✨ ANALIZAR RESEÑA"
            )
            self.btn_entrenar.configure(state="normal", text="✓ COMPLETADO")

        except Exception as e:
            self._log(f"❌ ERROR: {str(e)}")
            self.btn_entrenar.configure(state="normal", text="REINTENTAR")

    # =========================================================================
    # PREDICCIÓN
    # =========================================================================

    def _iniciar_prediccion(self):
        """Inicia la predicción en un hilo separado"""
        texto = self.input_text.get("0.0", "end").strip()

        if len(texto) < 10:
            return

        # Preparar UI
        self.btn_predecir.configure(state="disabled", text="⏳ Analizando...")
        self.lbl_score.configure(text="...")
        self._actualizar_narrativa("🤔 El agente está pensando...")

        threading.Thread(target=self._proceso_prediccion, args=(texto,), daemon=True).start()

    def _proceso_prediccion(self, texto):
        """Pipeline de predicción con agente sommelier"""
        try:
            # 1. Limpiar texto
            texto_limpio = limpiar_texto(texto)

            # 2. Predecir
            score = self.modelo.predict([texto_limpio])[0]

            # 3. Explicar con LIME
            explainer = LimeTextExplainer(verbose=False, random_state=LIME_RANDOM_STATE)
            exp = explainer.explain_instance(
                texto_limpio,
                lambda x: lime_wrapper(x, self.modelo),
                num_features=LIME_NUM_FEATURES,
                labels=[0]
            )

            # 4. Generar narrativa (con IA o agente básico)
            pesos_lime = exp.as_list(label=0)

            if USE_AI_FEEDBACK:
                # Usar IA para feedback avanzado
                try:
                    narrativa = generar_feedback_ia(
                        score=score,
                        lime_weights=pesos_lime,
                        texto_original=texto,
                        provider=AI_PROVIDER
                    )
                except Exception as e:
                    # Fallback al agente básico si falla la IA
                    print(f"Advertencia: Usando agente básico. Error IA: {e}")
                    narrativa = self.agent.generar_narrativa(score, pesos_lime)
            else:
                # Usar agente básico (sin IA)
                narrativa = self.agent.generar_narrativa(score, pesos_lime)

            # 5. Actualizar UI
            self._actualizar_resultados(score, narrativa)

        except Exception as e:
            self._actualizar_narrativa(f"❌ Error: {str(e)}")
        finally:
            self.btn_predecir.configure(
                state="normal",
                text="✨ CONSULTAR AL AGENTE SOMMELIER"
            )

    def _actualizar_resultados(self, score, narrativa):
        """Actualiza la UI con los resultados"""
        # Score y color
        self.lbl_score.configure(text=f"{score:.1f}")
        color, mensaje = self.agent.obtener_color_y_mensaje(score)
        self.lbl_score.configure(text_color=color)
        self.lbl_msg_score.configure(text=mensaje, text_color=color)

        # Narrativa
        self._actualizar_narrativa(narrativa)

    def _actualizar_narrativa(self, texto):
        """Actualiza el texto de la narrativa"""
        self.txt_narrativa.configure(state="normal")
        self.txt_narrativa.delete("0.0", "end")
        self.txt_narrativa.insert("0.0", texto)
        self.txt_narrativa.configure(state="disabled")

    # =========================================================================
    # VENTANAS EMERGENTES
    # =========================================================================

    def _mostrar_ayuda(self):
        """Muestra ventana de ayuda y créditos"""
        ventana = ctk.CTkToplevel(self)
        ventana.title("Acerca del Proyecto")
        ventana.geometry("500x650")
        ventana.resizable(False, False)
        ventana.attributes("-topmost", True)

        # Título
        ctk.CTkLabel(
            ventana,
            text="WINE AI PROPHET",
            font=("Roboto", 20, "bold"),
            text_color="#3B8ED0"
        ).pack(pady=(25, 10))

        # Descripción
        desc = (
            "Sistema de predicción de calidad de vinos mediante\n"
            "Procesamiento de Lenguaje Natural y Redes Neuronales.\n\n"
            "Entrenado con ~130,000 reseñas profesionales de sommeliers.\n"
            "Utiliza un Agente Experto para explicar las predicciones."
        )
        ctk.CTkLabel(ventana, text=desc, justify="center", font=("Roboto", 12)).pack(pady=10)

        # Créditos
        ctk.CTkLabel(
            ventana,
            text="👨‍💻 EQUIPO DE DESARROLLO",
            font=("Roboto", 16, "bold")
        ).pack(pady=(20, 10))

        frame_team = ctk.CTkFrame(ventana, fg_color="#2B2B2B")
        frame_team.pack(fill="x", padx=30, pady=5)

        estudiantes = [
            ("Oscar Portela", "22507314"),
            ("Jorge Fong", "2205016"),
            ("Jhojan Alexander Calambas Ramirez", "2190555"),
            ("Angelo Parra Cortez", "22506988"),
            ("Juan Sebastian Rodriguez", "2195060")
        ]

        for nombre, codigo in estudiantes:
            fila = ctk.CTkFrame(frame_team, fg_color="transparent")
            fila.pack(fill="x", pady=5, padx=10)
            ctk.CTkLabel(fila, text=f"• {nombre}", font=("Roboto", 13), width=250, anchor="w").pack(side="left")
            ctk.CTkLabel(fila, text=codigo, font=("Roboto", 13, "bold"), text_color="#2CC985").pack(side="right")

        ctk.CTkButton(
            ventana,
            text="Cerrar",
            command=ventana.destroy,
            fg_color="#C0392B",
            hover_color="#A93226"
        ).pack(pady=30)


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
def main():
    """Función principal"""
    app = WineAIApp()
    app.mainloop()


if __name__ == "__main__":
    main()
