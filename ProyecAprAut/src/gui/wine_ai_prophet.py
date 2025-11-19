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
        self.geometry("1100x750")
        self.resizable(False, False)

        # Inicializar agente sommelier
        self.agent = SommelierAgent()

        # Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Crear UI
        self._crear_header()
        self._crear_pestanas()
        self._crear_footer()

        # Estado
        self.modelo = None
        self._cargar_modelo_inicial()

    # =========================================================================
    # COMPONENTES UI
    # =========================================================================

    def _crear_header(self):
        """Crea el header superior con título y botón de ayuda"""
        header = ctk.CTkFrame(self, fg_color=("#E8F4F8", "#1a1a1a"), height=80)
        header.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header.grid_propagate(False)

        # Contenedor interno
        container = ctk.CTkFrame(header, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=30, pady=15)

        # Título con icono
        title_frame = ctk.CTkFrame(container, fg_color="transparent")
        title_frame.pack(side="left")

        ctk.CTkLabel(
            title_frame,
            text="Wine AI Prophet",
            font=("Segoe UI", 32, "bold"),
            text_color=("#1a1a1a", "#FFFFFF")
        ).pack(side="left")

        ctk.CTkLabel(
            title_frame,
            text="  Sistema Inteligente de Análisis de Vinos",
            font=("Segoe UI", 13),
            text_color=("#555555", "#AAAAAA")
        ).pack(side="left", padx=(10, 0))

        # Botones de acción
        btn_frame = ctk.CTkFrame(container, fg_color="transparent")
        btn_frame.pack(side="right")

        ctk.CTkButton(
            btn_frame,
            text="Acerca de",
            width=120,
            height=36,
            font=("Segoe UI", 13),
            fg_color=("#3498DB", "#2980B9"),
            hover_color=("#2980B9", "#1F618D"),
            corner_radius=8,
            command=self._mostrar_ayuda
        ).pack(side="right", padx=5)

    def _crear_footer(self):
        """Crea el footer con información del estado"""
        footer = ctk.CTkFrame(self, fg_color=("#F5F5F5", "#1a1a1a"), height=35)
        footer.grid(row=2, column=0, sticky="ew", padx=0, pady=0)
        footer.grid_propagate(False)

        self.lbl_estado = ctk.CTkLabel(
            footer,
            text="Estado: Modelo cargado y listo",
            font=("Segoe UI", 11),
            text_color=("#666666", "#999999")
        )
        self.lbl_estado.pack(side="left", padx=20)

        ctk.CTkLabel(
            footer,
            text=f"IA: {'Activada (' + AI_PROVIDER.title() + ')' if USE_AI_FEEDBACK else 'Desactivada'}",
            font=("Segoe UI", 11),
            text_color=("#27AE60" if USE_AI_FEEDBACK else "#95A5A6", "#27AE60" if USE_AI_FEEDBACK else "#95A5A6")
        ).pack(side="right", padx=20)

    def _crear_pestanas(self):
        """Crea las pestañas principales"""
        self.tabview = ctk.CTkTabview(self, width=1100, height=635)
        self.tabview.grid(row=1, column=0, padx=0, pady=0)

        # Configurar pestañas
        self.tabview._segmented_button.configure(
            font=("Segoe UI", 14, "bold"),
            height=45
        )

        self.tab_prediccion = self.tabview.add("  Análisis de Vino  ")
        self.tab_entrenamiento = self.tabview.add("  Entrenamiento  ")

        self._setup_prediccion()
        self._setup_entrenamiento()

    def _setup_prediccion(self):
        """Configura la pestaña de predicción con diseño mejorado"""
        frame = self.tab_prediccion

        # Layout principal: izquierda (input) y derecha (resultados)
        main_container = ctk.CTkFrame(frame, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=25, pady=20)

        # Panel izquierdo: INPUT
        left_panel = ctk.CTkFrame(main_container, fg_color=("#FFFFFF", "#2b2b2b"), corner_radius=12)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 15))

        # Título del panel
        ctk.CTkLabel(
            left_panel,
            text="Ingrese la Reseña del Vino",
            font=("Segoe UI", 18, "bold"),
            anchor="w"
        ).pack(anchor="w", padx=25, pady=(20, 5))

        ctk.CTkLabel(
            left_panel,
            text="Escriba una descripción en inglés del vino que desea analizar",
            font=("Segoe UI", 12),
            text_color=("#666666", "#AAAAAA"),
            anchor="w"
        ).pack(anchor="w", padx=25, pady=(0, 15))

        # Input de texto con placeholder
        self.input_text = ctk.CTkTextbox(
            left_panel,
            height=180,
            font=("Segoe UI", 14),
            corner_radius=8,
            border_width=2,
            border_color=("#E0E0E0", "#404040")
        )
        self.input_text.pack(fill="x", padx=25, pady=(0, 15))

        # Placeholder text
        placeholder = "Ejemplo:\nThis wine is elegant and complex with rich tannins, dark berry flavors and a long, smooth finish. The oak aging adds subtle vanilla notes."
        self.input_text.insert("0.0", placeholder)
        self.input_text.bind("<FocusIn>", lambda e: self._clear_placeholder())
        self.input_text.bind("<FocusOut>", lambda e: self._restore_placeholder())
        self._is_placeholder = True

        # Botón de predicción
        self.btn_predecir = ctk.CTkButton(
            left_panel,
            text="Analizar Vino",
            height=50,
            font=("Segoe UI", 16, "bold"),
            fg_color=("#27AE60", "#229954"),
            hover_color=("#229954", "#1E8449"),
            corner_radius=10,
            command=self._iniciar_prediccion
        )
        self.btn_predecir.pack(fill="x", padx=25, pady=(0, 20))

        # Ejemplos rápidos
        ctk.CTkLabel(
            left_panel,
            text="Ejemplos Rápidos:",
            font=("Segoe UI", 12, "bold"),
            anchor="w"
        ).pack(anchor="w", padx=25, pady=(5, 5))

        ejemplos_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        ejemplos_frame.pack(fill="x", padx=25, pady=(0, 20))

        ejemplos = [
            ("Alta Calidad", "This wine is elegant, complex and has a rich finish."),
            ("Media Calidad", "A decent wine with fruity notes and moderate acidity."),
            ("Baja Calidad", "This wine is flat, watery and lacks character.")
        ]

        for titulo, texto in ejemplos:
            btn = ctk.CTkButton(
                ejemplos_frame,
                text=titulo,
                height=28,
                font=("Segoe UI", 11),
                fg_color=("#ECF0F1", "#34495E"),
                text_color=("#2C3E50", "#ECF0F1"),
                hover_color=("#BDC3C7", "#2C3E50"),
                corner_radius=6,
                command=lambda t=texto: self._set_ejemplo(t)
            )
            btn.pack(fill="x", pady=2)

        # Panel derecho: RESULTADOS
        right_panel = ctk.CTkFrame(main_container, fg_color=("#FFFFFF", "#2b2b2b"), corner_radius=12)
        right_panel.pack(side="right", fill="both", expand=True)

        # Título del panel
        ctk.CTkLabel(
            right_panel,
            text="Resultados del Análisis",
            font=("Segoe UI", 18, "bold"),
            anchor="w"
        ).pack(anchor="w", padx=25, pady=(20, 15))

        # Score prominente
        score_container = ctk.CTkFrame(
            right_panel,
            fg_color=("#F8F9FA", "#1a1a1a"),
            corner_radius=10,
            height=140
        )
        score_container.pack(fill="x", padx=25, pady=(0, 15))
        score_container.pack_propagate(False)

        score_inner = ctk.CTkFrame(score_container, fg_color="transparent")
        score_inner.pack(expand=True)

        ctk.CTkLabel(
            score_inner,
            text="PUNTUACIÓN",
            font=("Segoe UI", 13, "bold"),
            text_color=("#7F8C8D", "#95A5A6")
        ).pack(pady=(10, 0))

        self.lbl_score = ctk.CTkLabel(
            score_inner,
            text="--",
            font=("Segoe UI", 72, "bold"),
            text_color=("#95A5A6", "#7F8C8D")
        )
        self.lbl_score.pack(pady=(0, 0))

        self.lbl_msg_score = ctk.CTkLabel(
            score_inner,
            text="Esperando análisis...",
            font=("Segoe UI", 13),
            text_color=("#7F8C8D", "#95A5A6")
        )
        self.lbl_msg_score.pack(pady=(0, 10))

        # Análisis del sommelier
        ctk.CTkLabel(
            right_panel,
            text="Análisis del Sommelier",
            font=("Segoe UI", 15, "bold"),
            anchor="w"
        ).pack(anchor="w", padx=25, pady=(0, 8))

        self.txt_narrativa = ctk.CTkTextbox(
            right_panel,
            font=("Segoe UI", 13),
            state="disabled",
            wrap="word",
            corner_radius=8,
            border_width=0,
            fg_color=("#F8F9FA", "#1a1a1a")
        )
        self.txt_narrativa.pack(fill="both", expand=True, padx=25, pady=(0, 20))

    def _setup_entrenamiento(self):
        """Configura la pestaña de entrenamiento con diseño mejorado"""
        frame = self.tab_entrenamiento

        # Contenedor principal
        main = ctk.CTkFrame(frame, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=40, pady=30)

        # Encabezado
        header_frame = ctk.CTkFrame(main, fg_color=("#FFFFFF", "#2b2b2b"), corner_radius=12)
        header_frame.pack(fill="x", pady=(0, 20))

        ctk.CTkLabel(
            header_frame,
            text="Entrenamiento del Modelo",
            font=("Segoe UI", 22, "bold")
        ).pack(pady=(25, 10))

        info = (
            "Pipeline de Machine Learning: Carga de datos → Limpieza NLP → Vectorización TF-IDF\n"
            "→ Entrenamiento de Red Neuronal MLP → Evaluación y Serialización"
        )
        ctk.CTkLabel(
            header_frame,
            text=info,
            justify="center",
            font=("Segoe UI", 12),
            text_color=("#666666", "#AAAAAA")
        ).pack(pady=(0, 25))

        # Información del modelo actual
        info_frame = ctk.CTkFrame(main, fg_color=("#E8F4F8", "#1a1a2e"), corner_radius=10)
        info_frame.pack(fill="x", pady=(0, 20))

        info_grid = ctk.CTkFrame(info_frame, fg_color="transparent")
        info_grid.pack(padx=20, pady=15)

        specs = [
            ("Dataset", "~130,000 reseñas de vinos"),
            ("Algoritmo", "MLP Neural Network"),
            ("Features", "TF-IDF (3000 palabras)"),
            ("Preprocesamiento", "Lemmatization + Stopwords")
        ]

        for i, (label, value) in enumerate(specs):
            row = i // 2
            col = i % 2

            item = ctk.CTkFrame(info_grid, fg_color="transparent")
            item.grid(row=row, column=col, padx=30, pady=5, sticky="w")

            ctk.CTkLabel(
                item,
                text=f"{label}:",
                font=("Segoe UI", 12, "bold"),
                width=140,
                anchor="w"
            ).pack(side="left")

            ctk.CTkLabel(
                item,
                text=value,
                font=("Segoe UI", 12),
                anchor="w"
            ).pack(side="left")

        # Botón de entrenamiento
        self.btn_entrenar = ctk.CTkButton(
            main,
            text="Iniciar Entrenamiento",
            height=55,
            font=("Segoe UI", 17, "bold"),
            fg_color=("#9B59B6", "#8E44AD"),
            hover_color=("#8E44AD", "#732D91"),
            corner_radius=10,
            command=self._iniciar_entrenamiento
        )
        self.btn_entrenar.pack(fill="x", pady=(0, 20))

        # Consola de progreso
        console_label = ctk.CTkLabel(
            main,
            text="Registro de Actividad:",
            font=("Segoe UI", 13, "bold"),
            anchor="w"
        )
        console_label.pack(fill="x", pady=(0, 8))

        self.console = ctk.CTkTextbox(
            main,
            height=200,
            font=("Consolas", 11),
            corner_radius=8,
            border_width=0,
            fg_color=("#F8F9FA", "#0d1117")
        )
        self.console.pack(fill="both", expand=True)
        self.console.insert("0.0", "Esperando inicio del entrenamiento...\n")

    # =========================================================================
    # HELPERS UI
    # =========================================================================

    def _clear_placeholder(self):
        """Limpia el placeholder al enfocar"""
        if self._is_placeholder:
            self.input_text.delete("0.0", "end")
            self._is_placeholder = False

    def _restore_placeholder(self):
        """Restaura el placeholder si está vacío"""
        if not self.input_text.get("0.0", "end").strip():
            placeholder = "Ejemplo:\nThis wine is elegant and complex with rich tannins, dark berry flavors and a long, smooth finish."
            self.input_text.insert("0.0", placeholder)
            self._is_placeholder = True

    def _set_ejemplo(self, texto):
        """Establece un ejemplo en el input"""
        self.input_text.delete("0.0", "end")
        self.input_text.insert("0.0", texto)
        self._is_placeholder = False

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
                self._actualizar_estado("Modelo cargado correctamente", "#27AE60")
            except Exception as e:
                print(f"✗ Error cargando modelo: {e}")
                self.btn_predecir.configure(state="disabled")
                self._actualizar_estado("Error al cargar modelo", "#E74C3C")
        else:
            self.btn_predecir.configure(
                state="disabled",
                text="Modelo No Disponible",
                fg_color="#95A5A6"
            )
            self._actualizar_estado("Modelo no encontrado - Ir a Entrenamiento", "#E67E22")

    def _actualizar_estado(self, mensaje, color="#666666"):
        """Actualiza el mensaje de estado en el footer"""
        self.lbl_estado.configure(text=f"Estado: {mensaje}", text_color=color)

    def _log(self, mensaje):
        """Escribe en la consola de entrenamiento"""
        self.console.insert("end", mensaje + "\n")
        self.console.see("end")

    # =========================================================================
    # ENTRENAMIENTO
    # =========================================================================

    def _iniciar_entrenamiento(self):
        """Inicia el entrenamiento en un hilo separado"""
        self.btn_entrenar.configure(state="disabled", text="Entrenando...", fg_color="#95A5A6")
        self.console.delete("0.0", "end")
        self._actualizar_estado("Entrenamiento en progreso...", "#E67E22")
        threading.Thread(target=self._proceso_entrenamiento, daemon=True).start()

    def _proceso_entrenamiento(self):
        """Pipeline completo de entrenamiento"""
        self._log("="*60)
        self._log("INICIANDO PIPELINE DE ENTRENAMIENTO")
        self._log("="*60)

        # Verificar dataset
        if not os.path.exists(DATASET_130K):
            self._log(f"\n ERROR: Dataset no encontrado")
            self._log(f"Ruta esperada: {DATASET_130K}")
            self.btn_entrenar.configure(state="normal", text="Reintentar", fg_color="#E74C3C")
            self._actualizar_estado("Error: Dataset no encontrado", "#E74C3C")
            return

        try:
            # 1. Cargar datos
            self._log("\n[1/7] Cargando dataset...")
            t0 = time.time()
            df = pd.read_csv(DATASET_130K, usecols=['description', 'points'])
            df = df.dropna().drop_duplicates()
            self._log(f"  Reseñas cargadas: {len(df):,}")
            self._log(f"  Tiempo: {time.time()-t0:.2f}s")

            # 2. Preprocesar
            self._log("\n[2/7] Aplicando limpieza NLP (Lemmatization)...")
            t0 = time.time()
            df['clean'] = df['description'].apply(limpiar_texto)
            self._log(f"  Procesado en: {time.time()-t0:.2f}s")

            # 3. Split
            self._log("\n[3/7] División Train/Test (80/20)...")
            X_train, X_test, y_train, y_test = train_test_split(
                df['clean'], df['points'],
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE
            )
            self._log(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

            # 4. Construir pipeline
            self._log("\n[4/7] Configurando pipeline (TF-IDF + MLP)...")
            pipeline = make_pipeline(
                TfidfVectorizer(max_features=TFIDF_MAX_FEATURES),
                MLPRegressor(
                    hidden_layer_sizes=MLP_HIDDEN_LAYERS,
                    max_iter=MLP_MAX_ITER,
                    random_state=MLP_RANDOM_STATE
                )
            )
            self._log(f"  Arquitectura: Input({TFIDF_MAX_FEATURES}) → {MLP_HIDDEN_LAYERS} → Output(1)")

            # 5. Entrenar
            self._log("\n[5/7] Entrenando Red Neuronal MLP...")
            self._log("  (Esto puede tardar varios minutos...)")
            t0 = time.time()
            pipeline.fit(X_train, y_train)
            tiempo_entrenamiento = time.time()-t0
            self._log(f"  Convergencia alcanzada en: {tiempo_entrenamiento:.2f}s")

            # 6. Evaluar
            self._log("\n[6/7] Evaluando rendimiento...")
            preds = pipeline.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            self._log(f"  MAE (Error Absoluto Medio): {mae:.4f} puntos")

            if mae < 1.5:
                self._log("  Rendimiento: EXCELENTE")
            elif mae < 2.0:
                self._log("  Rendimiento: BUENO")
            else:
                self._log("  Rendimiento: ACEPTABLE")

            # 7. Guardar
            self._log("\n[7/7] Guardando modelo...")
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            joblib.dump(pipeline, MODEL_PATH)
            self.modelo = pipeline
            self._log(f"  Modelo guardado en: {MODEL_PATH}")

            self._log("\n" + "="*60)
            self._log("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
            self._log("="*60)

            # Actualizar UI
            self.btn_predecir.configure(
                state="normal",
                text="Analizar Vino",
                fg_color=("#27AE60", "#229954")
            )
            self.btn_entrenar.configure(
                state="normal",
                text="Entrenamiento Completado",
                fg_color="#27AE60"
            )
            self._actualizar_estado("Modelo entrenado correctamente", "#27AE60")

        except Exception as e:
            self._log(f"\n ERROR CRÍTICO:")
            self._log(f"  {str(e)}")
            self.btn_entrenar.configure(state="normal", text="Reintentar", fg_color="#E74C3C")
            self._actualizar_estado("Error durante entrenamiento", "#E74C3C")

    # =========================================================================
    # PREDICCIÓN
    # =========================================================================

    def _iniciar_prediccion(self):
        """Inicia la predicción en un hilo separado"""
        texto = self.input_text.get("0.0", "end").strip()

        if self._is_placeholder or len(texto) < 10:
            self._actualizar_narrativa("Por favor, ingrese una reseña válida del vino.")
            return

        # Preparar UI
        self.btn_predecir.configure(state="disabled", text="Analizando...", fg_color="#95A5A6")
        self.lbl_score.configure(text="...")
        self._actualizar_narrativa("El sommelier está analizando su vino...\n\nPor favor espere.")
        self._actualizar_estado("Analizando vino...", "#E67E22")

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
            self._actualizar_estado("Análisis completado", "#27AE60")

        except Exception as e:
            self._actualizar_narrativa(f"Error durante el análisis:\n\n{str(e)}")
            self._actualizar_estado("Error en análisis", "#E74C3C")
        finally:
            self.btn_predecir.configure(
                state="normal",
                text="Analizar Vino",
                fg_color=("#27AE60", "#229954")
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
        ventana.geometry("550x700")
        ventana.resizable(False, False)
        ventana.attributes("-topmost", True)

        # Encabezado
        header = ctk.CTkFrame(ventana, fg_color=("#3498DB", "#2980B9"), height=100)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="Wine AI Prophet",
            font=("Segoe UI", 28, "bold"),
            text_color="#FFFFFF"
        ).pack(pady=(25, 5))

        ctk.CTkLabel(
            header,
            text="Sistema Inteligente de Análisis de Vinos",
            font=("Segoe UI", 13),
            text_color="#ECF0F1"
        ).pack()

        # Contenido
        content = ctk.CTkFrame(ventana, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=30, pady=25)

        # Descripción
        desc = (
            "Sistema de predicción de calidad de vinos mediante Procesamiento\n"
            "de Lenguaje Natural (NLP) y Redes Neuronales Artificiales.\n\n"
            "Entrenado con aproximadamente 130,000 reseñas profesionales\n"
            "de sommeliers. Utiliza análisis LIME para explicabilidad y puede\n"
            "integrar múltiples APIs de IA para feedback profesional."
        )
        ctk.CTkLabel(
            content,
            text=desc,
            justify="center",
            font=("Segoe UI", 12)
        ).pack(pady=15)

        # Tecnologías
        tech_frame = ctk.CTkFrame(content, fg_color=("#F8F9FA", "#2b2b2b"), corner_radius=10)
        tech_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            tech_frame,
            text="Tecnologías",
            font=("Segoe UI", 14, "bold")
        ).pack(pady=(15, 10))

        techs = "Python • Scikit-Learn • NLTK • TF-IDF\nCustomTkinter • LIME • Groq API"
        ctk.CTkLabel(
            tech_frame,
            text=techs,
            font=("Segoe UI", 11),
            justify="center"
        ).pack(pady=(0, 15))

        # Créditos
        ctk.CTkLabel(
            content,
            text="Equipo de Desarrollo",
            font=("Segoe UI", 15, "bold")
        ).pack(pady=(20, 10))

        frame_team = ctk.CTkFrame(content, fg_color=("#F8F9FA", "#2b2b2b"), corner_radius=10)
        frame_team.pack(fill="x")

        estudiantes = [
            ("Oscar Portela", "22507314"),
            ("Jorge Fong", "2205016"),
            ("Jhojan Alexander Calambas Ramirez", "2190555"),
            ("Angelo Parra Cortez", "22506988"),
            ("Juan Sebastian Rodriguez", "2195060")
        ]

        for nombre, codigo in estudiantes:
            fila = ctk.CTkFrame(frame_team, fg_color="transparent")
            fila.pack(fill="x", pady=4, padx=15)
            ctk.CTkLabel(
                fila,
                text=f"• {nombre}",
                font=("Segoe UI", 12),
                width=300,
                anchor="w"
            ).pack(side="left")
            ctk.CTkLabel(
                fila,
                text=codigo,
                font=("Segoe UI", 12, "bold"),
                text_color=("#27AE60", "#2ECC71")
            ).pack(side="right")

        # Botón cerrar
        ctk.CTkButton(
            content,
            text="Cerrar",
            command=ventana.destroy,
            fg_color=("#E74C3C", "#C0392B"),
            hover_color=("#C0392B", "#A93226"),
            corner_radius=8,
            height=40,
            font=("Segoe UI", 13)
        ).pack(pady=(20, 0))


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
def main():
    """Función principal"""
    # Verificar recursos NLTK al inicio
    print("Verificando diccionarios de lenguaje...")
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        print("✓ Recursos NLTK disponibles")
    except LookupError:
        print("Descargando recursos NLTK...")
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

    app = WineAIApp()
    app.mainloop()


if __name__ == "__main__":
    main()
