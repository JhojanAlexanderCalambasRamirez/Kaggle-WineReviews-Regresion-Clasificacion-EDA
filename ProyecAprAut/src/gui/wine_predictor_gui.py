# =============================================================================
# APP GRÁFICA "WINE AI" - VERSIÓN FINAL CON AYUDA Y CRÉDITOS
# =============================================================================
import customtkinter as ctk
import threading
import os
import joblib
import pandas as pd
import time
import re
import warnings

# Librerías de Lógica
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from lime.lime_text import LimeTextExplainer

# Configuración Inicial
warnings.filterwarnings('ignore')
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# Rutas y Constantes (dinámicas desde la ubicación del script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
CARPETA_SISTEMA = os.path.join(PROJECT_ROOT, "sistema_vino")
RUTA_MODELO = os.path.join(CARPETA_SISTEMA, "cerebro_vino.pkl")
RUTA_DATASET = os.path.join(PROJECT_ROOT, "data", "raw", "winemag-data-130k-v2.csv")

if not os.path.exists(CARPETA_SISTEMA):
    os.makedirs(CARPETA_SISTEMA)

# Recursos NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# --- LÓGICA DE LIMPIEZA (BACKEND) ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def limpiar_texto(texto):
    if not isinstance(texto, str): return ""
    texto = texto.lower()
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    tokens = [lemmatizer.lemmatize(t) for t in texto.split() if t not in stop_words]
    return " ".join(tokens)

def funcion_lime_wrapper(textos, modelo):
    return modelo.predict(textos).reshape(-1, 1)

# =============================================================================
# CLASE PRINCIPAL DE LA INTERFAZ
# =============================================================================
class WineApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configuración de la ventana
        self.title("Wine AI Prophet 🍷 - Proyecto Machine Learning")
        self.geometry("900x750")
        self.resizable(False, False)

        # Layout de Grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1) # El contenido se expande

        # --- HEADER (Título + Botón Ayuda) ---
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(10, 0))
        
        # Título Principal
        self.lbl_titulo = ctk.CTkLabel(self.header_frame, text="Wine Quality Predictor 🍷", 
                                     font=("Roboto", 28, "bold"))
        self.lbl_titulo.pack(side="left", pady=10)

        # Botón de Ayuda (A la derecha)
        self.btn_ayuda = ctk.CTkButton(self.header_frame, text="ℹ️ AYUDA & CRÉDITOS", width=150,
                                       fg_color="#5D6D7E", hover_color="#34495E",
                                       command=self.abrir_ventana_ayuda)
        self.btn_ayuda.pack(side="right", pady=10)

        # --- PESTAÑAS PRINCIPALES ---
        self.tabview = ctk.CTkTabview(self, width=860, height=600)
        self.tabview.grid(row=1, column=0, padx=20, pady=10)
        
        self.tab_prediccion = self.tabview.add("🔮 PREDICCIÓN")
        self.tab_entrenamiento = self.tabview.add("⚙️ ENTRENAMIENTO")

        self.setup_tab_prediccion()
        self.setup_tab_entrenamiento()

        # Variables de estado
        self.modelo = None
        self.cargar_modelo_inicio()

    # -------------------------------------------------------------------------
    # VENTANA EMERGENTE DE AYUDA (POP-UP)
    # -------------------------------------------------------------------------
    def abrir_ventana_ayuda(self):
        # Crear ventana secundaria (Toplevel)
        ventana_ayuda = ctk.CTkToplevel(self)
        ventana_ayuda.title("Acerca de Wine AI")
        ventana_ayuda.geometry("500x600")
        ventana_ayuda.resizable(False, False)
        
        # Hacer que la ventana sea modal (siempre enfrente)
        ventana_ayuda.attributes("-topmost", True)

        # Título
        ctk.CTkLabel(ventana_ayuda, text="INFORMACIÓN DEL PROYECTO", 
                     font=("Roboto", 20, "bold"), text_color="#3B8ED0").pack(pady=(20, 10))

        # Sección 1: Descripción
        ctk.CTkLabel(ventana_ayuda, text="¿QUÉ ES ESTA APP?", font=("Roboto", 14, "bold")).pack(pady=5)
        desc_text = (
            "Esta herramienta utiliza Inteligencia Artificial (Procesamiento\n"
            "de Lenguaje Natural) para predecir la calidad de un vino (80-100)\n"
            "basándose únicamente en la descripción textual del sommelier.\n"
            "Además, explica qué palabras influyeron en la decisión."
        )
        ctk.CTkLabel(ventana_ayuda, text=desc_text, justify="center", font=("Roboto", 12)).pack(pady=5)

        # Sección 2: Instrucciones
        ctk.CTkLabel(ventana_ayuda, text="¿CÓMO SE USA?", font=("Roboto", 14, "bold")).pack(pady=(15, 5))
        instr_text = (
            "1. Ve a la pestaña 'PREDICCIÓN'.\n"
            "2. Escribe una reseña de vino en INGLÉS.\n"
            "   (Ej: 'This wine is elegant, complex and rich')\n"
            "3. Haz clic en 'ANALIZAR CALIDAD'.\n"
            "4. Si no hay modelo, ve a 'ENTRENAMIENTO' primero."
        )
        ctk.CTkLabel(ventana_ayuda, text=instr_text, justify="left", font=("Roboto", 12), 
                     fg_color="#2b2b2b", corner_radius=8, padx=10, pady=10).pack(pady=5)

        # Sección 3: Creadores
        ctk.CTkLabel(ventana_ayuda, text="👨‍💻 EQUIPO DE DESARROLLO", font=("Roboto", 14, "bold")).pack(pady=(20, 10))
        
        # Marco para la lista de estudiantes
        frame_creadores = ctk.CTkFrame(ventana_ayuda, fg_color="transparent")
        frame_creadores.pack()

        lista_estudiantes = [
            ("Oscar Portela", "22507314"),
            ("Jorge Fong", "2205016"),
            ("Jhojan Alexander Calambas Ramirez", "2190555"),
            ("Angelo Parra Cortez", "22506988"),
            ("Juan Sebastian Rodriguez", "2195060")
        ]

        for nombre, codigo in lista_estudiantes:
            fila = ctk.CTkFrame(frame_creadores, fg_color="transparent")
            fila.pack(fill="x", pady=2)
            ctk.CTkLabel(fila, text=f"• {nombre}", font=("Roboto", 13), width=250, anchor="w").pack(side="left")
            ctk.CTkLabel(fila, text=f"ID: {codigo}", font=("Roboto", 13, "bold"), text_color="#2CC985").pack(side="right")

        # Botón Cerrar
        ctk.CTkButton(ventana_ayuda, text="Cerrar", command=ventana_ayuda.destroy, fg_color="#EB5757", hover_color="#C0392B").pack(pady=20)

    # -------------------------------------------------------------------------
    # PESTAÑA 1: PREDICCIÓN
    # -------------------------------------------------------------------------
    def setup_tab_prediccion(self):
        frame = self.tab_prediccion
        
        ctk.CTkLabel(frame, text="Ingresa la reseña del vino (en inglés):", font=("Roboto", 14)).pack(anchor="w", padx=20, pady=(10,0))
        self.input_text = ctk.CTkTextbox(frame, height=120, font=("Roboto", 14))
        self.input_text.pack(fill="x", padx=20, pady=(5, 15))

        self.btn_predecir = ctk.CTkButton(frame, text="✨ ANALIZAR CALIDAD", height=45, font=("Roboto", 15, "bold"),
                                          fg_color="#2CC985", hover_color="#229A65", command=self.iniciar_prediccion)
        self.btn_predecir.pack(pady=5)

        # ÁREA DE RESULTADOS
        res_frame = ctk.CTkFrame(frame, fg_color="transparent")
        res_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Columna Izquierda: SCORE
        left_frame = ctk.CTkFrame(res_frame, width=220)
        left_frame.pack(side="left", fill="y", padx=10)
        
        ctk.CTkLabel(left_frame, text="PUNTAJE", font=("Roboto", 14, "bold")).pack(pady=(30,5))
        self.lbl_score = ctk.CTkLabel(left_frame, text="--", font=("Roboto", 70, "bold"), text_color="#5D6D7E")
        self.lbl_score.pack(pady=10)
        self.lbl_msg_score = ctk.CTkLabel(left_frame, text="Esperando reseña...", font=("Roboto", 14))
        self.lbl_msg_score.pack()

        # Columna Derecha: EXPLICACIÓN
        right_frame = ctk.CTkFrame(res_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=10)

        ctk.CTkLabel(right_frame, text="🔍 ANÁLISIS DE PALABRAS CLAVE", font=("Roboto", 13, "bold")).pack(pady=5)
        self.txt_explicacion = ctk.CTkTextbox(right_frame, font=("Consolas", 14), state="disabled")
        self.txt_explicacion.pack(fill="both", expand=True, padx=5, pady=5)

    # -------------------------------------------------------------------------
    # PESTAÑA 2: ENTRENAMIENTO
    # -------------------------------------------------------------------------
    def setup_tab_entrenamiento(self):
        frame = self.tab_entrenamiento
        
        ctk.CTkLabel(frame, text="Panel de Entrenamiento del Modelo", font=("Roboto", 20, "bold")).pack(pady=15)
        
        info_text = (
            "Este módulo entrena una Red Neuronal (MLP) usando el dataset local.\n"
            "Pasos automáticos: Limpieza -> Lematización -> TF-IDF -> Entrenamiento.\n"
            "Tiempo estimado: 30 seg - 2 min."
        )
        ctk.CTkLabel(frame, text=info_text, font=("Roboto", 13), justify="center", text_color="#BDC3C7").pack(pady=10)

        self.btn_entrenar = ctk.CTkButton(frame, text="🚀 INICIAR ENTRENAMIENTO", height=50, 
                                          font=("Roboto", 16, "bold"), fg_color="#8E44AD", hover_color="#732D91",
                                          command=self.iniciar_entrenamiento_thread)
        self.btn_entrenar.pack(pady=15)

        ctk.CTkLabel(frame, text="Consola de Progreso:", anchor="w").pack(fill="x", padx=30)
        self.console_log = ctk.CTkTextbox(frame, height=220, font=("Consolas", 12))
        self.console_log.pack(fill="x", padx=30, pady=5)

    # -------------------------------------------------------------------------
    # LÓGICA DE NEGOCIO
    # -------------------------------------------------------------------------
    def log(self, mensaje):
        self.console_log.insert("end", mensaje + "\n")
        self.console_log.see("end")

    def cargar_modelo_inicio(self):
        if os.path.exists(RUTA_MODELO):
            try:
                self.modelo = joblib.load(RUTA_MODELO)
                self.btn_predecir.configure(state="normal")
                print("Modelo cargado.")
            except:
                self.btn_predecir.configure(state="disabled")
        else:
            self.btn_predecir.configure(state="disabled", text="⚠️ MODELO NO ENCONTRADO (Ve a Entrenar)")

    # HILO ENTRENAMIENTO
    def iniciar_entrenamiento_thread(self):
        self.btn_entrenar.configure(state="disabled", text="⏳ Entrenando... Espere")
        threading.Thread(target=self.proceso_entrenamiento, daemon=True).start()

    def proceso_entrenamiento(self):
        self.log("--- INICIANDO ---")
        if not os.path.exists(RUTA_DATASET):
            self.log(f"❌ ERROR: No encuentro {RUTA_DATASET}")
            self.btn_entrenar.configure(state="normal", text="INICIAR ENTRENAMIENTO")
            return

        try:
            self.log("1. Cargando dataset...")
            df = pd.read_csv(RUTA_DATASET, usecols=['description', 'points']).dropna().drop_duplicates()
            
            self.log(f"2. Procesando {len(df)} textos...")
            df['clean'] = df['description'].apply(limpiar_texto)

            self.log("3. Entrenando Red Neuronal...")
            X_train, X_test, y_train, y_test = train_test_split(df['clean'], df['points'], test_size=0.2)
            
            pipeline = make_pipeline(
                TfidfVectorizer(max_features=3000),
                MLPRegressor(hidden_layer_sizes=(50,50), max_iter=30)
            )
            pipeline.fit(X_train, y_train)
            
            mae = mean_absolute_error(y_test, pipeline.predict(X_test))
            self.log(f"✅ COMPLETADO. Error MAE: {mae:.2f}")
            
            joblib.dump(pipeline, RUTA_MODELO)
            self.modelo = pipeline
            
            self.btn_predecir.configure(state="normal", text="✨ ANALIZAR CALIDAD")
            self.btn_entrenar.configure(state="normal", text="ENTRENAMIENTO COMPLETADO")
            
        except Exception as e:
            self.log(f"❌ ERROR: {e}")
            self.btn_entrenar.configure(state="normal")

    # HILO PREDICCIÓN
    def iniciar_prediccion(self):
        texto = self.input_text.get("0.0", "end").strip()
        if len(texto) < 5: return
        
        self.btn_predecir.configure(state="disabled", text="⏳ Analizando...")
        self.lbl_score.configure(text="...")
        self.txt_explicacion.configure(state="normal")
        self.txt_explicacion.delete("0.0", "end")
        self.txt_explicacion.configure(state="disabled")
        
        threading.Thread(target=self.proceso_prediccion, args=(texto,), daemon=True).start()

    def proceso_prediccion(self, texto):
        try:
            texto_limpio = limpiar_texto(texto)
            prediccion = self.modelo.predict([texto_limpio])[0]
            
            # LIME
            explainer = LimeTextExplainer(verbose=False)
            exp = explainer.explain_instance(
                texto_limpio, 
                lambda x: funcion_lime_wrapper(x, self.modelo), 
                num_features=5, labels=[0]
            )
            
            # UI Updates
            self.lbl_score.configure(text=f"{prediccion:.1f}")
            
            # Semáforo de colores
            if prediccion >= 90:
                color, msg = "#2CC985", "¡Excelente! 🏆"
            elif prediccion >= 85:
                color, msg = "#F1C40F", "Muy Bueno 👌"
            else:
                color, msg = "#E74C3C", "Regular/Malo ⚠️"
            
            self.lbl_score.configure(text_color=color)
            self.lbl_msg_score.configure(text=msg, text_color=color)

            # Explicación
            txt_out = ""
            for palabra, peso in exp.as_list(label=0):
                icono = "📈 Sube" if peso > 0 else "📉 Baja"
                txt_out += f"{icono} ({peso:.2f}) -> '{palabra}'\n"
                
            self.txt_explicacion.configure(state="normal")
            self.txt_explicacion.insert("0.0", txt_out)
            self.txt_explicacion.configure(state="disabled")

        except Exception as e:
            print(e)
        finally:
            self.btn_predecir.configure(state="normal", text="✨ ANALIZAR CALIDAD")

# =============================================================================
# EJECUCIÓN
# =============================================================================
if __name__ == "__main__":
    app = WineApp()
    app.mainloop()