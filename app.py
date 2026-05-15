import gradio as gr
import google.generativeai as genai
from transformers import pipeline
import os

# Configuración Gemini
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# Modelos HuggingFace
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
translator = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")

# --- Funciones ---

def resumir_documento(texto):
    if not texto.strip():
        return "⚠️ Por favor ingresa un texto para resumir."
    try:
        prompt = f"""Eres un consultor experto. Resume el siguiente documento técnico en un resumen ejecutivo profesional, 
        estructurado con puntos clave, hallazgos principales y recomendaciones. Máximo 300 palabras.
        
        Documento:
        {texto}"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error al conectar con Gemini: {str(e)}"

def analizar_sentimiento(texto):
    if not texto.strip():
        return "⚠️ Por favor ingresa un texto para analizar."
    try:
        resultado = sentiment_analyzer(texto[:512])[0]
        label = resultado["label"]
        score = round(resultado["score"] * 100, 2)
        emoji = "✅" if label == "POSITIVE" else "❌"
        label_es = "POSITIVO" if label == "POSITIVE" else "NEGATIVO"
        return f"{emoji} Sentimiento: **{label_es}**\n📊 Confianza: {score}%"
    except Exception as e:
        return f"❌ Error en análisis: {str(e)}"

def traducir_texto(texto):
    if not texto.strip():
        return "⚠️ Por favor ingresa un texto en inglés para traducir."
    try:
        resultado = translator(texto[:512])[0]
        return resultado["translation_text"]
    except Exception as e:
        return f"❌ Error en traducción: {str(e)}"

# --- Interfaz Gradio ---

with gr.Blocks(theme=gr.themes.Soft(), title="IntelliDoc AI Suite") as app:
    gr.Markdown("""
    # 🏭 IntelliDoc AI Suite
    ### Plataforma de Análisis Inteligente de Documentos Técnicos
    *Herramienta de IA para consultoría industrial y empresarial*
    """)

    with gr.Tabs():
        # Tab 1 - Resumen Ejecutivo
        with gr.Tab("📄 Resumen Ejecutivo"):
            gr.Markdown("Ingresa un informe, manual o documento técnico y obtén un resumen ejecutivo profesional.")
            with gr.Row():
                with gr.Column():
                    input_resumen = gr.Textbox(
                        label="Documento técnico",
                        placeholder="Pega aquí el contenido del documento...",
                        lines=12
                    )
                    btn_resumen = gr.Button("Generar Resumen", variant="primary")
                with gr.Column():
                    output_resumen = gr.Textbox(label="Resumen Ejecutivo", lines=12)
            btn_resumen.click(fn=resumir_documento, inputs=input_resumen, outputs=output_resumen)

        # Tab 2 - Análisis de Sentimiento
        with gr.Tab("📊 Análisis de Sentimiento"):
            gr.Markdown("Analiza el tono de reportes, feedback de clientes o evaluaciones de proveedores.")
            with gr.Row():
                with gr.Column():
                    input_sentimiento = gr.Textbox(
                        label="Texto a analizar",
                        placeholder="Ej: The maintenance service was excellent and very efficient...",
                        lines=8
                    )
                    btn_sentimiento = gr.Button("Analizar", variant="primary")
                with gr.Column():
                    output_sentimiento = gr.Markdown(label="Resultado")
            btn_sentimiento.click(fn=analizar_sentimiento, inputs=input_sentimiento, outputs=output_sentimiento)

        # Tab 3 - Traductor Técnico
        with gr.Tab("🌐 Traductor Técnico"):
            gr.Markdown("Traduce documentación técnica del inglés al español con precisión industrial.")
            with gr.Row():
                with gr.Column():
                    input_traduccion = gr.Textbox(
                        label="Texto en inglés",
                        placeholder="Enter the technical document in English...",
                        lines=8
                    )
                    btn_traduccion = gr.Button("Traducir", variant="primary")
                with gr.Column():
                    output_traduccion = gr.Textbox(label="Traducción al español", lines=8)
            btn_traduccion.click(fn=traducir_texto, inputs=input_traduccion, outputs=output_traduccion)

    gr.Markdown("---\n*Desarrollado con Gradio · Gemini API · HuggingFace Transformers*")

app.launch()