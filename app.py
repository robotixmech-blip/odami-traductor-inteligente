import gradio as gr
from transformers import pipeline

# =========================
#  CSS personalizado
# =========================
custom_css = """
<style>
/* Fondo blanco general */
body, .gradio-container {
    background: #ffffff !important;
    margin: 0;
    padding: 0;
}

/* Contenedor principal centrado y más estrecho */
.gradio-container > div {
    max-width: 650px;
    margin: 0 auto !important;
}

/* Fondo blanco en todos los contenedores internos */
.gradio-container div {
    background-color: #ffffff !important;
}

/* Tarjeta principal SIN borde marcado (quitamos el recuadro grande) */
.gradio-container .gr-block,
.gradio-container .gr-form {
    border-radius: 16px !important;
    border: none !important;
}

/* Labels (Español, Ódami) bien negros, centrados y más fuertes */
.gradio-container label,
.gradio-container .gr-label,
.gradio-container .gr-text-label {
    color: #000000 !important;
    text-align: center;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    opacity: 1 !important;
}

/* Texto general en negro */
.gradio-container {
    color: #000000 !important;
}

/* Cuadros de texto blancos con borde gris claro */
textarea,
.gr-textbox textarea {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #cccccc !important;
}

/* Botones blancos, CON borde negro visible */
button,
.gr-button {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #000000 !important;
    box-shadow: none !important;
    border-radius: 6px !important;
}

/* Ocultar footer de Gradio si apareciera */
footer {
    display: none !important;
}
</style>
"""

# =========================
#  Modelo de traducción
# =========================
MODEL_NAME = "robotix123/odami_translator_v0.1"

translator = pipeline(
    "translation",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
)

def translate_es_to_odami(texto: str) -> str:
    """
    Traduce de español a Ódami usando tu modelo.
    Convierte a minúsculas para evitar que salga inglés.
    """
    texto = texto.strip().lower()
    if not texto:
        return ""
    try:
        out = translator(texto)
        return out[0]["translation_text"]
    except Exception as e:
        return f"Error al traducir: {e}"

def limpiar_campos():
    """Vaciar ambos cuadros de texto."""
    return "", ""

# =========================
#  Interfaz Gradio
# =========================
with gr.Blocks() as demo:
    # Inyectar CSS
    gr.HTML(custom_css)

    # (Quitamos el título interno, ya lo tienes en la web)

    # Cuadros de texto
    with gr.Column():
        entrada = gr.Textbox(
            label="Español",
            placeholder="Escribe una frase en español.",
            lines=4,
        )
        salida = gr.Textbox(
            label="Ódami",
            lines=4,
        )

    # Botones
    with gr.Row():
        btn_traducir = gr.Button("Traducir")
        btn_borrar = gr.Button("Borrar")

    # Conexiones
    btn_traducir.click(fn=translate_es_to_odami, inputs=entrada, outputs=salida)
    btn_borrar.click(fn=limpiar_campos, inputs=None, outputs=[entrada, salida])


if __name__ == "__main__":
    demo.launch()
