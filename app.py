# ============================================================
# App de Streamlit: Clasificador de frutas
# Modelo: MobileNetV2 (transferencia de aprendizaje) entrenado
# con el dataset Fruits-360 de Kaggle.
# La persona sube la foto de una fruta y el modelo dice cuál es.
# ============================================================

import os
import json
import base64
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# ------------------------------------------------------------
# Configuración general de la página
# ------------------------------------------------------------
st.set_page_config(page_title="Clasificador de frutas", page_icon="🍎", layout="centered")

# ------------------------------------------------------------
# Fondo de frutas: un patrón dibujado en SVG que se repite.
# No necesita subir ninguna imagen a GitHub.
# ------------------------------------------------------------
PATRON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="320" height="320" viewBox="0 0 320 320">
<rect width="320" height="320" fill="#fff4e3"/>
<g opacity="0.55">
<g transform="translate(50,50) rotate(-12)">
<path d="M0,-6 C-22,-24 -40,-2 -30,22 C-24,38 -10,40 0,34 C10,40 24,38 30,22 C40,-2 22,-24 0,-6Z" fill="#e4572e"/>
<path d="M0,-6 C0,-16 4,-24 10,-28" stroke="#6b4423" stroke-width="3" fill="none" stroke-linecap="round"/>
<path d="M4,-18 C14,-32 28,-30 30,-26 C24,-16 12,-14 4,-18Z" fill="#6aa84f"/></g>
<g transform="translate(160,40) rotate(10)">
<circle r="27" fill="#f9a03f"/><circle cx="-8" cy="-9" r="6" fill="#fcc57a"/>
<path d="M0,-27 C6,-38 18,-38 22,-32 C16,-24 6,-22 0,-27Z" fill="#6aa84f"/></g>
<g transform="translate(270,60) rotate(20)">
<path d="M-36,-8 C-22,26 22,30 42,-8 C26,8 -12,12 -36,-8Z" fill="#f6d55c" stroke="#c9a227" stroke-width="3" stroke-linejoin="round"/></g>
<g transform="translate(105,140) rotate(8)">
<path d="M0,-24 C20,-26 30,-8 20,12 C14,26 6,34 0,38 C-6,34 -14,26 -20,12 C-30,-8 -20,-26 0,-24Z" fill="#d62839"/>
<ellipse cx="-8" cy="-6" rx="2" ry="3" fill="#ffe08a"/><ellipse cx="8" cy="-4" rx="2" ry="3" fill="#ffe08a"/>
<ellipse cx="0" cy="8" rx="2" ry="3" fill="#ffe08a"/><ellipse cx="-10" cy="14" rx="2" ry="3" fill="#ffe08a"/><ellipse cx="10" cy="14" rx="2" ry="3" fill="#ffe08a"/>
<path d="M-14,-24 L0,-14 L14,-24 L6,-30 L0,-26 L-6,-30Z" fill="#4f9d3a"/></g>
<g transform="translate(225,150)">
<circle cx="-14" cy="-14" r="11" fill="#7b4fa3"/><circle cx="8" cy="-14" r="11" fill="#7b4fa3"/>
<circle cx="-22" cy="6" r="11" fill="#8a5db3"/><circle cx="0" cy="6" r="11" fill="#7b4fa3"/><circle cx="22" cy="6" r="11" fill="#8a5db3"/>
<circle cx="-10" cy="26" r="11" fill="#8a5db3"/><circle cx="12" cy="26" r="11" fill="#7b4fa3"/>
<path d="M-4,-26 C-4,-36 0,-42 6,-44" stroke="#6b4423" stroke-width="3" fill="none" stroke-linecap="round"/></g>
<g transform="translate(45,200) rotate(-8)">
<path d="M-34,0 A34,34 0 0 0 34,0 Z" fill="#ef476f"/><path d="M-34,0 A34,34 0 0 0 34,0" fill="none" stroke="#5aa845" stroke-width="6"/>
<ellipse cx="-14" cy="12" rx="2.5" ry="4" fill="#2b2b2b"/><ellipse cx="2" cy="16" rx="2.5" ry="4" fill="#2b2b2b"/><ellipse cx="16" cy="10" rx="2.5" ry="4" fill="#2b2b2b"/></g>
<g transform="translate(190,228) rotate(-18)">
<path d="M-32,0 C-24,-22 24,-22 32,0 C24,22 -24,22 -32,0Z" fill="#f4e04d" stroke="#d9c21f" stroke-width="2"/>
<path d="M32,0 L40,-3" stroke="#d9c21f" stroke-width="4" stroke-linecap="round"/></g>
<g transform="translate(272,250) rotate(6)">
<circle cx="-14" cy="14" r="12" fill="#b3122b"/><circle cx="14" cy="18" r="12" fill="#c81d3a"/>
<path d="M-14,2 C-12,-14 0,-26 6,-32" stroke="#5a7d2a" stroke-width="3" fill="none" stroke-linecap="round"/>
<path d="M14,6 C14,-10 8,-24 6,-32" stroke="#5a7d2a" stroke-width="3" fill="none" stroke-linecap="round"/>
<path d="M6,-32 C16,-42 30,-38 32,-34 C26,-26 14,-26 6,-32Z" fill="#6aa84f"/></g>
<g transform="translate(75,282) rotate(14)">
<path d="M0,-30 C14,-30 20,-14 18,0 C30,14 26,32 0,32 C-26,32 -30,14 -18,0 C-20,-14 -14,-30 0,-30Z" fill="#b7d34a"/>
<path d="M0,-30 C0,-38 4,-42 8,-44" stroke="#6b4423" stroke-width="3" fill="none" stroke-linecap="round"/></g>
</g>
</svg>"""

fondo = base64.b64encode(PATRON_SVG.encode("utf-8")).decode("utf-8")
TEXTO = "#3a2a1f"       # texto oscuro para que se lea bien sobre el fondo claro
ACENTO = "#d9531e"      # naranja de acento

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #fff4e3;
        background-image: url("data:image/svg+xml;base64,{fondo}");
        background-size: 320px 320px;
    }}
    /* La zona central es una tarjeta blanca para que el contenido se lea bien */
    .block-container {{
        background-color: rgba(255, 255, 255, 0.93);
        border-radius: 20px;
        padding: 2rem 2rem 3rem 2rem;
        margin-top: 1.5rem;
        box-shadow: 0 6px 24px rgba(120, 70, 20, 0.18);
    }}
    .stApp, .stApp p, .stApp label, .stApp li, .stApp span,
    .stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricLabel"],
    .stApp [data-testid="stCaptionContainer"] {{ color: {TEXTO}; }}
    h1, h2, h3 {{ color: {ACENTO} !important; }}
    [data-testid="stMetric"] {{
        background-color: #fff4e3;
        border-radius: 12px;
        padding: 10px 14px;
    }}
    [data-testid="stFileUploader"] section {{ background-color: #fff4e3; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🍎 Clasificador de frutas - Vianey Diaz - ISC - Campus Comayagua")


# ------------------------------------------------------------
# Nombres en español de las frutas más comunes del dataset.
# Si una clase no está aquí, se muestra su nombre original en inglés.
# ------------------------------------------------------------
TRADUCCION = {
    "Apple": "Manzana", "Apricot": "Albaricoque", "Avocado": "Aguacate",
    "Banana": "Banano", "Blueberry": "Arándano", "Cherry": "Cereza",
    "Coconut": "Coco", "Fig": "Higo", "Grape": "Uva", "Grapefruit": "Toronja",
    "Guava": "Guayaba", "Kiwi": "Kiwi", "Lemon": "Limón", "Lime": "Lima",
    "Lychee": "Lichi", "Mandarine": "Mandarina", "Mango": "Mango",
    "Melon": "Melón", "Orange": "Naranja", "Papaya": "Papaya",
    "Passion Fruit": "Maracuyá", "Peach": "Durazno", "Pear": "Pera",
    "Pineapple": "Piña", "Plum": "Ciruela", "Pomegranate": "Granada",
    "Raspberry": "Frambuesa", "Strawberry": "Fresa", "Tomato": "Tomate",
    "Watermelon": "Sandía", "Cucumber": "Pepino", "Carrot": "Zanahoria",
    "Potato": "Papa", "Onion": "Cebolla", "Corn": "Maíz", "Pepper": "Pimiento",
}


def nombre_visible(clase):
    """Devuelve 'Manzana (Apple Golden 1)' si se conoce la traducción."""
    for ingles in sorted(TRADUCCION, key=len, reverse=True):   # primero los nombres largos
        if clase.lower().startswith(ingles.lower()):
            return f"{TRADUCCION[ingles]} ({clase})"
    return clase


# ------------------------------------------------------------
# Función auxiliar: busca un archivo en la raíz o en las carpetas
# models/ y outputs/ (así funciona sin importar dónde lo subas)
# ------------------------------------------------------------
def buscar_archivo(nombre):
    rutas = [nombre, f"models/{nombre}", f"outputs/{nombre}"]
    for ruta in rutas:
        if os.path.exists(ruta):
            return ruta
    raise FileNotFoundError(f"No se encontró el archivo: {nombre}")


# ------------------------------------------------------------
# Preparar la imagen subida para que se parezca a las del entrenamiento
# (Fruits-360: fruta sola, centrada, imagen cuadrada y fondo blanco)
# ------------------------------------------------------------
def preparar_imagen(archivo):
    imagen = Image.open(archivo)
    imagen = ImageOps.exif_transpose(imagen)       # respeta la orientación de fotos de celular

    # PNG con transparencia: se pone sobre fondo blanco
    # (si no, lo transparente se vuelve negro y confunde al modelo)
    if imagen.mode in ("RGBA", "LA", "P"):
        imagen = imagen.convert("RGBA")
        blanco = Image.new("RGBA", imagen.size, (255, 255, 255, 255))
        imagen = Image.alpha_composite(blanco, imagen)
    imagen = imagen.convert("RGB")

    # Imagen no cuadrada: se completa con blanco hasta hacerla cuadrada,
    # así la fruta no se deforma al reducirla a 128x128
    lado = max(imagen.size)
    cuadrada = Image.new("RGB", (lado, lado), (255, 255, 255))
    cuadrada.paste(imagen, ((lado - imagen.width) // 2, (lado - imagen.height) // 2))
    return cuadrada


# ------------------------------------------------------------
# Carga del modelo y la metadata (una sola vez gracias al caché)
# ------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    # compile=False porque solo se usa para predecir
    modelo = tf.keras.models.load_model(buscar_archivo("frutas_mobilenetv2.keras"), compile=False)
    with open(buscar_archivo("model_metadata.json"), encoding="utf-8") as f:
        metadata = json.load(f)
    return modelo, metadata


modelo, metadata = cargar_modelo()

clases = metadata["clases"]                 # orden de salida del modelo
tam_img = tuple(metadata["tamano_imagen"])  # (128, 128)

st.caption(
    f"{metadata['nombre_modelo']} · Dataset: {metadata['dataset']} · "
    f"Accuracy en prueba: {metadata['accuracy_prueba'] * 100:.1f}%"
)

# Aviso si la versión de TensorFlow no coincide con la de Colab
if tf.__version__ != metadata.get("version_tensorflow"):
    st.warning(
        f"El modelo se entrenó con TensorFlow {metadata.get('version_tensorflow')} "
        f"pero la app usa {tf.__version__}. Ajuste requirements.txt."
    )

st.write(
    f"Sube la foto de una fruta y el modelo te dirá cuál es. Reconoce {len(clases)} "
    "categorías del dataset Fruits-360. Funciona mejor con una sola fruta, "
    "bien iluminada y sobre fondo claro."
)

# ------------------------------------------------------------
# Subir la imagen
# ------------------------------------------------------------
archivo = st.file_uploader("Sube una imagen de una fruta", type=["jpg", "jpeg", "png", "jfif"])

# ------------------------------------------------------------
# Predicción (solo cuando hay una imagen)
# ------------------------------------------------------------
if archivo is not None:
    # Abrir y preparar la imagen (fondo blanco y forma cuadrada)
    imagen = preparar_imagen(archivo)
    st.image(imagen, caption="Imagen como la ve el modelo", width="stretch")

    # Mismo preprocesamiento del cuaderno: redimensionar a 128x128.
    # El escalado de píxeles ya va dentro del modelo.
    entrada = tf.image.resize(np.array(imagen), tam_img).numpy()[None, ...]

    probabilidades = modelo.predict(entrada, verbose=0)[0]
    mejores = np.argsort(probabilidades)[::-1][:5]       # las 5 más probables
    indice = int(mejores[0])
    confianza = float(probabilidades[indice])

    st.subheader("Resultado")
    st.success(f"Fruta detectada: {nombre_visible(clases[indice])}")
    st.metric("Confianza", f"{confianza * 100:.1f}%")
    st.progress(min(max(confianza, 0.0), 1.0))

    # Si el modelo duda, se avisa
    if confianza < 0.60:
        st.warning(
            "La confianza es baja. Puede que la fruta no esté entre las categorías "
            "del modelo o que la foto no se vea bien (mucho fondo, poca luz)."
        )

    # Las 5 predicciones más probables
    st.write("Las 5 predicciones más probables")
    tabla = pd.Series(
        {nombre_visible(clases[i]): float(probabilidades[i]) * 100 for i in mejores}
    )
    st.bar_chart(tabla)

# ------------------------------------------------------------
# Información del modelo
# ------------------------------------------------------------
st.divider()
st.write("Información del modelo")
c1, c2, c3 = st.columns(3)
c1.metric("Categorías", len(clases))
c2.metric("Accuracy top-5", f"{metadata['accuracy_top5_prueba'] * 100:.1f}%")
c3.metric("Imágenes de prueba", metadata["imagenes_prueba"])
