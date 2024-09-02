from utils import *
import streamlit as st
from ultralytics import YOLO
import cv2

#Cargamos el mejor modelo para detección de lunares
@st.cache_resource  # Usar cache_resource para almacenar el modelo
def load_model():
    model = YOLO('models/best.pt')
    return model

# Título de la página
st.title("🔎 Calcula el número de lunares")

# Widget para subir la imagen
uploaded_file = st.file_uploader("Sube la imagen de una zona de la piel", type=["jpg", "jpeg", "png"])

# Mostrar la imagen una vez cargada
if uploaded_file is not None:

    # Botón para iniciar la predicción
    if st.button("Realiza el calculo"):
        # Guardar la imagen subida en el disco para usarla con YOLO
        with open("tmp/uploaded_image_det.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        # Cargar el modelo y realizar la predicción
        model = load_model()
        results = model.predict(source="tmp/uploaded_image_det.jpg")
            
        # Calcular el número de lunares detectados
        num_moles = sum(1 for cls in results[0].boxes.cls if cls == 0)

        # Cargar la imagen original
        image = cv2.imread("tmp/uploaded_image_det.jpg")
        for i, box in enumerate(results[0].boxes.xyxy):
            class_id = results[0].boxes.cls[i]
            if class_id == 0:  # Clase 0 es "Lunar"
                # Dibujar el bounding box
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 15)

        col1, col2 = st.columns(2)

        with col1:
            # Convertir la imagen de BGR a RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Mostrar la imagen con los bounding boxes
            st.image(image_rgb, caption="Lunares detectados", use_column_width=True)

        with col2:
            st.subheader(f"Número de lunares detectados: {num_moles}")