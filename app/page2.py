from utils import *
import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

#Cargamos el mejor modelo para segmentación de lunares
@st.cache_resource  # Usar cache_resource para almacenar el modelo
def load_model():
    model = YOLO('models/best_seg.pt')
    return model

# Título de la página
st.title("📏 Calcula el tamaño de un lunar")

# Widget para subir la imagen
uploaded_file = st.file_uploader("Sube la imagen de un lunar y el círculo de referencia", type=["jpg", "jpeg", "png"])
reference_diameter_cm = st.number_input("Introduce el diámetro del círculo de referencia en cm", value=1.8)

# Mostrar la imagen una vez cargada
if uploaded_file is not None and reference_diameter_cm is not None:

    # Botón para iniciar la predicción
    if st.button("Realiza la medición"):
        # Guardar la imagen subida en el disco para usarla con YOLO
        with open("tmp/uploaded_image_seg.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        # Cargar el modelo y realizar la predicción
        model = load_model()
        results = model.predict(source="tmp/uploaded_image_seg.jpg")

        # Área de círculo de referencia en cm2
        reference_radius_cm = reference_diameter_cm / 2
        reference_area_cm2 = np.pi * (reference_radius_cm ** 2)

        reference_polygon = None
        lunar_polygon = None

        for i, mask in enumerate(results[0].masks.xy):
            class_id = results[0].boxes.cls[i]
            if class_id == 1:  # Clase 1 es "Referencia"
                reference_polygon = mask
                reference_bbox = results[0].boxes.xywh[i]
            elif class_id == 0:  # Clase 0 es "Lunar"
                lunar_polygon = mask

        if reference_polygon is not None and lunar_polygon is not None:
            
            reference_area = calculate_area(reference_polygon) # Calcular el área del círculo de referencia en la imagen
            lunar_area = calculate_area(lunar_polygon) # Calcular el área del lunar en la imagen
            lunar_area_cm2 = (lunar_area / reference_area) * reference_area_cm2 # Calcular el área del lunar en cm²
            mole_longline, mole_longline_distance = calculate_longest_line(lunar_polygon) # Encontrar la línea longitudinal más larga del lunar
            ref_longline, ref_longline_distance = calculate_longest_line(reference_polygon) # Encontrar la línea longitudinal más larga del circulo de referencia
            longest_line_cm = (mole_longline_distance * reference_diameter_cm) / ref_longline_distance # Calcular linea longitudinal en cm²
            midpoint, intersection_points, mole_perpline_distance = calculate_perpendicular_intersections(mole_longline, lunar_polygon)
            perp_line_cm = (mole_perpline_distance * reference_diameter_cm) / ref_longline_distance # Calcular linea longitudinal en cm²
            
            col1, col2 = st.columns(2)

            with col1:
                st.header("Ejes del lunar")
                plot_mole_axes("tmp/uploaded_image_seg.jpg", mole_longline, midpoint, intersection_points, longest_line_cm, perp_line_cm)

            with col2:
                st.header("Dimensiones")
                st.text(f"Área del lunar: {lunar_area_cm2:.2f} cm²")
                st.text(f"Longitud eje mayor: {longest_line_cm:.2f} cm")
                st.text(f"Longitud eje menor: {perp_line_cm:.2f} cm")

        else:
            st.text("No se detectaron tanto el círculo de referencia como el lunar.")