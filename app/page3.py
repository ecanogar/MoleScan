from utils import *
import streamlit as st
import tensorflow as tf
import cv2

#Cargamos el mejor modelo para clasificación de lesiones dermatológicas
@st.cache_resource  # Usar cache_resource para almacenar el modelo
def load_model():
    model=tf.keras.models.load_model("models/model.h5")
    return model

# Título de la aplicación
st.title("👩‍⚕️ Diagnósis del tipo de lesión dermatológica")

# Widget para subir la imagen
uploaded_file = st.file_uploader("Sube una imagen dermatoscópica de la lesión", type=["jpg", "jpeg", "png"])

# Mostrar la imagen una vez cargada
if uploaded_file is not None:

    # Botón para iniciar la predicción
    if st.button("Realiza el diagnóstico"):
        # Guardar la imagen subida en el disco para usarla con YOLO
        with open("tmp/uploaded_image_derm.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        # Cargar el modelo y realizar la predicción
        model = load_model()

        # Preprocesar la imagen
        source=tf.io.read_file("tmp/uploaded_image_derm.jpg")
        image = tf.image.decode_jpeg(source, channels=3)
        image = tf.cast(image, tf.float64)
        image /= 255.0
        img_size = (256,192,3)
        image = tf.image.resize(image, img_size[0:-1])

        # Realizar la predicción
        results = model.predict(np.expand_dims(image, axis=0)).flatten()
        
        classes = ['Nevo melanocítico', 'Melanoma', 'Lesiones benignas (queratosis)', 'Carcinoma basoceluclar', 'Queratoris actínicas', 'Lesiones vaculares', 'Dermatofibroma']

        # Mostrar el resultado
        # Cargar la imagen original
        image = cv2.imread("tmp/uploaded_image_derm.jpg")

        col1, col2 = st.columns(2)

        with col1:
            # Cargar la imagen original
            image = cv2.imread("tmp/uploaded_image_derm.jpg")
            # Convertir la imagen de BGR a RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Mostrar la imagen con los bounding boxes
            st.header("Imagen de la lesión")
            st.image(image_rgb, caption="Imagen dermatoscópica", use_column_width=True)

        with col2:
            st.header("Diagnóstico")

            # Imprimo las predicciones con las probabilidade más altas
            top_indices = np.argsort(results)[-3:][::-1]
            for i, idx in enumerate(top_indices):
                st.text(f"{classes[idx]} ({results[idx]*100:.2f}%)")

        "🆘 Por su seguridad, este diagnóstico siempre tiene que ser revisado por un dermatólogo 👩‍⚕️"