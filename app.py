import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Project paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "Weights" / "best_covid_mobilenetv2.keras"
LABELS_PATH = BASE_DIR / "Weights" / "class_labels.json"

st.set_page_config(
    page_title="COVID-19 X-ray Classifier",
    page_icon="🩻",
    layout="centered"
)

st.title("COVID-19 Detection Using Deep Learning")
st.write("Upload a chest X-ray image to get a model prediction.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_labels():
    with open(LABELS_PATH, "r") as file:
        labels = json.load(file)
    return {int(index): name for name, index in labels.items()}

try:
    model = load_model()
    labels = load_labels()
except Exception as error:
    st.error(f"Could not load the model or labels: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing image..."):
            image = image.resize((224, 224))
            image_array = np.array(image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = preprocess_input(image_array)

            predictions = model.predict(image_array, verbose=0)[0]

            predicted_index = int(np.argmax(predictions))
            predicted_class = labels[predicted_index]
            confidence = float(predictions[predicted_index]) * 100

        st.subheader("Prediction")
        st.success(f"Predicted class: {predicted_class}")
        st.metric("Model confidence", f"{confidence:.2f}%")

        st.subheader("Class probabilities")
        for index, score in enumerate(predictions):
            st.write(f"{labels[index]}: {score * 100:.2f}%")
            st.progress(float(score))

st.divider()
st.warning(
    "Educational project only. This model is not clinically validated "
    "and must not be used for medical diagnosis or treatment decisions."
)