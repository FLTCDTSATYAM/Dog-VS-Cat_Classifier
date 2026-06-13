import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
import os

# Hugging Face model URL
MODEL_URL = "https://huggingface.co/satyamk59179/dog_cat_classifier.keras/resolve/main/dog_cat_classifier.keras"
MODEL_PATH = "dog_cat_classifier.keras"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... Please wait."):
        response = requests.get(MODEL_URL, stream=True)

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("🐶 Dog vs 🐱 Cat Classifier")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Preprocess image
    img = image.resize((256, 256))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img, verbose=0)
    confidence = float(prediction[0][0])

    st.write(f"Prediction Score: {confidence:.4f}")

    if confidence > 0.5:
        st.success(f"🐶 Dog ({confidence:.2%} confidence)")
    else:
        st.success(f"🐱 Cat ({(1 - confidence):.2%} confidence)")