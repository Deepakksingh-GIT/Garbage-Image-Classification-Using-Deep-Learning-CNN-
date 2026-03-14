import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Model path and image size settings
MODEL_PATH = 'models/garbage_classifier.h5'
IMAGE_SIZE = (224, 224)

@st.cache_resource
def load_model():
    """Load trained model once and cache it for fast repeated inference."""
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def get_class_labels():
    # Update this list if class order differs.
    return ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'organic']

def preprocess_image(image: Image.Image):
    """Resize + normalize image to model input format."""
    img = image.convert('RGB').resize(IMAGE_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# Streamlit UI
st.title('RecycleVision - Garbage Image Classification')
st.write('Upload a waste image and get predicted category + top-3 scores.')

uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded image', use_column_width=True)

    model = load_model()
    classes = get_class_labels()

    x = preprocess_image(img)
    preds = model.predict(x)[0]

    top3 = np.argsort(preds)[::-1][:3]

    st.subheader('Predictions')
    for i, idx in enumerate(top3, 1):
        st.write(f"{i}. {classes[idx]}: {preds[idx]*100:.2f}%")

    st.write('---')
    st.write('Raw probabilities:')
    st.json({classes[i]: float(preds[i]) for i in range(len(classes))})
