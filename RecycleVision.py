import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import pandas as pd
import altair as alt


# Page Config

st.set_page_config(
    page_title="RecycleVision",
    page_icon="♻️",
    layout="wide"
)

# Load Model

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "recyclevision_model.h5")
    return tf.keras.models.load_model(model_path)

model = load_model()

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'organic']

# Header

st.markdown(
    """
    <h1 style='text-align: center;'>♻️ RecycleVision</h1>
    <h4 style='text-align: center; color: gray;'>
    Garbage Image Classification using Deep Learning
    </h4>
    """,
    unsafe_allow_html=True
)

st.divider()

# Layout

col1, col2 = st.columns([1, 1])

# Upload Section

with col1:
    st.subheader("📤 Upload Garbage Image")
    uploaded_file = st.file_uploader(
        "Supported formats: JPG, JPEG, PNG",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

# Prediction Section

with col2:
    st.subheader("📊 Prediction Result")

    if uploaded_file and st.button("🔍 Predict Waste Type"):
        with st.spinner("Analyzing image..."):
            # Preprocess
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            predicted_class = class_names[predicted_index]
            confidence = prediction[0][predicted_index] * 100

        # Result card
        st.success(f"🗑️ **Predicted Category:** {predicted_class.upper()}")
        st.progress(int(confidence))
        st.write(f"**Confidence:** {confidence:.2f}%")

        # Probability chart
        prob_df = pd.DataFrame({
            "Category": class_names,
            "Probability (%)": prediction[0] * 100
        })

        st.subheader("📈 Class Probability Distribution (%)")

        chart = alt.Chart(prob_df).mark_bar().encode(
        x=alt.X("Category", sort=None),
        y=alt.Y("Probability (%)", scale=alt.Scale(domain=[0, 100])),
        tooltip=["Category", "Probability (%)"]
        )

        text = chart.mark_text(
        align='center',
        baseline='bottom',
        dy=-5
        ).encode(
        text=alt.Text("Probability (%):Q", format=".2f")
        )

        st.altair_chart(chart + text, use_container_width=True)


# Footer

st.divider()
st.markdown(
    "<p style='text-align: center; color: gray;'>Built with ❤️ using CNN & Streamlit</p>",
    unsafe_allow_html=True
)
