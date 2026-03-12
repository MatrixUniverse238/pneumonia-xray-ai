import streamlit as st
import cv2
import numpy as np

from tf_model import predict_tf
from torch_model import load_torch_model
from gradcam import generate_gradcam, overlay_heatmap


model_torch = load_torch_model()
st.title("🫁 Pneumonia Detection with Explainable AI")

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg","png","jpeg"])

if uploaded_file:

    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, caption="Uploaded X-ray")

    label, confidence = predict_tf(image)

    if label == "PNEUMONIA":

        st.error(f"Prediction: {label} | Confidence: {confidence:.2f}")

        # Generate heatmap only for pneumonia
        heatmap = generate_gradcam(model_torch, image)

        result = overlay_heatmap(image, heatmap)

        st.subheader("Grad-CAM Heatmap")

        st.image(result, caption="Highlighted Infection Region")

    else:

        st.success(f"Prediction: {label} | Confidence: {confidence:.2f}")

        st.info("No pneumonia detected. Heatmap not generated.")