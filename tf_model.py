import numpy as np
import cv2
import tensorflow as tf
import streamlit as st


@st.cache_resource
def load_tf_model():
    model = tf.saved_model.load("virus_model_saved")
    return model


def predict_tf(image):

    model = load_tf_model()
    infer = model.signatures["serving_default"]

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (150,150))
    img = img / 255.0

    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0).astype("float32")

    pred = infer(tf.constant(img))
    pred = list(pred.values())[0].numpy()[0][0]

    if pred > 0.5:
        return "PNEUMONIA", float(pred)
    else:
        return "NORMAL", float(1 - pred)