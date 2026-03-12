import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# force legacy keras compatibility
tf.keras.utils.get_custom_objects()

model = load_model("virus_model.h5", compile=False)

def predict_tf(image):

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (150,150))
    img = img / 255.0

    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0][0]

    if pred > 0.5:
        return "PNEUMONIA", float(pred)
    else:
        return "NORMAL", float(1 - pred)