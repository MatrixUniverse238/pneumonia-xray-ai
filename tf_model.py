import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model("virus_model.h5", compile=False)

def predict_tf(image):

    # convert to grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # correct resize
    img = cv2.resize(img, (150,150))

    # normalize
    img = img / 255.0

    # add channel dimension
    img = np.expand_dims(img, axis=-1)

    # add batch dimension
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0][0]

    if pred > 0.5:
        return "PNEUMONIA", float(pred)
    else:
        return "NORMAL", float(1 - pred)