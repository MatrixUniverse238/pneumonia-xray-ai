import tensorflow as tf

model = tf.keras.models.load_model("virus_model.h5", compile=False)

# Re-export model cleanly
model.save("virus_model_streamlit.h5")