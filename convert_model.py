import tensorflow as tf

# load your model
model = tf.keras.models.load_model("virus_model.h5", compile=False)

# export to TensorFlow SavedModel
model.export("virus_model_saved")

print("Model exported successfully!")