import tensorflow as tf

model = tf.keras.models.load_model("virus_model.h5", compile=False)

# correct filename
model.save_weights("virus.weights.h5")

print("Weights saved successfully")