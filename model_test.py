import cv2
import numpy as np
import tensorflow as tf
import pandas as pd

model = tf.keras.models.load_model('HackwithAI/model2')

# test the model with a single image
img_path = 'HackwithAI/bio2.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

predictions = model.predict(img)
predicted_class = np.argmax(predictions, axis=1)

# map the class to the label, the model is trained on six classes, we classified it to three classes
labels = {
    0: "biodegradable",
    1: "biodegradable",
    2: "recyclable",
    3: "recyclable",
    4: "biodegradable",
    5: "garbage",
}

# print the predicted class
print(img_path)
print(f"Predicted class: {labels[predicted_class[0]]}")
