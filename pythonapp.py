import numpy as np
import tensorflow as tf
import pandas as pd
import time
from flask import Flask, render_template, Response, request
import cv2

app = Flask(__name__)

# load the model
model = tf.keras.models.load_model('HackwithAI/model2')

# map the class to the label, the model is trained on six classes, we classified it to three classes
labels = {
    0: "biodegradable",
    1: "biodegradable",
    2: "recyclable",
    3: "recyclable",
    4: "biodegradable",
    5: "garbage",
}

# generate frames
def generate_frames():
    camera = cv2.VideoCapture(1)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/button_click', methods=['POST'])
def button_click():
    video = cv2.VideoCapture(1)
    success, frame = video.read()
    if not success:
        return '', 204
    else:
        # predict the class of the image
        img = frame
        img = cv2.resize(frame, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions, axis=1)
        print(labels[predicted_class[0]])
    return '', 204

if __name__ == "__main__":
    app.run(debug=True)
