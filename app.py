from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
from ultralytics import YOLO
import os

app = Flask(__name__)
model = YOLO('models/best.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', error='No image uploaded!')

    file = request.files['image']
    file_path = os.path.join('static', 'uploaded.jpg')
    file.save(file_path)

    img = cv2.imread(file_path)
    results = model.predict(img, imgsz=640, conf=0.02)
    processed_frame = results[0].plot(line_width=1)

    result_path = os.path.join('static', 'result.jpg')
    cv2.imwrite(result_path, processed_frame)

    return render_template('index.html', result_image='static/result.jpg')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
