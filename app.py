import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, request, render_template, send_file
import os

app = Flask(__name__)
session = ort.InferenceSession("models/best.onnx")

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

    # Preprocess the image for the ONNX model
    img_resized = cv2.resize(img, (640, 640))
    img_input = img_resized.astype(np.float32)
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0)

    # Run the ONNX model
    inputs = {session.get_inputs()[0].name: img_input}
    results = session.run(None, inputs)

    # Process results and draw bounding boxes on the image
    processed_frame = process_model_output(results, img_resized)

    result_path = os.path.join('static', 'result.jpg')
    if not cv2.imwrite(result_path, processed_frame):
        return render_template('index.html', error='Error saving result image!')

    return render_template('index.html', result_image='static/result.jpg')

def process_model_output(results, original_image):
    # Unpack model output
    detections = results[0][0]  # (1, 5, 8400) -> (5, 8400)

    # Iterate over detections
    for detection in detections.T:  # Iterate over each of the 8400 detections
        x, y, w, h, confidence = detection
        
        # Only draw detections with a confidence greater than a threshold
        if confidence > 0.5:  # Adjust the confidence threshold as needed
            # Convert normalized coordinates back to the image scale
            x1 = int((x - w / 2) * original_image.shape[1])  # Top-left x
            y1 = int((y - h / 2) * original_image.shape[0])  # Top-left y
            x2 = int((x + w / 2) * original_image.shape[1])  # Bottom-right x
            y2 = int((y + h / 2) * original_image.shape[0])  # Bottom-right y

            # Draw bounding box on the image
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Optionally, draw confidence score as text
            label = f'Conf: {confidence:.2f}'
            cv2.putText(original_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return original_image

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
