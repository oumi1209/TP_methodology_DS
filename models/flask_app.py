import cv2
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import threading

app = Flask(__name__)

# Load your trained model (replace 'model.h5' with your model file)
model = load_model('modelll_cnn.h5')

# Define a route for inference
@app.route('/predict', methods=['POST'])
def predict():
    try:

        img_file = request.files['image'].read()


        img = cv2.imdecode(np.frombuffer(img_file, np.uint8),-1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (224, 224))
        img = img.astype('float32') / 255

        predictions = model.predict(np.expand_dims(img, axis=0))

        result = {
            'predicted_classy': 'cc' if predictions[0][0] >= 0.5 else 'dd',
            'class_probability': float(predictions[0][0])
        }

        return jsonify(result)
    except Exception as e:
        return str(e), 400

if __name__ == '__main__':
    threading.Thread(target=lambda: app.run(debug=False)).start()