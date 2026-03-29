import os
import io
import requests
import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.75
MODEL_PATH = "final_eye_disease_model.h5"
HF_URL = "https://huggingface.co/Nik-23/eye-disease-detection/resolve/main/final_eye_disease_model%20(1).h5"
CLASS_NAMES = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']


# ─────────────────────────────────────────────
# DOWNLOAD MODEL FROM HUGGINGFACE
# ─────────────────────────────────────────────
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from HuggingFace...")
        try:
            response = requests.get(HF_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"Failed to download model: {e}")
    else:
        print("Model already exists, skipping download.")


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
download_model()

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# ─────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_image(image_bytes):
    """
    Accepts JPG, PNG, WEBP, BMP, PPM, TIFF formats.
    Resizes to 224x224 and normalizes pixel values.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except UnidentifiedImageError:
        raise ValueError("Invalid image format. Supported: JPG, PNG, WEBP, PPM, BMP, TIFF.")


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def home():
    return "Eye Disease Detection Backend is running!"


@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded. Please try again later.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    try:
        processed_img = preprocess_image(file.read())

        predictions = model.predict(processed_img)

        predicted_index = np.argmax(predictions[0])
        raw_confidence = float(predictions[0][predicted_index])

        if raw_confidence < CONFIDENCE_THRESHOLD:
            predicted_class = "Unknown / Invalid Image"
            confidence = raw_confidence
            message = "Low confidence. This may not be a retinal scan."
        else:
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = raw_confidence
            message = "Analysis successful."

        scores_dict = {
            class_name: float(score)
            for class_name, score in zip(CLASS_NAMES, predictions[0])
        }

        return jsonify({
            'class': predicted_class,
            'confidence': confidence,
            'scores': scores_dict,
            'message': message
        })

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────
# RUN APP
# ─────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)