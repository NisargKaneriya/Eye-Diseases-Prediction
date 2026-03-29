import os

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError 
import io

app = Flask(__name__)
CORS(app)

# 1. CONFIGURATION
MODEL_PATH = 'final_eye_disease_model (1).h5'
CONFIDENCE_THRESHOLD = 0.75 

print(f"Loading model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

CLASS_NAMES = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

def preprocess_image(image_bytes):
    """
    Handles multiple formats: JPG, PNG, WEBP, BMP, PPM, TIFF
    """
    try:
        # Image.open automatically detects the format (JPG, PNG, WEBP, PPM, etc.)
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB (Crucial for PNGs/WebPs with transparency or PPMs)
        img = img.convert('RGB')
        
        img = img.resize((224, 224)) 
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except UnidentifiedImageError:
        # This runs if the file is not a valid image (e.g., PDF, Text file)
        raise ValueError("Invalid image format. Supported formats: JPG, PNG, WEBP, PPM, BMP, TIFF.")

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        # 1. Process
        # This will now accept .webp, .ppm, .bmp, etc.
        processed_img = preprocess_image(file.read())
        
        # 2. Predict
        predictions = model.predict(processed_img)
        
        # 3. Analyze Results
        predicted_index = np.argmax(predictions[0])
        raw_confidence = float(predictions[0][predicted_index])
        
        # Filter Logic
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
        # Specific error for bad image formats
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# @app.route("/")
# def home():
#     return "Backend is running"

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8000, debug=True)
@app.route("/")
def home():
    return "Backend is running"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)