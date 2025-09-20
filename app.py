# Modified app.py for Vercel deployment

from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from PIL import Image
import io
import base64
import tempfile

app = Flask(__name__)

# Initialize model as None - will load when needed
model = None
classes = ['Coccidiosis', 'Healthy', 'Salmonella', 'New Castle Disease']

def load_model():
    """Load model only when needed to reduce cold start time"""
    global model
    if model is None:
        try:
            # Import tensorflow here to reduce initial load time
            import tensorflow as tf
            model = tf.keras.models.load_model("healthy_vs_rotten.h5")
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = False
    return model

def predict_image(image_data):
    """Make prediction from image data"""
    try:
        # Load model if not already loaded
        current_model = load_model()
        if not current_model:
            return "Model loading error"
        
        # Process image
        img = Image.open(io.BytesIO(image_data))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize((224, 224))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        
        # Make prediction
        pred = current_model.predict(arr)[0]
        return classes[np.argmax(pred)]
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return f"Prediction error: {str(e)}"

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/training')
def training():
    return render_template('training.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def upload():
    """Handle file upload and prediction"""
    try:
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('index.html', prediction="No file uploaded")

        # Read file data
        file_data = file.read()
        if len(file_data) == 0:
            return render_template('index.html', prediction="Empty file")

        # Make prediction
        prediction = predict_image(file_data)
        
        # Convert image to base64 for display
        img_base64 = base64.b64encode(file_data).decode('utf-8')
        img_src = f"data:image/jpeg;base64,{img_base64}"

        return render_template('index.html', 
                             prediction=prediction, 
                             img_path=img_src)
    
    except Exception as e:
        print(f"Upload error: {e}")
        return render_template('index.html', 
                             prediction=f"Error: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def predict_api():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        prediction = predict_image(image_data)
        
        return jsonify({"prediction": prediction})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route('/api/health')
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

if __name__ == '__main__':
    app.run(debug=True)
