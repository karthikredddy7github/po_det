from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model("healthy_vs_rotten.h5")
classes = ['Coccidiosis', 'Healthy', 'Salmonella', 'New Castle Disease']
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Prediction function
def predict(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        arr = image.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        pred = model.predict(arr)[0]
        return classes[np.argmax(pred)]
    except Exception as e:
        print("Error in prediction:", e)
        return "Invalid image"

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
    file = request.files.get('file')
    if not file:
        return render_template('index.html', prediction="No file uploaded")

    filename = secure_filename(file.filename)
    if filename == "":
        return render_template('index.html', prediction="Invalid file")

    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    pred = predict(path)
    img_path = '/' + path.replace('\\', '/')  # To ensure browser loads image

    return render_template('index.html', prediction=pred, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
