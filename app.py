from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from PIL import Image

app = Flask(__name__)

MODEL_PATH = os.path.join('Model', 'final_model.keras')
model = load_model(MODEL_PATH)

class_labels = ['Asparagus', 'Banana', 'Broccoli', 'Carrot', 'Corn', 'Eggplant', 'Orange', 'Pineapple', 'Potato', 'Tomato']

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    pred_class = np.argmax(preds, axis=1)[0]
    return class_labels[pred_class]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    img_path = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction=None, img_path=None)
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction=None, img_path=None)
        if file:
            filepath = os.path.join('static', file.filename)
            file.save(filepath)
            prediction = model_predict(filepath, model)
            img_path = filepath
    return render_template('index.html', prediction=prediction, img_path=img_path)

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
