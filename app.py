from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from PIL import Image
import joblib

app = Flask(__name__)

MODEL_PATH = os.path.join('Model', 'final_model.keras')
model = load_model(MODEL_PATH)

# Load Random Forest model and scaler
RF_MODEL_PATH = os.path.join('Model', 'final_best_rf_model_vegetables.joblib')
RF_SCALER_PATH = os.path.join('Model', 'final_rf_scaler_vegetables.joblib')
rf_model = joblib.load(RF_MODEL_PATH)
rf_scaler = joblib.load(RF_SCALER_PATH)

class_labels = ['Asparagus', 'Banana', 'Broccoli', 'Carrot', 'Corn', 'Eggplant', 'Orange', 'Pineapple', 'Potato', 'Tomato']

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    pred_class = np.argmax(preds, axis=1)[0]
    return class_labels[pred_class]

def rf_predict(img_path, model, scaler):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((128, 128))
    x = np.array(img)
    # Extract features (must match training pipeline)
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    import cv2
    def extract_color_histogram(image, bins=(8, 8, 8)):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()
    def extract_haralick_features(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        glcm = graycomatrix(gray_image, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast').ravel()
        dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()
        homogeneity = graycoprops(glcm, 'homogeneity').ravel()
        energy = graycoprops(glcm, 'energy').ravel()
        correlation = graycoprops(glcm, 'correlation').ravel()
        asm = graycoprops(glcm, 'ASM').ravel()
        haralick_features = np.hstack([
            np.mean(contrast), np.std(contrast),
            np.mean(dissimilarity), np.std(dissimilarity),
            np.mean(homogeneity), np.std(homogeneity),
            np.mean(energy), np.std(energy),
            np.mean(correlation), np.std(correlation),
            np.mean(asm), np.std(asm)
        ])
        return haralick_features
    def extract_lbp_features(image, P=24, R=3, method='uniform'):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray_image, P, R, method=method)
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return hist
    def extract_hu_moments(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        moments = cv2.moments(gray_image)
        hu_moments = cv2.HuMoments(moments).flatten()
        for i in range(0, 7):
            hu_moments[i] = -1 * np.sign(hu_moments[i]) * np.log10(abs(hu_moments[i]) + 1e-7)
        return hu_moments
    color_hist_features = extract_color_histogram(x)
    haralick_features = extract_haralick_features(x)
    lbp_features = extract_lbp_features(x)
    hu_features = extract_hu_moments(x)
    combined_features = np.hstack([
        color_hist_features,
        haralick_features,
        lbp_features,
        hu_features,
    ])
    combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
    features_scaled = scaler.transform([combined_features])
    pred = model.predict(features_scaled)
    return class_labels[int(pred[0])]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    img_path = None
    selected_model = 'cnn'
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction=None, img_path=None, selected_model=selected_model)
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction=None, img_path=None, selected_model=selected_model)
        if file:
            filepath = os.path.join('static', file.filename)
            file.save(filepath)
            selected_model = request.form.get('model', 'cnn')
            if selected_model == 'cnn':
                prediction = model_predict(filepath, model)
            else:
                prediction = rf_predict(filepath, rf_model, rf_scaler)
            img_path = filepath
    return render_template('index.html', prediction=prediction, img_path=img_path, selected_model=selected_model)

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
