import os
import pickle
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'image_embeddings.pkl'
model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_embedding(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    embedding = model.predict(img_array)
    return embedding.flatten()


def load_embeddings():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        return []


def find_similar_images(query_embedding, image_embeddings, top_n=5):
    distances = []
    for img_name, img_embedding in image_embeddings:
        dist = np.linalg.norm(query_embedding - img_embedding)
        distances.append((img_name, dist))
    distances.sort(key=lambda x: x[1])
    return [img_name for img_name, _ in distances[:top_n]]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        upload_path = upload_path.replace('\\', '/')
        file.save(upload_path)
      
        app.logger.debug(f'File saved at {upload_path}')

        
        query_embedding = get_embedding(upload_path)
        image_embeddings = load_embeddings()
        similar_images = find_similar_images(query_embedding, image_embeddings)

        return render_template('result.html', images=similar_images)
    else:
        return 'File type not allowed'


if __name__ == '__main__':
    app.run(debug=True)
