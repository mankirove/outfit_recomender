# outfit_recomender
A small online application that used EfficientNet pretrained model to give outfit style recomendation based on uploaded photo.


## Features
- Upload an image and get recommendations for similar images.
- Uses embeddings from a pre-trained EfficientNet model for image similarity.
- Flask-based web application with image upload functionality.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/USERNAME/outfit-recommender.git
    ```
2. Navigate to the project directory:
    ```bash
    cd outfit-recommender
    ```
3. Install the required packages:
    ```bash
    pip install tensorflow,Flask,pandas,numpy
    ```

## Usage

1. Run the Flask application:
    ```bash
    python app.py
    ```
2. Open your browser and go to `http://127.0.0.1:5000`.
3. Upload an image to get recommendations for similar images.

