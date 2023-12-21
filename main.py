from flask import Flask, request, jsonify
import os
from PIL import Image
import numpy as np
import tensorflow
from keras.models import load_model
from keras.preprocessing.image import img_to_array

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.json.sort_keys = False


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Load the pre-trained semaphore model
model_path = 'model_ml.h5'
model = load_model(model_path)

# Semaphore classes
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z']


# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((150, 150))  # Adjust the size to match the expected input size of the model
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


# Route for semaphore image classification
@app.route('/classify_semaphore', methods=['POST'])
def classify_semaphore():
    if 'image' not in request.files:
        response = \
            {
                'success': False,
                'message': 'No image uploaded'
            }
        return jsonify(response), 400, {'Content-Type': 'application/json; charset=utf-8'}

    image_file = request.files['image']

    if image_file and allowed_file(image_file.filename):
        # Save the uploaded image temporarily
        temp_image_path = 'temp_image.jpg'
        image_file.save(temp_image_path)

        # Preprocess the image
        preprocessed_image = preprocess_image(temp_image_path)

        # Perform prediction
        predictions = model.predict(preprocessed_image)

        # Get the class with the highest probability
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = classes[predicted_class_index]

        # Get the confidence of the prediction
        confidence = float(predictions[0][predicted_class_index])

        # Set a threshold for confidence (adjust as needed)
        confidence_threshold = 0.5  # Example threshold value

        # Prepare JSON response
        if confidence >= confidence_threshold:
            response = \
                {
                    'success': True,
                    'message': {
                        'predicted_class': predicted_class,
                        'confidence': confidence
                    }
                }
        else:
            response = \
                {
                    'success': True,
                    'message': {
                        'predicted_class': 'Not a Semaphore',
                        'confidence': confidence
                    }
                }

        # Remove the temporary image file
        os.remove(temp_image_path)

        return jsonify(response), {'Content-Type': 'application/json; charset=utf-8'}
    else:
        response = \
            {
                'success': False,
                'message': "Invalid file format. Please upload a JPG, JPEG, or PNG image."
            }
        return jsonify(response), 400, {'Content-Type': 'application/json; charset=utf-8'}


if __name__ == '__main__':
    app.run(debug=True)

# @app.route('/')
# def index():
#     return jsonify({
#         "status" : {
#             "code" : 200,
#             "message" : "Succes fetching the API",
#         },
#         "data" : None,
#     }),200
