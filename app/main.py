import os
import time
import subprocess
import requests

from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from PIL import Image

import numpy as np

import model

import process_image

app = Flask(__name__)
CORS(app)

retina_model = model.build_model()
retina_model.load_weights("efficientNet-aptos2019-processed-grey-withBackground-multilabel-checkpoint2.keras")
print("model loaded")

zero_thresh = 0.98
one_thresh = 0.5
two_thresh = 0.33
three_thresh = 0.51
four_thresh = 0.2

INPUT_IMAGES_PATH = "/app/input_images"
OUTPUT_PROCESSED_PATH = "/app/output_processed_images"
DIR_OUT = '/app/output_final_images'
img_index = 0

@app.route("/predict", methods=['POST'])
def translate_text_ncslgr_use_dict_add_word():
    print("Predicting DR severity with APTOS 2019 EfficientNet model")

    print(request.files)

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    img_file = request.files["image"]
    print(img_file)
    
    img = Image.open(img_file)

    x = np.empty((1, 512, 512, 3), dtype=np.uint8)
    x[0, :, :, :] = img

    print("read image")

    # filename = f"{img_index}.png"
    # file_path = os.path.join(INPUT_IMAGES_PATH, filename)
    # img.save(file_path)

    # # Process, no grey
    # process_input_image(file_path, filename, grey=False)

    # # Process, grey
    # process_input_image(file_path, "grey-"+filename, grey=True)

    # file_path_processed = os.path.join(OUTPUT_PROCESSED_PATH,filename)
    # file_path_background = os.path.join(OUTPUT_PROCESSED_PATH,"grey-"+filename)

    # # Overlay segmentations
    # segment_image(file_path_processed, file_path_background)
    # final_file_name = f"predict-{filename}.png"
    # final_path =  f"{DIR_OUT}/{final_file_name}"

    # # Final image read
    # final_img = Image.open(final_path)
    # x = np.empty((1, 512, 512, 3), dtype=np.uint8)
    # x[0, :, :, :] = final_img

    prediction = retina_model.predict(x, use_multiprocessing=False)[0]
    print("prediction done")
    print(prediction)
    if prediction[1] < one_thresh:
        pred_category = 0
    elif prediction[2] < two_thresh:
        pred_category = 1
    elif prediction[3] < three_thresh:
        if prediction[4] > prediction[3] and prediction[4] > 0.16:
            pred_category = 4
        else:
            pred_category = 2
    elif prediction[4] < four_thresh:
        pred_category = 3
    else:
        pred_category = 4

    return jsonify({"prediction_category": pred_category})

def process_input_image(file_path, filename, grey):
    process_image.preprocess_image(file_path, filename, grey=grey)

# def segment_image(file_path, file_path_background):
    # result = subprocess.run(["./app/run_predict.sh", file_path, file_path_background], stdout=subprocess.PIPE)
    # requests.post("http://localhost:8081/segment", json={"file_path": file_path, "file_path_background": file_path_background})

# Health check route
@app.route("/isalive")
def is_alive():
    print("/isalive request")
    status_code = Response(status=200)
    return status_code

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)