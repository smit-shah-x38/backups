from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from nbdt.utils import load_image_from_path
from IPython.display import display
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

from PIL import Image
import requests
import urllib

import requests
from PIL import Image
import numpy as np
import cv2

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

model = load_model("my_model.h5")


@app.route("/test", methods=["POST"])
@cross_origin()
def helloWorld():
    return {"Output": "Hello, cross-origin-world!"}


@app.route("/image", methods=["POST"])
@cross_origin()
def helloImage():
    # return {"Output": "Hello, image-world!"}
    url = request.json["url"]

    # headers = {
    #     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    # }

    # image = tf.keras.utils.load_img(url, target_size=(180, 180), color_mode="rgb")

    # req = urllib.request.Request(url, headers=headers)
    # response = urllib.request.urlopen(req)
    # # response = urllib.request.urlopen(url)  # get the image content
    # image = np.asarray(
    #     bytearray(response.read()), dtype="uint8"
    # )  # create an array of bytes  # decode the image

    # # Assuming 'image' is the array you want to resize
    # image_copy = np.copy(image)  # Make a copy of the array

    # resized_image = image_copy.resize((180, 180))  # Resize the copied image

    # image_copy = image_copy.resize((180, 180))  # resize the image to 224x224
    # # image_copy = image_copy / 255.0  # normalize the pixel values
    # # image_copy = np.expand_dims(image_copy, axis=0)  # add a batch dimension
    # hist = model.predict(image)  # get the model's prediction

    # print(url)
    # url = str(url)
    # if encode_string:
    #     return {"got image": encode_string}
    # else:
    #     return {"Didn't get Image": "Empty"}
    # im = load_image_from_path(encode_string)
    # display(im)

    # url = "https://example.com/image.jpg" # The url of the image
    # im = load_image_from_path(url)
    # display(im)

    url = request.json["url"]
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open("temp_img.jpg", "wb") as f:
        f.write(response.content)

    # Preprocess the image
    img = load_img(
        "temp_img.jpg", target_size=(180, 180)
    )  # adjust the target size to match your model's expected input size
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Make a prediction
    preds = model.predict(x)
    print(str(preds))
    lister = str(preds)[0]
    max_idx = np.argmax(lister)
    print(max_idx)

    return {"got image": str(preds)}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
