from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from nbdt.utils import load_image_from_path
from IPython.display import display
from keras.models import load_model

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
    url = request.json.get("url")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    req = urllib.request.Request(url, headers=headers)
    response = urllib.request.urlopen(req)
    # response = urllib.request.urlopen(url)  # get the image content
    image = np.asarray(
        bytearray(response.read()), dtype="uint8"
    )  # create an array of bytes  # decode the image

    image = image.resize((180, 180))  # resize the image to 224x224
    image = image / 255.0  # normalize the pixel values
    image = np.expand_dims(image, axis=0)  # add a batch dimension
    pred = model.predict(image)  # get the model's prediction

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
    return {"got image": str(pred)}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
