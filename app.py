from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from nbdt.utils import load_image_from_path
from IPython.display import display

from PIL import Image
import requests

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


@app.route("/test", methods=["POST"])
@cross_origin()
def helloWorld():
    return {"Output": "Hello, cross-origin-world!"}


@app.route("/img", methods=["POST"])
@cross_origin()
def helloImage():
    # return {"Output": "Hello, image-world!"}
    url = request.form["url_key"]
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
    if url:
        return {"got image": url}
    else:
        return {"Didn't get Image": "Empty"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
