from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

from PIL import Image
from io import BytesIO
import base64

from deepface import DeepFace
import os
import pandas as pd

embeddings = []
face_coords = []

df = pd.DataFrame({"embedding": [], "facial_area": []})


def decode_base64_image(base64_string, image_path):
    # Decode the base64 string into bytes
    image_bytes = base64.b64decode(base64_string)

    # Convert the bytes to an image object
    image = Image.open(BytesIO(image_bytes))

    # Save the image
    image.save(image_path)


@app.route("/ask", methods=["POST"])
@cross_origin()
def ask():
    global df
    global embeddings
    global face_coords
    string_img = request.json["string"]

    img_path = "save_image.jpg"
    decode_base64_image(string_img, img_path)

    embedding = DeepFace.represent(img_path, model_name="Facenet")
    embeddings.append(embedding[0]["embedding"])
    face_coords.append(embedding[0]["facial_area"])

    dftemp = pd.DataFrame(embedding)
    df = pd.concat([df, dftemp], ignore_index=True)

    return list(dftemp["embedding"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
