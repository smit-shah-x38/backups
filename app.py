from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

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
    encode_string = request.form["url"]
    # if encode_string:
    #     return {"got image": encode_string}
    # else:
    #     return {"Didn't get Image": "Empty"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
