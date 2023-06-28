from flask import Flask
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)


@app.route("/test", methods=["POST"])
@cross_origin()
def helloWorld():
    return {"Output": "Hello, cross-origin-world!"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
