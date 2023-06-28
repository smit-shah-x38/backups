from flask import Flask
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)

@app.route("/test")
@cross_origin()
def helloWorld():
  return "Hello, cross-origin-world!"
