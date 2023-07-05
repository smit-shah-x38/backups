from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from langchain.cache import InMemoryCache
import langchain

langchain.llm_cache = InMemoryCache()

import requests
import urllib

from langchain.llms import OpenAI

llm = OpenAI(openai_api_key="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")

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


@app.route("/predict", methods=["POST"])
@cross_origin()
def respond():
    prompt = request.json["prompt"]
    response = llm(prompt)
    return {"Output": response}


@app.route("/test", methods=["POST"])
@cross_origin()
def answer():
    return {"Output": "Hello World"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
