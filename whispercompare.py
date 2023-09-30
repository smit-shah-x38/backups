from flask import Flask, request, jsonify
from flask import Response
from flask_cors import CORS, cross_origin
import whisper
import os
import base64
import pandas as pd
import tfidf_matcher as tm

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

# Load a Whisper model (you can choose from small, medium, large or xlarge)

model = whisper.load_model("small")
database = pd.DataFrame(
    [
        {"id": 1, "command": "Turn on the light"},
        {"id": 2, "command": "Play some music"},
        {"id": 3, "command": "Open the door"},
    ]
)


def matcherizer(inputcommand):
    result = match4 = tm.matcher(inputcommand, database["command"], k_matches=1)
    matches4_sorted = result.sort_values(by=["Lookup 1 Confidence"], ascending=False)
    rslt = matches4_sorted.loc[matches4_sorted["Lookup 1 Confidence"] >= 0.75]

    if not len(rslt) == 0:
        return rslt
    # If the result is empty, print a message
    else:
        db = pd.DataFrame([{"id": 0, "command": "Not Found"}])
        return db


# Define a route for uploading audio files


@app.route("/upload", methods=["POST"])
@cross_origin()
def upload():
    encode_string = request.form["transcription"]

    if encode_string != "hello":
        wav_file = open("temporary.wav", "wb")
        decode_string = base64.b64decode(encode_string)
        wav_file.write(decode_string)
        transcript = model.transcribe("temporary.wav")

        result = matcherizer(transcript)
        print(result)
        # Return the transcript as a JSON response
        return jsonify({"Works, somehow": transcript, "Results": result.to_json()})
    else:
        return Response(
            jsonify({"Request works, but doesn't": "hello"}),
            headers={"Access-Control-Allow-Origin": "*"},
        )

        # Return an error message if no audio file is provided


#     return jsonify(error="No audio file provided"), 400

# Define a route for testing


@app.route("/test", methods=["GET", "POST"])
@cross_origin()
def test():
    return jsonify({"Hello": "Hello"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
