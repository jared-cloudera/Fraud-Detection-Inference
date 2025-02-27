from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from pprint import pprint

app = Flask(__name__)

# Load your ONNX model
session = ort.InferenceSession("fraud_classifier.onnx")

# Get the name of the input node
input_name = session.get_inputs()[0].name

@app.route('/predict', methods=['POST'])
def predict():
    print("got request")
    data = request.get_json(force=True)
    pprint(data)
    # Expecting the input data as a list of lists, for example
    input_data = np.array(data["input"], dtype=np.float32)
    print("input data")
    pprint(input_data)
    # Run inference
    preds = session.run(None, {input_name: input_data})

    print("preds")
    pprint(preds)
    return jsonify({"prediction": preds[0].tolist()})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8181)
