import os
import grpc

from keras_image_helper import create_preprocessor

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from flask import Flask
from flask import request
from flask import jsonify

from proto import np_to_protobuf

host = os.getenv('TF_SERVING_HOST', '0.0.0.0:8500')
channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
preprocessor = create_preprocessor('xception', target_size=(299, 299))


classes = ['dress',
 'hat',
 'longsleeve',
 'outwear',
 'pants',
 'shirt',
 'shoes',
 'shorts',
 'skirt',
 't-shirt']


def prepare_request(X):
    pb_request = predict_pb2.PredictRequest()

    pb_request.model_spec.name = 'clothing-model'
    pb_request.model_spec.signature_name = 'serving_default'

    pb_request.inputs['input_22'].CopyFrom(np_to_protobuf(X))
    return pb_request


def prepare_response(pb_response):
    preds = pb_response.outputs['dense_20'].float_val
    return dict(zip(classes, preds))

def predict(url):
    X = preprocessor.from_url(url)
    pb_request = prepare_request(X)
    pb_response = stub.Predict(pb_request, timeout=20.0)
    response = prepare_response(pb_response)
    return response


app = Flask('gateway')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    url = data['url']
    result = predict(url)
    return jsonify(result)

if __name__ == '__main__':
    url = "https://raw.githubusercontent.com/alexeygrigorev/clothing-dataset-small/master/test/pants/0dfec862-c49f-430b-a6ef-c7ceb187225e.jpg"
    response = predict(url)
    print(response)
    # app.run(debug=True, host='0.0.0.0', port=9696)