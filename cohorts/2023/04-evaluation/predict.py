import pickle

from flask import Flask
from flask import request
from flask import jsonify

input_file = 'model_C=1.0.bin'

with open(input_file, 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

app = Flask('carprice prediction')

@app.route('/predict', methods=['POST'])
def predict():
  car = request.get_json()
  (churn_prob, churn_result) = generate_prediction(car)

  result = {
    'churn_probability' : float(churn_prob),
    'above_average' : bool(churn_result)
  }
  return jsonify(result)

def generate_prediction(car):
  X = dv.transform([car])
  y_pred = model.predict_proba(X)[0, 1]
  above_average = y_pred >= 0.5

  return (y_pred, above_average)

if __name__ == "__main__":
  app.run(debug=True, host='0.0.0.0', port=9696)
