import pickle

from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model2.bin'
dv_file = 'dv.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)

app = Flask('credit prediction')

@app.route('/predict', methods=['POST'])
def predict():
  entry = request.get_json()
  (credit_probability, get_credit) = generate_prediction(entry)

  result = {
    'credit_probability' : float(credit_probability),
    'get_credit' : bool(get_credit)
  }
  return jsonify(result)

def generate_prediction(entry):
  X = dv.transform([entry])
  y_pred = model.predict_proba(X)[0, 1]
  get_credit = y_pred >= 0.5

  return (y_pred, get_credit)

if __name__ == "__main__":
  app.run(debug=True, host='0.0.0.0', port=9696)
