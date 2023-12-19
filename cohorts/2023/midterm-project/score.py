import pickle
from flask import Flask
from flask import request
from flask import jsonify

input_file = 'model_C=1.0.bin'

with open(input_file, 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

app = Flask('high income prediction')

@app.route('/predict', methods=['POST'])
def predict():
  person = request.get_json()
  (high_income_probability, high_income) = generate_prediction(person)

  result = {
    'high_income_probability' : float(high_income_probability),
    'high_income' : bool(high_income)
  }
  return jsonify(result)

def generate_prediction(person):
  X = dv.transform([person])
  y_pred = model.predict_proba(X)[0, 1]
  high_income = y_pred >= 0.5

  return (y_pred, high_income)

if __name__ == "__main__":
  app.run(debug=True, host='0.0.0.0', port=9696)