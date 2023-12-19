import pickle

input_file = 'model_C=1.0.bin'

with open(input_file, 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

person = {'workclass': 'private',
  'education': '11th',
  'marital_status': 'never_married',
  'occupation': 'machine_op_inspct',
  'relationship': 'own_child',
  'race': 'black',
  'sex': 'male',
  'native_country': 'united_states',
  'age': 25,
  'capital_gain': 0,
  'capital_loss': 0,
  'hours_per_week': 40}

X = dv.transform([person])

print(model.predict_proba(X)[0, 1])