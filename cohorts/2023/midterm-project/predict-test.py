#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://localhost:9696/predict'

customer_id = 'xyz-123'
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


response = requests.post(url, json=person).json()
print(response)

if response['high_income'] == True:
  print('Sending promo email to %s' % (customer_id))
else:
  print('Not sending promo email to %s' % (customer_id))