#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://localhost:9696/predict'

customer_id = 'xyz-123'
person = {'make': 'aston_martin',
  'model': 'v8_vantage',
  'transmission_type': 'automated_manual',
  'vehicle_style': 'convertible',
  'year': 2014,
  'engine_hp': 420.0,
  'engine_cylinders': 8.0,
  'highway_mpg': 21,
  'city_mpg': 14}


response = requests.post(url, json=car).json()
print(response)

if response['above_average'] == True:
  print('Sending promo email to %s' % (customer_id))
else:
  print('Not sending promo email to %s' % (customer_id))


