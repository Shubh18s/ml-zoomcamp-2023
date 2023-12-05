#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://localhost:9696/predict'

customer_id = 'xyz-123'
client = {"job": "unknown", "duration": 270, "poutcome": "failure"}

response = requests.post(url, json=client).json()
print(response)

# if response['above_average'] == True:
#   print('Sending promo email to %s' % (customer_id))
# else:
#   print('Not sending promo email to %s' % (customer_id))


