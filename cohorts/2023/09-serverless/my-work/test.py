import requests

url = 'http://localhost:8080/2023-12-05/functions/function/invocations'

dataset = "clothing-dataset-small/test"
item = "pants"
file = "c8d21106-bbdb-4e8d-83e4-bf3d14e54c16.jpg"
path = f'{dataset}/{item}/{file}'

data = {'path': path}

result = requests.post(url, json=data).json()
print(result)