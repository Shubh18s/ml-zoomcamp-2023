import requests

# url = 'http://localhost:9696/predict'
url = 'http://localhost:8080/predict'

data = {'url': "https://raw.githubusercontent.com/alexeygrigorev/clothing-dataset-small/master/test/pants/0dfec862-c49f-430b-a6ef-c7ceb187225e.jpg"}

result = requests.post(url, json=data).json()
print(result)