import requests
url = 'http://localhost:8080/predict'
data = {'url':"https://storage.googleapis.com/kagglesdsdata/datasets/130737/312053/Mushrooms/Agaricus/001_2jP9N_ipAo8.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20240109%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240109T123315Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=c313106f5c2796100994472df76254aba79e47db3ae27247cc0daac3323229a61a73ec8c6f6c3de29873f65ccf3fe643189c5bbbedbf40e40ecb234088acb80fd1618b5a4ce409280140cffe7d99b75869edf1e91a64e223f6e4ee8a83d293b591ff01fbcb8638885200384caacab9ea2de093b691fa12eb4c13bd22182ba81a60d3ca6bbfe3417e774bc8e8dbd063c9b18967536aa594d7c5af68b4aa9c768092ca28da6b35eff936198df19c07f5c356b9b238e10a0fa0d47232451dfc6a4ef3a5a90d39bc497e25357f580ba2765e9edef7d72d8d870b883a8d97a1bc0e25bb1ad5c7d134fb2d2978cecefeea6e1dbe97eddd9e349f279f6259a14d50c1db"}
response = requests.post(url,json=data).json()
print(response)