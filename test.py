import requests

url = 'http://localhost:3500/api/predict'
file_path = r'C:\Users\sabir.ali_infobeans.IB-SEZ-LAP-505.000\Downloads\test_dataset.csv'

files = {'file': ('test_dataset.csv', open(file_path, 'rb'))}

response = requests.post(url, files=files)

print(response.text)
