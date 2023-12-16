python app.py (server will start at 3000)
just use simple curl command to send test_data.csv file:
curl -X POST -F "file=@test_dataset.csv" http://localhost:3000/api/predict
(it will return data in json format)


