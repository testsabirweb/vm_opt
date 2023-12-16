## Installation


```bash
pip install -r requirements.txt
```

## Usage

```
python3 app.py
```

It will start the server at port 3000. To test it just use curl.

```
curl -X POST -F "file=@test_dataset.csv" http://localhost:3000/api/predict 
```
(it will return data in json format)