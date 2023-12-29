## Installation


```bash
pip install -r requirements.txt
```
## Note: For the first time we need to create modal in our local setup (if modal created move to next step directly)

```
python .\model_creation.py
```


## Usage

```
python app.py
```

It will start the server at port 3500. To test it just use curl.

```
curl -X POST -F "file=@test_dataset.csv" http://localhost:3500/api/predict 
```
(it will return data in json format)

## Access the Swagger UI at

```
http://localhost:3500/apidocs/
```
