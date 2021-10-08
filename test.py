
from src.convnet.nodes.etl import load_dataset
from src.convnet.pipes import prep
import json 
import requests

def run():
    (X_train, Y_train), (X_test, Y_test) = load_dataset(name="fashion_mnist")
    (X_train, X_test) = prep.run(X_train, X_test)
    return (X_train, X_test, Y_train, Y_test)


def probe(X_test):
    data = json.dumps({"signature_name": "serving_default", "instances": X_test[0:3].tolist()})
    headers = {"content-type":"applicaton/json"}
    
    # submit request and collect response
    json_response = requests.post(f"http://localhost:8504/v1/models/model:predict", data=data, headers=headers)
    predictions = json.loads(json_response.text)["predictions"]