
from src.convnet.pipes import train
import yaml 
import sys
import os 
from fastapi import FastAPI, Request
import uvicorn 

# web page templating
from pyngrok import ngrok
import nest_asyncio
from fastapi.responses import HTMLResponse     
from fastapi.templating import Jinja2Templates

# get jinja templates' path
templates = Jinja2Templates(directory="templates")

# instantiate web server
app = FastAPI()

with open("conf/parameters.yml") as file:
    PARAMS = yaml.safe_load(file)

@app.get("/")
def home():
    return {"Home"}

@app.get("/train")
async def train(PARAMS):
    train.run(params=PARAMS)

@app.get("/predict")
async def predict(PARAMS):
    pass

@app.get("/items/{id}", response_class=HTMLResponse)
async def read_item(request:Request, id:str):
    return templates.TemplateResponse("item.html", {"request":request, "id":id})


if __name__ == "__main__":
    """
    entry point
    """
    if sys.argv[0]=="train":
        
        train(PARAMS)

    else:

        # serve model        
        os.environ["MODEL_PATH"] = PARAMS["MODEL_PATH"]
        # ngrok_tunnel = ngrok.connect(8502)
        # print('\nModel served HERE on Public URL:', ngrok_tunnel.public_url, "\n")    
        # os.system("bash serve.sh")
        os.system("bg nohup tensorflow_model_server \
            --rest_api_port=8502 \
                --model_name=model \
                    --model_base_path=/root/convnet/model > logs/tf_server.out 2>&1")
        
        # expose the localhost to the net w/ a temporary public URL
        ngrok_tunnel = ngrok.connect(8000)
        print('\nExposed HERE on Public URL:', ngrok_tunnel.public_url, "\n")
        nest_asyncio.apply()
        uvicorn.run(app, port=8000)

