import os
import sys

import nest_asyncio
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# web page templating
from pyngrok import ngrok

from src.convnet.pipes import predict
from src.convnet.pipes import train as training

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
async def train():
    train.run(params=PARAMS)


@app.get("/predict")
async def predict():
    predict.run()


@app.get("/items/{id}", response_class=HTMLResponse)
async def read_item(request: Request, id: str):
    return templates.TemplateResponse(
        "item.html", {"request": request, "id": id}
    )


if __name__ == "__main__":
    """
    entry point
    """
    if sys.argv[1] == "train":
        # train
        print("training")
        training.run(params=PARAMS)
    else:
        # expose the localhost to the net w/ a temporary public URL
        ngrok_tunnel = ngrok.connect(8000)
        print(
            "\nExposed HERE on Public URL:",
            ngrok_tunnel.public_url,
            "\n",
        )
        nest_asyncio.apply()
        uvicorn.run(app, port=8000)
