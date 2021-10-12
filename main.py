import sys

import uvicorn
import yaml
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.datastructures import UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.convnet.nodes.etl import read_convert_image
from src.convnet.nodes.prep import to_gray
from src.convnet.pipes import inference, train

# web page templating
# get jinja templates' path
templates = Jinja2Templates(directory="templates")

# instantiate web server
app = FastAPI()

# get parameters
params_path = "conf/parameters.yml"
with open(params_path) as file:
    PARAMS = yaml.safe_load(file)


@app.get("/")
def home():
    return {"Home"}


@app.post("/predict")
async def predict(img: UploadFile = File(...)):

    # handle exceptions
    if img is None or img.file is None:
        raise HTTPException(
            status_code=400,
            detail="Please provide an image",
        )
    ext = img.filename.split(".")[-1] in (
        "jpg",
        "jpeg",
        "png",
    )
    # check extension
    if not ext:
        raise HTTPException(
            status_code=400,
            detail="Please load a .jpg or .png",
        )
    # preprocesisng
    # make (height, width, 3) RGB
    img_data = read_convert_image(
        img.file.read(), height=28, width=28
    )
    # make (height, width, 1) gray
    img_data = to_gray(img_data)
    predicted = inference.run(img_data)
    return {"prediction:", predicted}


@app.get("/items/{id}", response_class=HTMLResponse)
async def read_item(request: Request, id: str):
    return templates.TemplateResponse(
        "item.html", {"request": request, "id": id}
    )


if __name__ == "__main__":
    """
    entry point
    usage:
        # training
        python main.py train
    """
    args = sys.argv
    is_arg = len(args) == 2
    if is_arg and args[1] == "train":
        print("training")
        train.run(params=PARAMS)
    if is_arg and args[1] == "web_serve":
        print("serving web")
        uvicorn.run(app, port=8000)
