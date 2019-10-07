from fastai.vision import *
from io import BytesIO
from starlette.middleware.cors import CORSMiddleware

import logging, sys

from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.templating import Jinja2Templates

import uvicorn

import aiohttp
import asyncio

import os

app = Starlette()

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

templates = Jinja2Templates(directory='templates')

# ----------------------------  LOAD LEARNER
file_path = 'export.pkl'
learner = load_learner('',file_path)
# learner = load_learner(path, file_path)
print('learer.data.classes = ', learner.data.classes)
# ----------------------------  end LOAD LEARNER


@app.middleware("http")
async def add_custom_header(request, call_next):
    logging.info("====infostart====")
    logging.info("====something====")
    logging.debug(request.headers)
    response = await call_next(request)
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Allow'] = 'OPTIONS, GET, POST'
    if ('origin' in request.headers.keys()):
        logging.debug("ORIGIN header found")
        response.headers['Access-Control-Allow-Origin'] = request.headers['origin']
    else:
        logging.debug("ORIGIN header NOT found")
        response.headers['Access-Control-Allow-Origin'] = '*'
    logging.info("====debugend====")
    return response


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    print('open_imgage = ', img)
    # out_class, out_index_tensor, probabilities = learner.predict(img)
    # max_index = int(out_index_tensor) 

    # make a dictionary of classes with probabilities
    classes = learner.data.classes
    _, _, probabilities = learner.predict(img)
    out_classification = dict(zip( classes, map(float, probabilities) ))
    # sort by value to have best prediction first in order
    out_classification = sorted(out_classification.items(), key=lambda kv: kv[1], reverse=True)
    print('classification completed: ', out_classification)
    return JSONResponse({
        "predictions": out_classification
    })


@app.route('/')
async def homepage(request):
    env = os.environ
    return templates.TemplateResponse('app.html', {'request': request, 'env': env})


class OptionsResponse(Response):
    media_type = None
    headers = {
            'Allow': 'OPTIONS, GET, POST',
    }


@app.route("/classify-url", methods=["OPTIONS"])
async def classify_url(request):
    headers = {'Allow': 'OPTIONS, GET, POST'}
    return OptionsResponse(None)

@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    print('\n\n------ /classify-url [GET]')
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)

@app.route("/classify-url", methods=["POST"])
async def classify_url(request):
    print('\n\n------ /classify-url [POST]')
    bytes = await request.body()
    return predict_image_from_bytes(bytes)

