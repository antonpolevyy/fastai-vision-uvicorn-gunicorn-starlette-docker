FROM tiangolo/uvicorn-gunicorn-starlette:python3.7


RUN pip install fastai

RUN pip install jinja2

RUN pip install starlette uvicorn python-multipart aiohttp

COPY ./app /app

WORKDIR /app

EXPOSE 80
