FROM tiangolo/uvicorn-gunicorn-starlette:python3.7


RUN pip install fastai aiohttp

RUN pip install jinja2

RUN pip install starlette

COPY ./app /app

WORKDIR /app

EXPOSE 80
