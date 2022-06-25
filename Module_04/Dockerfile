FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile","Pipfile.lock","./"]


RUN pipenv install --system --deploy

COPY ["score_agri_docker.py","fhv_tripdata_2021-04.csv","./"]

RUN python score_agri_docker.py fhv 2021 4

