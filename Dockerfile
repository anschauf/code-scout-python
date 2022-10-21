# FROM jupyter/datascience-notebook:aarch64-lab-3.4.7
FROM arm64v8/python:3.10.7

ENV HOME="/home/jovyan/"

RUN python3 -m pip install --upgrade pip

COPY requirements.txt .
COPY constraints.txt .
RUN pip3 install -r requirements.txt -c constraints.txt

#WORKDIR "/home/jovyan/work"
#COPY . .
