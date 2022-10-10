FROM jupyter/datascience-notebook:aarch64-lab-3.4.7

RUN python3 -m pip install --upgrade pip

COPY requirements.txt .
COPY constraints.txt .

RUN PIP_CONSTRAINT=constraints.txt pip3 install -r requirements.txt
