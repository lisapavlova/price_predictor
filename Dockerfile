
from ubuntu:20.04
MAINTAINER Elizaveta Pavlova
RUN apt-get update -y
COPY . /opt/price_predictor
WORKDIR /opt/price_predictor
RUN apt install -y python3-pip
RUN pip3 install -r requirements.txt
CMD python3 app.py
