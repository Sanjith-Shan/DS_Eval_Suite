FROM python:3.11-slim

RUN pip install --no-cache-dir numpy==1.26.4 pandas==2.2.2

WORKDIR /workspace
COPY sensor_data.csv /workspace/sensor_data.csv

RUN mkdir -p /output /logs/verifier
