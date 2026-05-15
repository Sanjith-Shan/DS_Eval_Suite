FROM python:3.11-slim

RUN pip install --no-cache-dir \
        numpy==1.26.4 \
        pandas==2.2.2 \
        python-dateutil==2.9.0 \
        pytz==2024.1

WORKDIR /workspace
COPY Q1.csv /workspace/Q1.csv
COPY Q2.csv /workspace/Q2.csv
COPY Q3.csv /workspace/Q3.csv
COPY README.md /workspace/README.md

RUN mkdir -p /output /logs/verifier
