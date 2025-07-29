FROM python:3.12-slim

WORKDIR /app
COPY prometheus_exporter.py .

RUN pip install prometheus_client

CMD ["python", "prometheus_exporter.py"]
