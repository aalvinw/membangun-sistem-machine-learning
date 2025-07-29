# Gunakan image Python sebagai base
FROM python:3.10

# Set direktori kerja di dalam container
WORKDIR /app

# Salin semua file project ke dalam container
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install pandas scikit-learn mlflow

# Jalankan skrip saat container start
ENTRYPOINT ["python", "modelling.py"]
