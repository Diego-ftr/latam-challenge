# syntax=docker/dockerfile:1.2
# Stage 1: Builder - Entrenar modelo
FROM python:3.10-slim as builder

WORKDIR /build

# Instalar dependencias del sistema para ML
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y instalar dependencias
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código y datos
COPY challenge/ ./challenge/
COPY data/data.csv ./data/data.csv

# Entrenar modelo
RUN python -c "\
import pandas as pd; \
from challenge.model import DelayModel; \
model = DelayModel(auto_train_for_tests=False); \
data = pd.read_csv('data/data.csv', dtype={'Vlo-I': 'object', 'Vlo-O': 'object'}); \
features, target = model.preprocess(data, target_column='delay'); \
model.fit(features, target); \
print('Modelo entrenado y guardado exitosamente')"

# Stage 2: Runtime - Solo API y modelo entrenado
FROM python:3.10-slim as runtime

WORKDIR /app

# Copiar solo requirements de runtime
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY challenge/ ./challenge/

# Copiar modelo entrenado desde builder stage
COPY --from=builder /build/challenge/model.pkl ./challenge/model.pkl

# Exponer puerto
EXPOSE 8080

# Comando para ejecutar la aplicación
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]