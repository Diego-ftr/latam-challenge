import fastapi
from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import List
import pandas as pd
import logging
from challenge.model import DelayModel

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar el modelo
model = DelayModel()

app = fastapi.FastAPI()

# Cambiar el código de error de validación de 422 a 400
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"detail": "Validation error"}
    )

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

# Definir modelos de datos para validación
class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int
    
    @validator('MES')
    def mes_valido(cls, v):
        if v < 1 or v > 12:
            raise ValueError('MES debe estar entre 1 y 12')
        return v
    
    @validator('TIPOVUELO')
    def tipovuelo_valido(cls, v):
        tipos_validos = ['N', 'I']  # Nacional e Internacional
        if v not in tipos_validos:
            raise ValueError(f'TIPOVUELO debe ser uno de: {tipos_validos}')
        return v
    
    @validator('OPERA')
    def opera_valida(cls, v):
        # Lista de operadoras válidas basada en los datos del modelo
        operadoras_validas = [
            "Grupo LATAM",
            "Sky Airline",
            "Aerolineas Argentinas",
            "Copa Air",
            "Latin American Wings",
            "Avianca",
            "JetSmart SPA",
            "Gol Trans",
            "American Airlines",
            "Air Canada",
            "Iberia",
            "Delta Air",
            "United Airlines",
            "Oceanair Linhas Aereas",
            "Alitalia",
            "K.L.M.",
            "Air France",
            "British Airways",
            "Qantas Airways",
            "Lacsa",
            "Austral",
            "Plus Ultra Lineas Aereas",
            "Aerolineas Galapagos (Aerogal)"
        ]
        if v not in operadoras_validas:
            raise ValueError(f'OPERA no reconocida: {v}')
        return v

class PredictRequest(BaseModel):
    flights: List[Flight]

@app.post("/predict", status_code=200)
async def post_predict(request: PredictRequest) -> dict:
    try:
        # Convertir la lista de vuelos a DataFrame
        flights_data = [flight.dict() for flight in request.flights]
        df = pd.DataFrame(flights_data)
        
        # Agregar columnas dummy requeridas por el modelo
        # El modelo espera columnas de fechas, las simularemos
        df['Fecha-I'] = '2022-01-01 10:00:00'  # Fecha dummy
        df['Fecha-O'] = '2022-01-01 10:30:00'  # Fecha dummy con 30 min de diferencia
        
        # Preprocesar los datos
        logger.info(f"Procesando {len(df)} vuelos para predicción")
        features = model.preprocess(df)
        
        # Realizar predicciones
        predictions = model.predict(features)
        
        logger.info(f"Predicciones completadas: {predictions}")
        
        return {"predict": predictions}
        
    except ValueError as e:
        logger.error(f"Error de validación: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")