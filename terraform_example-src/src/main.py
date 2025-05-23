from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List
import os
import boto3
import joblib
import tempfile
from datetime import datetime, timedelta
import numpy as np
import calendar
import uvicorn 

# --- Configuración ---
S3_BUCKET_NAME = os.environ.get("S3_MODEL_BUCKET", "proyecto-2-0953476d60561c955")
PM25_MODEL_KEY = "models/best_model_pm25.pkl"
PM10_MODEL_KEY = "models/best_model_pm10.pkl"

LOCAL_PM25_PATH = os.path.join(tempfile.gettempdir(), "best_model_pm25.pkl")
LOCAL_PM10_PATH = os.path.join(tempfile.gettempdir(), "best_model_pm10.pkl")

app = FastAPI()

# --- Modelos globales ---
model_pm25 = None
model_pm10 = None

# --- Clases Pydantic ---
class FechaInput(BaseModel):
    fecha: str  # formato: 'YYYY-MM-DD'

class DiaPrediccion(BaseModel):
    fecha: str
    pm25: float
    pm10: float
    calidad_aire: str

class PrediccionRespuesta(BaseModel):
    predicciones: List[DiaPrediccion]

# --- Cargar modelos desde S3 ---
def cargar_modelos():
    global model_pm25, model_pm10
    s3 = boto3.client("s3")

    s3.download_file(S3_BUCKET_NAME, PM25_MODEL_KEY, LOCAL_PM25_PATH)
    s3.download_file(S3_BUCKET_NAME, PM10_MODEL_KEY, LOCAL_PM10_PATH)

    model_pm25 = joblib.load(LOCAL_PM25_PATH)
    model_pm10 = joblib.load(LOCAL_PM10_PATH)

# --- Generar características ---
def extraer_features(date_obj):
    day_of_week = date_obj.weekday()
    day_of_year = date_obj.timetuple().tm_yday
    is_weekend = int(day_of_week >= 5)
    month = date_obj.month

    # Temporada (codificada como número)
    season = (
        0 if month in [12, 1, 2] else
        1 if month in [3, 4, 5] else
        2 if month in [6, 7, 8] else
        3
    )

    return [date_obj.year, date_obj.month, date_obj.day, day_of_week, day_of_year, is_weekend, season]

# --- Clasificar calidad del aire ---
def clasificar_calidad(pm25, pm10):
    if pm25 <= 12 and pm10 <= 50:
        return "Buena"
    elif pm25 <= 35.4 and pm10 <= 100:
        return "Moderada"
    else:
        return "Mala"

# --- Endpoint principal de predicción ---
@app.post("/predict", response_model=PrediccionRespuesta)
async def predecir_calidad(input: FechaInput):
    if model_pm25 is None or model_pm10 is None:
        raise HTTPException(status_code=500, detail="Modelos no cargados.")

    try:
        fecha_base = datetime.strptime(input.fecha, "%Y-%m-%d")
        predicciones = []

        for i in range(7):
            fecha_pred = fecha_base + timedelta(days=i)
            features = np.array([extraer_features(fecha_pred)])

            pred_pm25 = model_pm25.predict(features)[0]
            pred_pm10 = model_pm10.predict(features)[0]
            calidad = clasificar_calidad(pred_pm25, pred_pm10)

            predicciones.append(DiaPrediccion(
                fecha=fecha_pred.strftime("%Y-%m-%d"),
                pm25=round(pred_pm25, 2),
                pm10=round(pred_pm10, 2),
                calidad_aire=calidad
            ))

        return {"predicciones": predicciones}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Health check ---
@app.get("/health")
async def health_check():
    estado = {
        "status": "ok",
        "pm25_model": "loaded" if model_pm25 else "not loaded",
        "pm10_model": "loaded" if model_pm10 else "not loaded"
    }
    return estado

# --- Servir frontend HTML ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")

app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return FileResponse("static/index.html")

# --- Cargar modelos al iniciar ---
@app.on_event("startup")
def startup_event():
    cargar_modelos()

# --- Para correr localmente ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
