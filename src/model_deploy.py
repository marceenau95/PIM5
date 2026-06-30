# src/ model_deploy.py

# Librerias
import pandas as pd
import numpy as pd
import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


# Inicializacióm de aplicación FastAPI
app = FastAPI(
    title = " API de predicción de pago a tiempo",
    description = "Esta API permite predecir si un cliente pagará a tiempo o no",
    version = "1.1.1"
)

# Cargar modelo pkl de la carpeta models

try:
    # Cargar modelo desde archivo pkl
    with open("../models/model.pkl","rb") as f:
        modelo = pickle.load(f)
        print("Modelo cargado correctamente")
except Exception as e:
    print(f"error al cargar el modelo:{e}")
    modelo = None

# Definimos endpoints

@app.get("/saludo")
def saludo():
    return {"mensaje":"Hola! esta es una API para predecir si un cliente pagará a tiempo o no"}

# defino endpoint para hacer predicciones
@app.post("/predict")
def predict_batch(input_data:dict):
    if modelo is None:
        return {"El modelo no se cargó, revisar los logs del servidor para más detalles"}
    
    try:
        return{"El modelo está cargado y listo para hacer predicciones"}
    except Exception as e:
        return {"Error al hacer predicciones"}