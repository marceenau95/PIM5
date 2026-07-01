# src/ model_deploy.py

# Librerias
import os 
import pandas as pd
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.custom_transformers import (
    CustomCleaner,
    DataFrameTransformer
)


# Inicializacióm de aplicación FastAPI
app = FastAPI(
    title = " API de predicción de pago a tiempo",
    description = "Esta API permite predecir si un cliente pagará a tiempo o no",
    version = "1.1.1"
)

# Cargar modelo pkl de la carpeta models

BASE_DIR = os.path.dirname(__file__)

MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

try:

    model_pkl = joblib.load(
        os.path.join(
            MODELS_DIR,
            "model.pkl"
        )
    )

    print("Pipeline cargado correctamente.")

except Exception as e:

    print(f"Error cargando pipeline: {e}")

    model_pkl = None

# ============================================================
# Esquema de entrada
# ============================================================

class Cliente(BaseModel):

    tipo_credito: int
    capital_prestado: float
    salario_cliente: float
    plazo_meses: int
    edad_cliente: int
    tipo_laboral: str
    puntaje_datacredito: float
    huella_consulta: float
    saldo_mora: float
    saldo_total: float
    creditos_sectorFinanciero: float
    creditos_sectorCooperativo: float
    creditos_sectorReal: float
    tendencia_ingresos: str
    puntaje: float


# Definimos endpoints

@app.get("/saludo")
def saludo():
    return {"mensaje":"Hola! esta es una API para predecir si un cliente pagará a tiempo o no"}

# defino endpoint para hacer predicciones
@app.post("/predict")
def predict_batch(clientes:list[Cliente]):
    if model_pkl is None:
        return {"El modelo no se cargó, revisar los logs del servidor para más detalles"}
    
    try:
        # Convertir JSON a DataFrame

        df = pd.DataFrame(

            [cliente.model_dump() for cliente in clientes]

        )

        # Predicción

        predicciones = model_pkl.predict(df)

        resultados = []

        for pred in predicciones:

            resultados.append(

                {

                    "prediccion": int(pred),

                    "resultado":

                        "Pagará a tiempo"

                        if pred == 1

                        else

                        "No pagará a tiempo"

                }

            )

        return {

            "cantidad_registros": len(resultados),

            "predicciones": resultados

        }

    except Exception as e:
        raise HTTPException(

            status_code=500,

            detail=str(e))
        

# ============================================================
# Ejecutar API
# ============================================================

if __name__ == "__main__":

    uvicorn.run(

        "model_deploy:app",

        host="127.0.0.1",

        port=8000,

        reload=True

    )