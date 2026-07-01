import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Transformador personalizado de np_array a Dataframe
class DataFrameTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        return pd.DataFrame(
            X,
            columns=self.columns
        )

# Transformador personalizado de limpieza

# ==========================================================
# LIMPIEZA PERSONALIZADA
# ==========================================================

class CustomCleaner(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        # salario = 0 -> NaN
        if "salario_cliente" in X.columns:
            X["salario_cliente"] = X["salario_cliente"].replace(0, np.nan)

        # puntaje <=0 -> NaN
        if "puntaje" in X.columns:
            X["puntaje"] = X["puntaje"].mask(X["puntaje"] <= 0)

        # puntaje_datacredito <=0 -> NaN
        if "puntaje_datacredito" in X.columns:
            X["puntaje_datacredito"] = X["puntaje_datacredito"].mask(
                X["puntaje_datacredito"] <= 0
            )

        # tendencia_ingresos
        if "tendencia_ingresos" in X.columns:

            def limpiar_tendencia(valor):

                if pd.isna(valor):
                    return np.nan

                try:
                    float(str(valor).strip())
                    return np.nan
                except:
                    return valor

            X["tendencia_ingresos"] = X["tendencia_ingresos"].apply(
                limpiar_tendencia
            )

        return X
