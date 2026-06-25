import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from cargar_datos import cargarDatos


def ft_engineering():
    # cargar datos
    df = cargarDatos()

    # Filtrar registros inválidos
    df = df[(df['edad_cliente'] <= 100) & (df['tipo_credito'] != 68)].copy()

    # Seleccion Variables relevantes
    selected_features = [
        'tipo_credito', 'capital_prestado', 'salario_cliente',
        'plazo_meses', 'edad_cliente', 'tipo_laboral',
        'puntaje_datacredito', 'huella_consulta',
        'saldo_mora', 'saldo_total', 'creditos_sectorFinanciero',
        'creditos_sectorCooperativo', 'creditos_sectorReal',
        'tendencia_ingresos', 'Pago_atiempo', 'puntaje'
    ]
    df = df[selected_features]

    # Definición de funciones de limpieza de valores vacios/nulos
    def limpiar_salario(x):
        return np.nan if x == 0 else x

    def limpiar_puntaje(x):
        return np.nan if x <= 0 else x

    def limpiar_puntaje_datacredito(x):
        return np.nan if x <= 0 else x

    # Creación de un Transformador personalizado
    class CustomCleaner(BaseEstimator, TransformerMixin):

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            X = X.copy()

            # Aplicar reglas de limpieza
            if 'salario_cliente' in X.columns:
                X['salario_cliente'] = X['salario_cliente'].apply(limpiar_salario)
            if 'puntaje' in X.columns:
                X['puntaje'] = X['puntaje'].apply(limpiar_puntaje)
            if 'puntaje_datacredito' in X.columns:
                X['puntaje_datacredito'] = X['puntaje_datacredito'].apply(limpiar_puntaje_datacredito)

            # Asegurar que las categóricas sean strings (evita mezcla int/str)
            for col in ['tipo_credito', 'tipo_laboral', 'tendencia_ingresos']:
                if col in X.columns:
                    X[col] = X[col].astype(str)

            # NO eliminar columnas aquí si ColumnTransformer espera columnas concretas
            return X

    # split features/target
    X = df.drop('Pago_atiempo', axis=1)
    y = df['Pago_atiempo']

    # definir variables (basadas en X antes de pipeline)
    num_features = X.select_dtypes(include=['number']).columns.tolist()
    cat_features = X.select_dtypes(include=['object']).columns.tolist()

    # pipeline de limpieza
    cleaner = CustomCleaner()

    # pipeline numéricas 
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
        # Crear OneHotEncoder compatible con distintas versiones de sklearn
    try:
        # versiones nuevas (sklearn >= 1.2)
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        # versiones antiguas
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    # pipeline categóricas
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encode', ohe)
    ])

    # combinar
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ],
        remainder='drop'  # explícito: descarta columnas no listadas
    )

    # Pipeline completo
    pipeline = Pipeline(steps=[
        ('cleaner', cleaner),
        ('preprocessor', preprocessor)
    ])

    # split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # aplicar preprocesamiento
    X_train_processed = pipeline.fit_transform(X_train, y_train)
    X_test_processed = pipeline.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, pipeline

