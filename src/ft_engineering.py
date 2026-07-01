# ft_engineering.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from src.cargar_datos import cargarDatos
from src.custom_transformers import CustomCleaner
# ==========================================================
# VARIABLES DEL MODELO
# ==========================================================

TARGET = "Pago_atiempo"

FEATURES = [
    'tipo_credito',
    'capital_prestado',
    'salario_cliente',
    'plazo_meses',
    'edad_cliente',
    'tipo_laboral',
    'puntaje_datacredito',
    'huella_consulta',
    'saldo_mora',
    'saldo_total',
    'creditos_sectorFinanciero',
    'creditos_sectorCooperativo',
    'creditos_sectorReal',
    'tendencia_ingresos',
    'puntaje'
]

NUMERIC_FEATURES = [
    'capital_prestado',
    'salario_cliente',
    'plazo_meses',
    'edad_cliente',
    'puntaje_datacredito',
    'huella_consulta',
    'saldo_mora',
    'saldo_total',
    'creditos_sectorFinanciero',
    'creditos_sectorCooperativo',
    'creditos_sectorReal',
    'puntaje'
]

CATEGORICAL_FEATURES = [
    'tipo_credito',
    'tipo_laboral',
    'tendencia_ingresos'
]


# ==========================================================
# PIPELINE DE LIMPIEZA
# ==========================================================

def build_cleaning_pipeline():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    cleaning_pipeline = Pipeline([

        ("cleaner", CustomCleaner()),

        ("imputer",

            ColumnTransformer([

                ("num", numeric_pipeline, NUMERIC_FEATURES),

                ("cat", categorical_pipeline, CATEGORICAL_FEATURES)

            ],
            remainder="passthrough")

        )

    ])

    return cleaning_pipeline

# ==========================================================
# PIPELINE DE PREPROCESAMIENTO
# ==========================================================

def build_preprocessing_pipeline():

    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )

    categorical_pipeline = Pipeline([
        ("encoder", encoder)
    ])

    preprocessing = ColumnTransformer(

        [

            ("num", numeric_pipeline, NUMERIC_FEATURES),

            ("cat", categorical_pipeline, CATEGORICAL_FEATURES)

        ],

        remainder="drop"

    )

    return preprocessing
# ==========================================================
# BUILD PIPELINE
# ==========================================================

def build_pipeline(df):

    df = df.copy()

    df = df[
        (df["edad_cliente"] <= 100)
        & (df["tipo_credito"] != 68)
    ]

    df = df[FEATURES + [TARGET]]

    X = df[FEATURES]
    y = df[TARGET]

    cleaning_pipeline = build_cleaning_pipeline()

    preprocessing_pipeline = build_preprocessing_pipeline()

    return (
        df,
        X,
        y,
        cleaning_pipeline,
        preprocessing_pipeline
    )
# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

def ft_engineering():

    df = cargarDatos()

    (
        df,
        X,
        y,
        cleaning_pipeline,
        preprocessing_pipeline
    ) = build_pipeline(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # -----------------------
    # LIMPIEZA
    # -----------------------

    X_train_clean = cleaning_pipeline.fit_transform(X_train)

    X_test_clean = cleaning_pipeline.transform(X_test)

    # Recuperar DataFrame

    columnas = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X_train_clean = pd.DataFrame(
        X_train_clean,
        columns=columnas,
        index=X_train.index
    )

    X_test_clean = pd.DataFrame(
        X_test_clean,
        columns=columnas,
        index=X_test.index
    )

    # Convertir nuevamente tipos

    for c in CATEGORICAL_FEATURES:
        X_train_clean[c] = X_train_clean[c].astype("object")
        X_test_clean[c] = X_test_clean[c].astype("object")

    for c in NUMERIC_FEATURES:
        X_train_clean[c] = pd.to_numeric(X_train_clean[c])
        X_test_clean[c] = pd.to_numeric(X_test_clean[c])

    # -----------------------
    # PREPROCESAMIENTO ML
    # -----------------------

    X_train_processed = preprocessing_pipeline.fit_transform(X_train_clean)

    X_test_processed = preprocessing_pipeline.transform(X_test_clean)

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        cleaning_pipeline,
        preprocessing_pipeline
    )