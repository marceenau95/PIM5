
# Importo librerias
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from  xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_recall_curve, auc,classification_report
from sklearn.model_selection import cross_validate
from imblearn.over_sampling import SMOTE
import joblib
from ft_engineering import ft_engineering

# defino función de modelos a evaluar

def build_models():
    models = [
        ("Logistic Regresion", LogisticRegression(max_iter=1000,random_state=42, class_weight="balanced")),
        ("RandomForest", RandomForestClassifier(n_estimators=100,random_state=42,class_weight="balanced")),
        ("GradientBoosting",GradientBoostingClassifier(n_estimators=100,random_state=42)),
        ("XGBoost",XGBClassifier(n_estimators=100,scale_pos_weight=(5/95), random_state=42)),  
        ("CatBoost",CatBoostClassifier(iterations=500,auto_class_weights='Balanced'))

    ]
    return models
 
 # defino métricas a evaluar por modelo

def sumarize_classification(y_true, y_pred,y_scores=None):
    metrics = {
        'accuracy': accuracy_score(y_true,y_pred),
        'f1-score': f1_score(y_true,y_pred,average='binary'),
        'recall': recall_score(y_true,y_pred,average='binary'),
        'auc': roc_auc_score(y_true,y_pred,multi_class='ovr'),
        'classification report':classification_report(y_true,y_pred)
    }

     # ROC-AUC y PR-AUC requieren probabilidades
    if y_scores is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        metrics['pr_auc'] = auc(recall, precision)
    return metrics

def mostrar_importancia_variables(model, preprocessing_pipeline, top_n=10):

    try:
        feature_names = preprocessing_pipeline.get_feature_names_out()

    except Exception:

        feature_names = []

        for nombre, trans, columnas in preprocessing_pipeline.transformers_:

            if nombre == "remainder":
                continue

            try:

                nombres = trans.get_feature_names_out(columnas)

            except Exception:

                nombres = columnas

            feature_names.extend(nombres)

    feature_names = np.array(feature_names)

    if hasattr(model, "feature_importances_"):

        importancias = model.feature_importances_

    elif hasattr(model, "coef_"):

        coef = model.coef_

        if coef.ndim == 1:
            importancias = np.abs(coef)
        else:
            importancias = np.sum(np.abs(coef), axis=0)

    else:

        print("El modelo no posee importancia de variables.")

        return None

    if len(importancias) != len(feature_names):

        n = min(len(importancias), len(feature_names))

        importancias = importancias[:n]

        feature_names = feature_names[:n]

    df = (
        pd.DataFrame(
            {
                "Variable": feature_names,
                "Importancia": importancias,
            }
        )
        .sort_values("Importancia", ascending=False)
        .reset_index(drop=True)
    )

    print("\nTop variables")

    print(df.head(top_n))

    return df

    

# Defino función para entrenar y seleccionar el mejor modelo

def train_and_select_model(X_train, y_train, X_test, y_test, preprocessing_pipeline):
    models = build_models()
    cv_results = []
    global_results = []
    report_results = []
    modelos_entrenados = {}

    for name, model in models:
        # --- Validación cruzada ---
        cv_scores = cross_validate(
            model, X_train, y_train,
            cv=3,
            scoring=['f1', 'recall', 'precision'],
            return_train_score=False
        )

        # --- Entrenar modelo ---
        model.fit(X_train, y_train)
        modelos_entrenados[name] = model  

        y_pred = model.predict(X_test)

        # --- Probabilidades ---
        try:
            y_scores = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_scores = None

        # --- Métricas ---
        metrics = sumarize_classification(y_test, y_pred, y_scores)

        # --- Población de Tabla 1: Cross Validation ---
        cv_results.append({
            'Modelo': name,
            'F1_mean': cv_scores['test_f1'].mean(),
            'F1_std': cv_scores['test_f1'].std(),
            'Recall_mean': cv_scores['test_recall'].mean(),
            'Recall_std': cv_scores['test_recall'].std(),
            'Precision_mean': cv_scores['test_precision'].mean(),
            'Precision_std': cv_scores['test_precision'].std()
        })

        # --- Población de Tabla 2: Métricas globales ---
        global_results.append({
            'Modelo': name,
            'Accuracy': metrics['accuracy'],
            'F1-score': metrics['f1-score'],
            'Recall': metrics['recall'],
            'AUC': metrics['auc'],
            'ROC-AUC': metrics.get('roc_auc', None),
            'PR-AUC': metrics.get('pr_auc', None)
        })

        # --- Población de Tabla 3: Classification Report ---
        report_results.append({
            'Modelo': name,
            'Classification Report': metrics['classification report']
        })

    # Convertir a DataFrames
    df_cv = pd.DataFrame(cv_results)
    df_global = pd.DataFrame(global_results)
    df_report = pd.DataFrame(report_results)

    # Mostrar resultados
    mostrar_resultados(df_cv, df_global, df_report)

    # Selección del mejor modelo
    if df_global['ROC-AUC'].notnull().any():
        best_model_row = df_global.loc[df_global['ROC-AUC'].idxmax()]
        criterio = "ROC-AUC"
    else:
        best_model_row = df_global.loc[df_global['F1-score'].idxmax()]
        criterio = "F1-score"

    best_model_name = best_model_row['Modelo']
    best_model = modelos_entrenados.get(best_model_name)

    if best_model is None:
        print(f" Nombre {best_model_name} no encontrado en modelos_entrenados. Claves disponibles: {list(modelos_entrenados.keys())}")
    
    # Mostrar importancia de variables automáticamente
    mostrar_importancia_variables(best_model, preprocessing_pipeline, top_n=10)

    print(f"\n Mejor modelo seleccionado según {criterio}:")
    print(best_model_row)

    return df_cv, df_global, df_report, best_model_row, best_model

# Defino función para imprimir resultados en pantalla

def mostrar_resultados(df_cv, df_global, df_report):
    # --- Tabla 1: Cross Validation ---
    print("\n" + "="*60)
    print(" Resultados de Cross Validation")
    print("="*60)
    print(df_cv.to_string(index=False))  # muestra todo sin truncar

    # --- Tabla 2: Métricas globales ---
    print("\n" + "="*60)
    print(" Métricas Globales en Test")
    print("="*60)
    print(df_global.to_string(index=False))

    # --- Tabla 3: Classification Reports ---
    print("\n" + "="*60)
    print(" Classification Reports por Modelo")
    print("="*60)
    for _, row in df_report.iterrows():
        print(f"\n--- {row['Modelo']} ---")
        print(row['Classification Report'])


# Preprocesamiento de variables categoricas y numéricas 

# obtener datos preprocesados
X_train, X_test, y_train, y_test, cleaning_pipeline,preprocessing_pipeline = ft_engineering()

# Aplicación de SMOTE para desbalanceo de clases

print("Distribución original:", y_train.value_counts(normalize=True))
smote = SMOTE(sampling_strategy=0.5,random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# entrenamiento de modelos
df_cv, df_global, df_report, best_model_row, best_model = train_and_select_model(
    X_train_res,
    y_train_res,
    X_test,
    y_test,
    preprocessing_pipeline
)
# Guardar best_model como pkl

# Ruta absoluta del directorio donde está este script
BASE_DIR = os.path.dirname(__file__)

# Ruta a la carpeta models
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# Guardar archivos
joblib.dump(best_model,
            os.path.join(MODELS_DIR, "model.pkl"))



