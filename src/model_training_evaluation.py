
# Importo librerias
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from  xgboost import XGBClassifier
from catboost import CatBoostClassifier
from ft_engineering import ft_engineering
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_recall_curve, auc,classification_report
from sklearn.model_selection import cross_validate
from imblearn.over_sampling import SMOTE

# defino función de modelos a evaluar



def build_models():
    models = [
        ("Logistic Regresion", LogisticRegression(max_iter=1000,random_state=42,)),
        ("RandomForest", RandomForestClassifier(n_estimators=100,random_state=42,class_weight="balanced")),
        ("GradientBoosting",GradientBoostingClassifier(n_estimators=100,random_state=42)),
        ("XGBoost",XGBClassifier(random_state=42)),
        ("CatBoost",CatBoostClassifier(scale_pos_weight=20))

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

# Defino función para entrenar y seleccionar el mejor modelo

def train_and_select_model(X_train,y_train,X_test,y_test):
    models = build_models()
    results =[]
    for name,model in models:
        # Validación cruzada
        cv_scores = cross_validate(
            model,X_train,y_train,
            cv=5,
            scoring=['f1', 'recall', 'precision'],
            return_train_score=False
        )
        # Entrenar modelo con todos los datos de train
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)

        # Probabilidades para ROC-AUC y PR-AUC
        try:
            y_scores = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_scores = None

        # Almacenar resultados
        result ={
            'name':name,
            'model':model,
            'cv_f1_mean': cv_scores['test_f1'].mean(),
            'cv_f1_std': cv_scores['test_f1'].std(),
            'cv_recall_mean': cv_scores['test_recall'].mean(),
            'cv_recall_std': cv_scores['test_recall'].std(),
            'cv_precision_mean': cv_scores['test_precision'].mean(),
            'cv_precision_std': cv_scores['test_precision'].std(),
            'test_metrics': sumarize_classification(y_test, y_pred, y_scores),
            **sumarize_classification(y_test, y_pred, y_scores)
        }
        results.append(result)
        # Convertir resultados a DataFrame
        
    df_results = pd.DataFrame(results)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print("\n📊 Resultados de modelos:\n",df_results)

    # Seleccionar mejor modelo según PR-AUC (si existe), si no por F1-score
    if 'pr_auc' in df_results.columns:
        best_model = df_results.loc[df_results['pr_auc'].idxmax()]
        criterio = "PR-AUC"
    else:
        best_model = df_results.loc[df_results['f1-score'].idxmax()]
        criterio = "F1-score"

    print(f"\n Mejor modelo seleccionado según {criterio}:")
    print(best_model)

    return df_results, best_model

# Preprocesar variables categoricas y numéricas Y entrenar modelos

# obtener datos preprocesados
X_train, X_test, y_train, y_test, preprocessor = ft_engineering()

# Aplicación de SMOTE para desbalanceo de clases



# --- Paso 2: aplicar SMOTE solo en el conjunto de entrenamiento ---
print("Distribución original:", y_train.value_counts(normalize=True))
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# entrenar y comparar modelos
df_results,best_model = train_and_select_model(X_train_res, y_train_res, X_test, y_test)

# Defino función para graficar Matrices de confusión 
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrices(best_models, X_test, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, (name, model) in zip(axes, best_models):
        y_pred = model.predict(X_test)
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, ax=ax, cmap="Blues", colorbar=False
        )
        ax.set_title(f"Matriz de Confusión - {name}")
    
    plt.suptitle("Comparación de los 2 mejores modelos", fontsize=16)
    plt.tight_layout()
    plt.show()
# Graficar matrices de confusión de los mejores modelos 
if 'pr_auc' in df_results.columns:
    top2 = df_results.nlargest(2, 'pr_auc')
else:
    top2 = df_results.nlargest(2, 'f1-score')

# Extraer los modelos entrenados
best_models = [(row['name'], row['model']) for _, row in top2.iterrows()]

# Graficar matrices de confusión
plot_confusion_matrices(best_models, X_test, y_test)