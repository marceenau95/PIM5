import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import ks_2samp, chi2_contingency

# Importar funciones propias
from ft_engineering import build_pipeline, NUMERIC_FEATURES, CATEGORICAL_FEATURES   # usamos solo procesamiento

# -------------------------------
# Funciones auxiliares
# -------------------------------

def calcular_drift_num(var_hist, var_new):
    stat, p_value = ks_2samp(var_hist, var_new)
    return p_value

def calcular_drift_cat(var_hist, var_new):

    categorias = sorted(
        list(set(var_hist.astype(str)).union(set(var_new.astype(str))))
    )

    hist = var_hist.astype(str).value_counts().reindex(categorias, fill_value=0)
    new = var_new.astype(str).value_counts().reindex(categorias, fill_value=0)

    tabla = np.vstack([hist.values, new.values])

    _, p_value, _, _ = chi2_contingency(tabla)

    return p_value


def indicador_alerta(p_value, umbral=0.05):
    if p_value < umbral/2:
        return "🔴 Crítico"
    elif p_value < umbral:
        return "🟠 Riesgo"
    else:
        return "🟢 Estable"

def format_pvalue(p_value):
    # Si el valor es extremadamente pequeño, mostrar en notación científica
    if p_value < 1e-4:
        return f"{p_value:.4e}"   
    else:
        return round(p_value, 4)  

# -------------------------------
# Interfaz Streamlit
# -------------------------------

st.title("📈 Monitoreo de Data Drift")

# Dataset histórico usando cargarDatos()
df_hist = pd.read_excel('Base_de_datos.xlsx')

# Intentar leer dataset nuevo desde ruta relativa
ruta_relativa = os.path.join(os.path.dirname(__file__), "..", "Base_de_datos_con_Data_Drift_Simulado.xlsx")
df_new = None

if os.path.exists(ruta_relativa):
    st.success(f"✅ Dataset nuevo encontrado en {ruta_relativa}")
    df_new = pd.read_excel(ruta_relativa)
else:
    st.warning("⚠️ No se encontró el dataset en la carpeta anterior. Por favor súbelo manualmente.")
    new_file = st.sidebar.file_uploader("Subir dataset nuevo (CSV)", type=["csv"])
    if new_file is not None:
        df_new = pd.read_csv(new_file)
        st.success("✅ Dataset nuevo cargado manualmente")

# -------------------------------
# Comparación y drift
# -------------------------------

if df_new is not None:
    st.write("### Datos cargados")
    st.write("Histórico:", df_hist.shape, " | Nuevo:", df_new.shape)

    # Construir pipeline y aplicar preprocesamiento
    (
        _,
        X_hist,
        y_hist,
        cleaning_pipeline,
        _
    ) = build_pipeline(df_hist)
    cleaning_pipeline.fit(X_hist)  # fit solo una vez sobre histórico

    X_hist_clean = cleaning_pipeline.transform(X_hist)
    
    columnas = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X_hist_clean = pd.DataFrame(X_hist_clean, columns=columnas)
    
    (
        _,
        X_new,
        _,
        _,
        _
    ) = build_pipeline(df_new)

    X_new_clean= cleaning_pipeline.transform(X_new)

    X_new_clean = pd.DataFrame(
        X_new_clean,
        columns=columnas
    )
     # Recuperar tipos

    for c in NUMERIC_FEATURES:
        X_hist_clean[c] = pd.to_numeric(X_hist_clean[c])
        X_new_clean[c] = pd.to_numeric(X_new_clean[c])

    for c in CATEGORICAL_FEATURES:
        X_hist_clean[c] = X_hist_clean[c].astype(object)
        X_new_clean[c] = X_new_clean[c].astype(object)

    st.success("Pipeline de limpieza aplicado correctamente.")


    # Selección de variables originales para comparar distribuciones
    variables = st.multiselect("Selecciona variables a comparar", columnas, default=columnas)

    resultados = []
    for var in variables:
        st.subheader(var)
        if var in NUMERIC_FEATURES:

            hist = X_hist_clean[var].dropna()

            new = X_new_clean[var].dropna()

            p = calcular_drift_num(hist, new)

            fig, ax = plt.subplots()

            ax.hist(hist, bins=30, alpha=0.5, label="Histórico")

            ax.hist(new, bins=30, alpha=0.5, label="Nuevo")

            ax.legend()

            st.pyplot(fig)

        else:

            hist = X_hist_clean[var].fillna("Missing")

            new = X_new_clean[var].fillna("Missing")

            p = calcular_drift_cat(hist, new)

            fig, ax = plt.subplots()

            pd.concat(
                [
                    hist.value_counts(normalize=True).rename("Histórico"),
                    new.value_counts(normalize=True).rename("Nuevo"),
                ],
                axis=1,
            ).fillna(0).plot.bar(ax=ax)

            st.pyplot(fig)

        resultados.append(
            {
                "Variable": var,
                "p-value": format_pvalue(p),
                "Estado": indicador_alerta(p, umbral=0.05),
            }
        )
    st.subheader("Resumen")

    df_res = pd.DataFrame(resultados)

    st.dataframe(df_res, use_container_width=True)

    if (df_res["Estado"] == "🔴 Crítico").any():

        st.error(
            "Se detectó Data Drift crítico."
        )

    elif (df_res["Estado"] == "🟠 Riesgo").any():

        st.warning(
            "Se detectó Data Drift moderado."
        )

    else:

        st.success(
            "No se detectó Data Drift significativo."
        )

else:

    st.info("Carga un dataset para comenzar.")
    