#Importación de librerías necesarias
from pathlib import Path
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Definición de las rutas principales del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_MODELO = BASE_DIR / "models" / "modelo_final.joblib"
RUTA_DATOS = BASE_DIR / "data" / "processed" / "inferencia_df_transformado.csv"


@st.cache_resource(show_spinner=False)
def cargar_modelo():
    """
    Carga el modelo entrenado desde la carpeta de modelos.
    Intenta usar joblib y, si falla, utiliza pickle como alternativa.
    """
    if not RUTA_MODELO.exists():
        raise FileNotFoundError(f"No se ha encontrado el archivo del modelo en: {RUTA_MODELO}")

    try:
        return joblib.load(RUTA_MODELO)
    except Exception:
        with open(RUTA_MODELO, "rb") as archivo:
            return pickle.load(archivo)

@st.cache_data(show_spinner=False)
def cargar_datos():
    """
    Carga el dataset de inferencia preprocesado (con variables de calendario,
    lags, medias móviles y codificación one-hot del catálogo) y convierte
    la columna de fecha a tipo datetime.
    """
    if not RUTA_DATOS.exists():
        raise FileNotFoundError(f"No se ha encontrado el dataset en: {RUTA_DATOS}")

    df = pd.read_csv(RUTA_DATOS)
    
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"])

    # Ordenamos por producto y fecha para que las filas sigan un orden temporal correcto
    columnas_orden = [col for col in ["nombre", "fecha"] if col in df.columns]
    if columnas_orden:
        df = df.sort_values(columnas_orden).reset_index(drop=True)

    return df


def cargar_recursos():
    """
    Función para cargar el modelo y los datos de forma simultánea en la aplicación.
    """
    modelo = cargar_modelo()
    df = cargar_datos()
    return modelo, df


def _ajustar_competencia(df_producto, escenario_competencia):
    """
    Ajusta los precios de las tiendas de la competencia según el escenario seleccionado
    y recalcula el precio medio de la competencia.
    """
    mapa_escenarios = {
        "Actual (0%)": 1.00,
        "Competencia -5%": 0.95,
        "Competencia +5%": 1.05,
    }
    factor = mapa_escenarios.get(escenario_competencia, 1.00)
    columnas_competidores = ["Amazon", "Decathlon", "Deporvillage"]

    # Nos aseguramos de trabajar sobre una copia explícita para evitar SettingWithCopyWarning de Pandas
    df_producto = df_producto.copy()

    if all(col in df_producto.columns for col in columnas_competidores):
        for col in columnas_competidores:
            df_producto[col] = pd.to_numeric(df_producto[col], errors="coerce") * factor
        df_producto["precio_competencia"] = df_producto[columnas_competidores].mean(axis=1)
    elif "precio_competencia" in df_producto.columns:
        # Forzamos la conversión numérica antes de aplicar el factor para asegurar la matemática
        precio_num = pd.to_numeric(df_producto["precio_competencia"], errors="coerce")
        df_producto["precio_competencia"] = precio_num * factor
    else:
        raise ValueError("No se encontraron las columnas de la competencia en el dataset.")

    return df_producto


def simular_escenario(df, producto, descuento_usuario, escenario_competencia, modelo=None):
    """
    Simula las ventas diarias de un producto para el mes proyectado
    aplicando estrategias de precio y escenarios de la competencia.
    """
    if df is None or df.empty:
        raise ValueError("El DataFrame de entrada está vacío.")

    if modelo is None:
        modelo = cargar_modelo()

    # Filtramos las filas asociadas al producto seleccionado
    df_producto = df[df["nombre"] == producto].copy()
    if df_producto.empty:
        raise ValueError(f"No se encontraron registros para el producto: {producto}")

    if "dia_del_mes" not in df_producto.columns:
        raise ValueError("Falta la columna 'dia_del_mes' en el dataset.")

    # Aseguramos la secuencia temporal diaria
    df_producto = df_producto.sort_values("dia_del_mes").reset_index(drop=True)

    # Validamos las columnas esenciales para los cálculos autoregresivos y de precios
    columnas_lag = [f"unidades_vendidas_lag{i}" for i in range(1, 8)]
    columnas_requeridas = columnas_lag + [
        "unidades_vendidas_ma7", "precio_base", "precio_venta", 
        "descuento_pct", "precio_competencia", "ratio_precio"
    ]
    for col in columnas_requeridas:
        if col not in df_producto.columns:
            raise ValueError(f"Falta la columna requerida en el dataset: {col}")

    # Reutilizamos la función interna para ajustar los precios de los competidores (modifica precio_competencia)
    df_producto = _ajustar_competencia(df_producto, escenario_competencia)

    # Convertimos los tipos de datos de las variables de precio para evitar conflictos
    columnas_numericas = ["precio_base", "precio_venta", "descuento_pct", "ratio_precio"]
    for col in columnas_numericas:
        df_producto[col] = pd.to_numeric(df_producto[col], errors="coerce")

    # Aplicamos el ajuste de descuento del usuario y recalculamos el precio de venta resultante
    df_producto["descuento_pct"] = df_producto["descuento_pct"] + (float(descuento_usuario) / 100.0)
    df_producto["precio_venta"] = df_producto["precio_base"] * (1 - df_producto["descuento_pct"])
    
    # Recalculamos el ratio de precios siempre para que el modelo detecte cualquier cambio (propio o ajeno)
    df_producto["ratio_precio"] = (
        df_producto["precio_venta"] / df_producto["precio_competencia"].replace({0: np.nan})
    )
    df_producto["ratio_precio"] = df_producto["ratio_precio"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Inicializamos el historial dinámico para la predicción autoregresiva (lags y media móvil)
    historial_lags = pd.to_numeric(df_producto.loc[0, columnas_lag], errors="coerce").fillna(0).astype(float).tolist()
    historial_ma7 = historial_lags.copy()

    columnas_modelo = list(modelo.feature_names_in_)
    registros = []

    # Bucle de predicción recursiva día a día
    for idx in range(len(df_producto)):
        fila = df_producto.loc[idx].copy()

        # Inyectamos los lags calculados dinámicamente en las iteraciones previas
        for i, col in enumerate(columnas_lag):
            fila[col] = historial_lags[i]
        
        fila["unidades_vendidas_ma7"] = float(np.mean(historial_ma7))

        # Creamos la estructura exacta que el modelo espera recibir
        fila_modelo = pd.DataFrame([fila]).reindex(columns=columnas_modelo, fill_value=0)
        
        # Ejecutamos la predicción y evitamos valores negativos
        prediccion = max(float(modelo.predict(fila_modelo)[0]), 0)
        ingresos = prediccion * float(fila["precio_venta"])

        registros.append({
            "dia_del_mes": int(fila["dia_del_mes"]),
            "dia_semana": fila.get("dia_semana", ""),
            "precio_venta": float(fila["precio_venta"]),
            "precio_competencia": float(fila["precio_competencia"]),
            "descuento_pct": float(fila["descuento_pct"]),
            "ratio_precio": float(fila["ratio_precio"]),
            "unidades_predichas": prediccion,
            "ingresos_proyectados": ingresos,
            "es_Black_Friday": bool(fila.get("es_Black_Friday", False)),
        })

        # Actualizamos las ventanas temporales con el nuevo valor predicho
        historial_lags = [prediccion] + historial_lags[:6]
        historial_ma7 = [prediccion] + historial_ma7[:6]

    # Consolidamos los resultados de la simulación
    detalle = pd.DataFrame(registros)

    resumen = {
        "producto": producto,
        "escenario_competencia": escenario_competencia,
        "descuento_usuario": float(descuento_usuario),
        "unidades_totales": float(detalle["unidades_predichas"].sum()),
        "ingresos_totales": float(detalle["ingresos_proyectados"].sum()),
        "precio_promedio": float(detalle["precio_venta"].mean()),
        "descuento_promedio": float(detalle["descuento_pct"].mean() * 100),
    }

    return {
        "detalle": detalle,
        "resumen": resumen,
    }