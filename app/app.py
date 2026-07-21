# Importación de librerías necesarias
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st
from utils import cargar_recursos, simular_escenario

# Configuración de la interfaz de la aplicación
st.set_page_config(page_title="Simulador de ventas - Noviembre 2025", page_icon="📈", layout="wide")

# Estilos CSS optimizados para el diseño de tarjetas y fondo
st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #f6f8ff 0%, #eef2ff 100%); }
        .bloque-card { 
            background: white; 
            padding: 1.2rem; 
            border-radius: 14px; 
            border: 1px solid rgba(102,126,234,0.15); 
            box-shadow: 2px 2px 8px rgba(0,0,0,0.03);
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Funciones rápidas de formato de texto
formatear_euro = lambda v: f"€{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
formatear_num = lambda v: f"{v:,.0f}".replace(",", ".")

try:
    modelo, df = cargar_recursos()
except Exception as error:
    st.error(f"Error al cargar recursos: {error}")
    st.stop()

# Controles de la barra lateral
st.sidebar.title("Controles de Simulación")
producto_seleccionado = st.sidebar.selectbox("Seleccione un Producto", sorted(df["nombre"].unique().tolist()))

# 1. Slider con formato '%' e icono de ayuda explicativo
descuento_usuario = st.sidebar.slider(
    "Ajuste de descuento", 
    min_value=-50, 
    max_value=50, 
    value=0, 
    step=5,
    format="%d%%",
    help="Aplica un porcentaje de descuento adicional (+) o una subida de precio (-) sobre el precio base del producto para todo el mes simulado."
)

# 2. Radio buttons e icono de ayuda explicativo
escenario_competencia = st.sidebar.radio(
    "Escenario de competencia", 
    ["Actual (0%)", "Competencia -5%", "Competencia +5%"],
    help="Simula cómo responderá el mercado si tus competidores principales (Amazon, Decathlon, Deporvillage) modifican sus precios al alza o a la baja."
)

if st.sidebar.button("Simular Ventas", use_container_width=True, type="primary"):
    with st.spinner("Simulando..."):
        # Guardamos el producto actual simulado en la sesión para congelar los títulos principales
        st.session_state["producto_activo"] = producto_seleccionado
        st.session_state["resultados"] = {
            esc: simular_escenario(df, producto_seleccionado, descuento_usuario, esc, modelo)
            for esc in ["Actual (0%)", "Competencia -5%", "Competencia +5%"]
        }

# Pantalla de bienvenida (antes de la simulación)
if "resultados" not in st.session_state:
    st.markdown('# <span style="font-family: inherit;">🤖</span> Simulador de ventas — Noviembre 2025', unsafe_allow_html=True)
    st.info("Seleccione un producto, el descuento adicional deseado y el escenario de la competencia en la barra lateral, luego pulse **Simular Ventas**.")
    st.stop()

# Recuperamos de forma segura el nombre del producto que realmente se simuló
producto_activo = st.session_state["producto_activo"]

# Extracción de los resultados activos de la sesión
res_escenarios = st.session_state["resultados"]
detalle = res_escenarios[escenario_competencia]["detalle"]
resumen = res_escenarios[escenario_competencia]["resumen"]

# Cabecera principal
st.markdown(f"# 📊 Simulación de ventas — Noviembre 2025")
st.markdown(f"### **{producto_activo}**")
st.caption(f"Escenario de mercado seleccionado: **{escenario_competencia}** | Ajuste Descuento aplicado: **{descuento_usuario:+d}%**")
st.divider()

# Corregimos visualmente el descuento promedio si viene negativo o ya escalado
desc_promedio = resumen["descuento_promedio"]
if desc_promedio > 100 or desc_promedio < -100:
    desc_promedio = desc_promedio / 100.0
desc_promedio = abs(desc_promedio)

# KPIs principales integrados con tarjetas
metrics = [
    ("📦 Unidades totales", formatear_num(resumen["unidades_totales"]), "#ff4b4b"),
    ("💰 Ingresos totales", formatear_euro(resumen["ingresos_totales"]), "#2e7d32"), 
    ("🏷️ Precio promedio", formatear_euro(resumen["precio_promedio"]), "#0288d1"),
    ("📉 Descuento promedio", f"{desc_promedio:.2f}%", "#fbc02d")
]

for col, (label, val, color) in zip(st.columns(4), metrics):
    with col:
        st.markdown(
            f"""
            <div class="bloque-card" style="border-left: 5px solid {color};">
                <p style="margin: 0; font-size: 13px; color: #6c757d; font-weight: bold; text-transform: uppercase;">{label}</p>
                <p style="margin: 5px 0 0 0; font-size: 26px; font-weight: bold; color: #1c1c1c;">{val}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

st.write("") # Espaciador sutil

# Gráfico evolutivo optimizado
st.subheader("📈 Predicción diaria de la demanda")
fig, ax = plt.subplots(figsize=(12, 3.5))
sns.lineplot(data=detalle, x="dia_del_mes", y="unidades_predichas", ax=ax, color="#667eea", linewidth=2, marker="o")
ax.axvline(28, color="#d62728", linestyle="--", alpha=0.7)

# Flecha que señala el valor real de la predicción en Black Friday (día 28, fecha fija para noviembre de 2025 — cambiaría en otro año)
if 28 in detalle["dia_del_mes"].values:
    val_bf = float(detalle.loc[detalle["dia_del_mes"] == 28, "unidades_predichas"].values[0])
    ax.annotate(
        "Black Friday", 
        xy=(28, val_bf), 
        xytext=(24, val_bf * 0.9), 
        arrowprops=dict(arrowstyle="->", color="#d62728"), 
        color="#d62728", 
        fontweight="bold"
    )

ax.set_title("Evolución diaria de las ventas proyectadas")
sns.despine(ax=ax)
st.pyplot(fig, clear_figure=True)
st.divider()

# Comparativa de los 3 escenarios automatizada y contenida estéticamente dentro de tarjetas
st.subheader("⚖️ Comparativa de impacto por escenarios")
for col, esc in zip(st.columns(3), ["Actual (0%)", "Competencia -5%", "Competencia +5%"]):
    r = res_escenarios[esc]["resumen"]
    with col:
        st.markdown(
            f"""
            <div class="bloque-card">
                <h4 style="margin:0 0 15px 0; color:#764ba2; font-weight: bold; border-bottom: 1px solid #f0f0f0; padding-bottom: 8px;">{esc}</h4>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #6c757d; font-size: 14px;">📦 Unidades totales:</span>
                    <span style="font-weight: bold; color: #1c1c1c;">{formatear_num(r["unidades_totales"])}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #6c757d; font-size: 14px;">💰 Ingresos totales:</span>
                    <span style="font-weight: bold; color: #2e7d32;">{formatear_euro(r["ingresos_totales"])}</span>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )

st.divider()

# Tabla de los registros desglosada por días 
st.subheader("🧾 Desglose analítico diario")
st.markdown("Registros diarios procesados recursivamente por las ventanas dinámicas del modelo predictivo:")

tabla = detalle.copy()
tabla["evento"] = np.where(tabla["dia_del_mes"] == 28, "🔥 Black Friday", "📆 Ordinario")

# Diccionario para traducir los días de la semana al español
dias_es = {
    "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
    "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo"
}
tabla["dia_semana"] = tabla["dia_semana"].map(dias_es).fillna(tabla["dia_semana"])

columnas_ver = [
    "evento", "dia_del_mes", "dia_semana", "precio_venta", 
    "precio_competencia", "ratio_precio", "descuento_pct", 
    "unidades_predichas", "ingresos_proyectados"
]

# Damos formato a cada columna (euros, porcentajes, decimales) para facilitar la lectura
st.dataframe(
    tabla[columnas_ver],
    use_container_width=True,
    hide_index=True,
    column_config={
        "evento": st.column_config.TextColumn("Tipo de Día", help="Identifica días clave como Black Friday"),
        "dia_del_mes": st.column_config.NumberColumn("Día", format="%d"),
        "dia_semana": st.column_config.TextColumn("Día Semana"),
        "precio_venta": st.column_config.NumberColumn("Precio Venta", format="€%.2f"),
        "precio_competencia": st.column_config.NumberColumn("Precio Comp.", format="€%.2f"),
        "ratio_precio": st.column_config.NumberColumn("Ratio Precio", format="%.2f"),
        "descuento_pct": st.column_config.NumberColumn("Descuento", format="%.2f%%"),
        "unidades_predichas": st.column_config.NumberColumn("Uds. Predichas", format="%d"),
        "ingresos_proyectados": st.column_config.NumberColumn("Ingresos Est.", format="€%.2f"),
    }
)