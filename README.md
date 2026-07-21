# 📊 Pricing & Demand Forecasting E-Commerce Engine

> **Proyecto de Data Science end-to-end: del dato crudo a una aplicación interactiva de simulación**

---

## 📌 Visión general

Este repositorio contiene mi **primer proyecto completo de Data Science**, centrado en un problema habitual en e-commerce: **¿cómo afectan los descuentos propios y los movimientos de precio de la competencia a las ventas previstas?**

El objetivo no era solo entrenar un modelo en un notebook, sino construir un pipeline completo: limpieza y preparación de datos, ingeniería de variables, entrenamiento y validación de un modelo de forecasting, y una **aplicación interactiva en Streamlit** para simular distintos escenarios de precio y competencia.

---

## 🏗️ Pipeline del proyecto

1. **Carga y preparación de datos:** histórico de ventas, precios propios y precios de tres competidores (*Amazon, Decathlon, Deporvillage*).
2. **Feature engineering:**
   - **Calendario:** festivos en España (librería `holidays`), fin de semana, y cálculo dinámico de Black Friday / Cyber Monday (a partir de la regla real de calendario, no de fechas fijas).
   - **Inercia de demanda:** variables autorregresivas (*lags* de 1 a 7 días) y media móvil de 7 días (`ma7`), calculadas por producto.
   - **Precio y competencia:** porcentaje de descuento (`descuento_pct`), precio medio de la competencia (`precio_competencia`) y ratio de precio propio frente a competencia (`ratio_precio`).
   - **Catálogo:** one-hot encoding de `nombre`, `categoria` y `subcategoria`.
3. **Modelado y validación:** `HistGradientBoostingRegressor` de scikit-learn, validado con un split cronológico (no aleatorio, para evitar fuga de información temporal) y evaluado con MAE, RMSE, MAPE, WAPE y R², comparado contra un baseline simple (media histórica).
4. **Despliegue:** exportación del modelo (`.joblib`) y del dataset de inferencia procesado, consumidos por una app de Streamlit para simular escenarios en tiempo real.

---

## 🐛 Del notebook a producción: bugs reales encontrados y corregidos

Una parte importante de este proyecto fue detectar y corregir inconsistencias entre el notebook de entrenamiento y el de inferencia — un problema muy común en proyectos reales de ML (*training-serving skew*), y que documento aquí porque el proceso de depuración fue tan formativo como el modelado en sí:

1. **Columnas de competencia perdidas en inferencia:** el notebook de inferencia calculaba `precio_competencia` a partir de `Amazon`, `Decathlon` y `Deporvillage`, y a continuación eliminaba estas tres columnas originales. El modelo, sin embargo, había sido entrenado usándolas de forma individual — así que en producción siempre recibía estas tres variables a `0`, y el simulador no reaccionaba a los cambios de competencia. Solución: dejar de eliminar esas columnas en el pipeline de inferencia.
2. **Variables de precio ausentes en el entrenamiento:** `descuento_pct`, `precio_competencia` y `ratio_precio` se calculaban en el notebook de inferencia, pero nunca se habían calculado en el de entrenamiento — por lo que el modelo jamás llegó a aprender de ellas. Solución: añadir el mismo cálculo en el notebook de entrenamiento antes de entrenar el modelo final.

Ambos bugs se diagnosticaron comparando explícitamente las columnas que el modelo esperaba (`model.feature_names_in_`) contra las columnas realmente disponibles en cada punto del pipeline, en lugar de modificar código a ciegas.

---

## 📈 Variables más influyentes (Permutation Importance)

Con el modelo ya corregido, un análisis de *permutation importance* sobre el modelo final muestra que:

- **`descuento_pct`** es, con diferencia, la variable más influyente — el modelo sí ha aprendido a asociar el descuento propio con el volumen de ventas.
- La **identidad del producto** (`nombre_h=Nike Air Zoom Pegasus 40`, `nombre_h=Adidas Ultraboost 23`) y el **`precio_base`** forman el segundo grupo más relevante — el histórico propio de cada artículo pesa mucho en la predicción.
- El **histórico reciente** (`unidades_vendidas_lag1`) y la **competencia** (`Decathlon`, `ratio_precio`, `precio_venta`) completan el grupo de variables relevantes, aunque con menor peso que las anteriores.

---

## 🖥️ Aplicación interactiva (Streamlit)

La app permite simular distintos escenarios de negocio:
- **Ajuste de descuento** mediante slider, sobre el precio base del producto seleccionado.
- **Escenarios de competencia:** competencia actual, un 5% más barata o un 5% más cara.
- **Resultados:** unidades e ingresos proyectados para noviembre, con desglose diario y comparativa entre escenarios.

> 🔗 **[Próximamente: enlace a la app en Streamlit Cloud]**

---

## ⚠️ Limitaciones conocidas

Documentar las limitaciones de un modelo es tan importante como documentar lo que funciona. Estas son las que identifiqué en este proyecto:

1. **Comportamiento "en escalón" ante cambios de precio:** al ser un modelo basado en árboles de decisión sin restricciones de monotonía, la predicción no cambia de forma suave y proporcional al variar el descuento — se mantiene estable dentro de ciertos rangos y salta de golpe al cruzar un umbral de decisión interno del árbol. Por ejemplo, un incremento de descuento pequeño puede no cambiar la predicción, mientras que cruzar cierto punto provoca un salto notable. Esto es un comportamiento estructural esperable en este tipo de algoritmos, no un error de los datos o el código.
2. **Sensibilidad a la competencia no siempre monótona:** en algunos productos, un escenario de "competencia más barata" puede predecir ventas ligeramente superiores a un escenario de "competencia más cara" — lo contrario de lo esperable por lógica de negocio. Se observó en un producto concreto (una variación de 1 unidad sobre ~320-327, un margen pequeño); en otro producto probado, la dirección sí fue la correcta. Es probable que se deba a la falta de restricciones de monotonía combinada con relaciones parcialmente ruidosas en los datos históricos disponibles para ese producto en concreto, aunque no lo he verificado de forma exhaustiva para todo el catálogo.
3. **Dataset sintético:** los datos de ventas y competencia proceden de un dataset generado para fines formativos, no de un negocio real — los patrones pueden ser más "limpios" o menos complejos que en un caso real.

### Posibles mejoras futuras
- Aplicar restricciones de monotonía (`monotonic_cst` en scikit-learn) para forzar que el modelo respete relaciones de sentido económico (a más descuento, más ventas; a competencia más cara, más ventas propias), evitando el comportamiento en escalón.
- Evaluar modelos alternativos (lineales, o árboles con mayor profundidad) y comparar su capacidad de generalizar tanto la identidad de producto como la sensibilidad al precio.
- Ampliar la validación a todos los productos del catálogo, no solo a los probados manualmente en el simulador.
- **Optimizar el algoritmo de forma sistemática:** en este proyecto no se ha realizado una búsqueda de hiperparámetros (*grid search*, *random search*, optimización bayesiana, etc.) ni una comparación rigurosa entre distintas familias de modelos — los hiperparámetros actuales son una configuración razonable, pero no una configuración optimizada. Esta seria una mejora clara para siguientes iteraciones.

> **Nota:** este es mi primer proyecto de Data Science end-to-end, y el objetivo principal ha sido recorrer y entender el proceso completo (desde el dato crudo hasta una aplicación funcional), no conseguir un algoritmo perfecto ni exprimir al máximo su rendimiento. Soy consciente de que el modelo tiene un margen de mejora considerable, y las limitaciones y mejoras futuras recogidas en esta sección son precisamente el resultado de haber auditado el proyecto con sentido crítico.

---

## 🛠️ Estructura del repositorio

```text
forecasting-demand-app/
│
├── app/                  # Código de la aplicación Streamlit
│   ├── app.py            # Interfaz y visualización
│   └── utils.py          # Carga de recursos y lógica de simulación
├── data/                 # Datasets (raw y processed)
├── models/               # Modelo entrenado (.joblib)
├── notebooks/            # Notebooks de entrenamiento e inferencia
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 👤 Autor

**Pol Rivero Zaragoza**
*Estudiante de Ingeniería Industrial (UPC - ETSEIB)*

Interesado en la aplicación de **Data Science e Inteligencia Artificial** para la optimización de procesos y la toma de decisiones estratégicas, transformando datos complejos en soluciones tecnológicas de alto impacto. Mi objetivo es aplicar la analítica avanzada y la IA para resolver retos complejos de ingeniería y negocio.

📩 [LinkedIn](https://www.linkedin.com/in/tu-perfil) | [GitHub](https://github.com/tu-usuario)
