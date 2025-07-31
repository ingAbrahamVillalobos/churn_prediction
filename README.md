# Proyecto: Predicción de la Tasa de Cancelación de Clientes (Churn) en Telecomunicaciones

## Introducción

Este proyecto se enfoca en el desarrollo de un modelo de analítica predictiva para Interconnect, una empresa de telecomunicaciones. El objetivo principal es anticipar la cancelación de contratos por parte de los clientes (`churn`) para permitir a la compañía actuar proactivamente mediante ofertas personalizadas y estrategias de retención.

Se construyó un modelo de clasificación binaria que identifica clientes con alta probabilidad de abandono, basándose en su comportamiento histórico. La métrica principal para la evaluación del modelo fue el AUC-ROC, complementada con la exactitud (accuracy).

## Problema de Negocio

Interconnect busca reducir la tasa de `churn` de sus clientes. Anticipar quiénes son los clientes con mayor riesgo de cancelar su servicio permitirá a la empresa implementar estrategias de retención proactivas y optimizar sus recursos.

## Metodología

El proyecto incluyó las siguientes etapas:

### 1. Carga y Exploración de Datos (EDA)

Se cargaron cuatro datasets principales: `contract.csv`, `personal.csv`, `internet.csv`, y `phone.csv`. Se realizó un análisis exploratorio detallado de cada uno para comprender su estructura, tipos de datos, y la presencia de valores nulos o duplicados.

### 2. Limpieza de Datos

* **Estandarización de nombres:** Las columnas fueron renombradas a formato `snake_case`.
* **Corrección de tipos de datos:** Columnas de fechas (`begin_date`, `end_date`) se convirtieron a tipo `datetime`. La columna `total_charges` se convirtió a numérico (`float`), imputando los valores nulos (clientes nuevos sin facturación) con 0.
* **Manejo de valores nulos post-unión:** Tras unir los datasets, se imputaron valores nulos en columnas de servicios (`online_security`, `multiple_lines`, etc.) con "No" para indicar la ausencia del servicio, y "Sin Servicio" para `internet_service`.

### 3. Unión de Datasets

Los cuatro datasets (`contract`, `personal`, `internet`, `phone`) se unieron en un único `df_master` utilizando `customer_id` como clave y un `left join` con `df_contract` como base, asegurando la inclusión de todos los clientes.

### 4. Ingeniería de Características

* **Creación de la variable objetivo `churn`:** Se definió una variable binaria `churn` (1 si el cliente canceló, 0 si está activo) basada en la columna `end_date`.
* **Cálculo de `contract_duration`:** Se agregó una característica crucial, `contract_duration` (duración del contrato en meses), calculada a partir de `begin_date` y `end_date`. Para clientes activos, se asumió la duración hasta la fecha más reciente de churn o una fecha de referencia posterior.

### 5. Preprocesamiento para Modelado

* **Balanceo de clases:** Se analizó el desequilibrio de clases en la variable `churn` (proporción de 0s y 1s). Se aplicó un sobremuestreo (upsampling) de la clase minoritaria (`churn = 1`) para igualar su tamaño con la clase mayoritaria (`churn = 0`) en el conjunto de entrenamiento.
* **One-Hot Encoding:** Las variables categóricas se transformaron a variables numéricas utilizando One-Hot Encoding.
* **Escalado de variables numéricas:** Las columnas numéricas (`monthly_charges`, `total_charges`, `contract_duration`) fueron escaladas utilizando `StandardScaler`.
* **División de datos:** Los datos se dividieron en conjuntos de entrenamiento y prueba (80%-20% respectivamente), manteniendo la estratificación de la variable `churn`.

### 6. Modelado y Evaluación

Se entrenaron y evaluaron varios modelos de clasificación, comparando su rendimiento con la métrica AUC-ROC y otras métricas como precisión, recall y f1-score:

* **Regresión Logística:** (Línea base) ROC AUC = 0.828
* **Random Forest Classifier:** ROC AUC = 0.8878
* **XGBoost Classifier:** ROC AUC = 0.8874
* **CatBoost Classifier:** **ROC AUC = 0.8915** (¡Mejor resultado!)
* **LightGBM Classifier:** ROC AUC = 0.8844

## Conclusión y Recomendación

Tras la evaluación de todos los modelos entrenados, **CatBoost Classifier** demostró el mejor desempeño general, alcanzando un **ROC AUC de 0.8915**. Este resultado supera el umbral requerido y muestra una excelente capacidad para discriminar entre clientes que cancelan y los que no.

El modelo CatBoost también logró un buen balance en las métricas de clasificación para la clase positiva (clientes que cancelan), con una precisión de 0.63, un recall de 0.78 y un f1-score de 0.70. Esto indica que el modelo es eficaz tanto en la detección de clientes en riesgo de `churn` como en la identificación de aquellos que permanecerán, siendo particularmente útil para la estrategia de retención de Interconnect.

Por lo tanto, se recomienda el modelo **CatBoost Classifier** como la solución final para predecir la cancelación de clientes, debido a su robusto rendimiento predictivo.

## Tecnologías Utilizadas

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `sklearn` (train_test_split, StandardScaler, LogisticRegression, RandomForestClassifier, classification_report, confusion_matrix, roc_auc_score, resample)
* `xgboost` (XGBClassifier)
* `catboost` (CatBoostClassifier)
* `lightgbm` (LGBMClassifier)
