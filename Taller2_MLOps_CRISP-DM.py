# -*- coding: utf-8 -*-
"""Taller2_MLOps.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HEilImXCtGU3tNlceqWNYp3c28-ZgkSv

#Taller 2 - MLOps
##Carmen Carvajal Gutiérrez

#Entendimiento del Negocio

##Objetivo del Negocio:
El objetivo principal es predecir la probabilidad de quiebra de una empresa con base en indicadores financieros.
*   Gestión del riesgo crediticio: Ayudar a bancos e instituciones financieras a tomar decisiones informadas sobre la aprobación de créditos o préstamos.

##Objetivo Analítico:
Construir un modelo de Machine Learning para clasificar las empresas en dos categorías, No quebrada y Quebrada, acorde a sus indicadores financieros. El modelo debe ser capaz de predecir la probabilidad de quiebra con un rendimiento robusto.
"""

# Importar las bibliotecas necesarias para el análisis y modelado
import pandas as pd  # Para manipulación y análisis de datos
import numpy as np   # Para operaciones numéricas
import matplotlib.pyplot as plt  # Para visualización
import seaborn as sns  # Para visualización estadística

# Importar herramientas de scikit-learn para preparación y evaluación de modelos
from sklearn.model_selection import train_test_split  # Para dividir los datos en train y test
from sklearn.model_selection import GridSearchCV, cross_val_score  # Para validación cruzada
from sklearn.preprocessing import StandardScaler  # Para normalizar las variables
from sklearn.metrics import (
    classification_report,  # Reporte detallado de métricas
    confusion_matrix,      # Matriz de confusión
    roc_auc_score,        # Área bajo la curva ROC
    roc_curve,            # Para graficar curva ROC
    f1_score,             # Métrica F1
    precision_score,      # Precisión
    recall_score          # Recall/Sensibilidad
)

# Importar los modelos de clasificación que se van a evaluar
from sklearn.linear_model import LogisticRegression  # Regresión logística básica
from sklearn.ensemble import RandomForestClassifier  # Bosques aleatorios
from sklearn.ensemble import GradientBoostingClassifier  # Gradient Boosting
from sklearn.svm import SVC  # Support Vector Machine
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors
from sklearn.neural_network import MLPClassifier  # Red neuronal multicapa
from sklearn.tree import DecisionTreeClassifier  # Árbol de decisión

# Importar herramientas para manejar el desbalance de clases
from imblearn.over_sampling import SMOTE  # Synthetic Minority Over-sampling Technique

# Desactivar warnings para una salida más limpia
import warnings
warnings.filterwarnings('ignore')

"""#Entendimiento de los datos"""

# Cargar los datos
file_path = '/content/data.csv'
data = pd.read_csv(file_path)

#Primeras 5 filas del dataset para entender su estructura
data.head()

# Información básica sobre el tamaño del dataset
data.shape

#Validación de valores nulos en el dataset
data.info()

# Estadísticas descriptivas básicas del dataset
data.describe()

# Análisis de la distribución de la variable objetivo
target_distribution = data['Bankrupt?'].value_counts()

# Porcentaje de cada clase
target_percentage = target_distribution / target_distribution.sum() * 100

# Resumen de la distribución
target_summary = pd.DataFrame({
    'Count': target_distribution,
    'Percentage': target_percentage
})

# Visualización de la distribución de la variable objetivo
plt.figure(figsize=(8, 6))
plt.bar(target_summary.index.astype(str), target_summary['Count'], alpha=0.7)
plt.title('Distribución de la variable objetivo: Bankrupt?', fontsize=14)
plt.xlabel('Clases (0 = No Quiebra, 1 = Quiebra)', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

target_summary

"""#Preparación de los datos"""

# DETECCIÓN Y MANEJO DE OUTLIERS (VALORES ATÍPICOS)

# Identificar las columnas numéricas (features)
numerical_cols = data.columns.drop('Bankrupt?')

# Visualizar la distribución y outliers de cada variable con boxplots
plt.figure(figsize=(20, 15))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(10, 10, i)
    sns.boxplot(y=data[col])
    plt.title(col, fontsize=8)
plt.tight_layout()
plt.show()

# Tratamiento de outliers mediante winsorización (reemplazar valores extremos por el percentil 1 y 99)
for col in numerical_cols:
    lower_percentile = data[col].quantile(0.01)
    upper_percentile = data[col].quantile(0.99)
    data[col] = np.where(data[col] < lower_percentile, lower_percentile, data[col])
    data[col] = np.where(data[col] > upper_percentile, upper_percentile, data[col])

# BALANCEO DE CLASES CON SMOTE

# Separar features (X) y variable objetivo (y)
X = data.drop('Bankrupt?', axis=1)  # Features
y = data['Bankrupt?']  # Target variable

# Mostrar la distribución inicial de clases
print("Distribución de clases antes del balanceo:")
print(y.value_counts())

# Aplicar SMOTE para balancear las clases (genera ejemplos sintéticos de la clase minoritaria)
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

print("\nDistribución de clases después de aplicar SMOTE:")
print(pd.Series(y_balanced).value_counts())

# NORMALIZACIÓN DE VARIABLES

# Aplicar StandardScaler para normalizar las variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_balanced)

# DIVISIÓN EN TRAIN Y TEST

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_balanced,
    test_size=0.2,  # 20% para test
    random_state=42,  # Para reproducibilidad
    stratify=y_balanced  # Mantener proporción de clases
)

print(f"Muestras de entrenamiento: {X_train.shape[0]}")
print(f"Muestras de prueba: {X_test.shape[0]}")

# FUNCIÓN DE EVALUACIÓN

def evaluate_model(model, X_test, y_test):
    # Realizar predicciones
    y_pred = model.predict(X_test)
    # Obtener probabilidades predichas (o decision function para SVM)
    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

    # Calcular múltiples métricas
    return {
        'ROC AUC': roc_auc_score(y_test, y_proba),
        'F1 Score': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred)
    }

"""#Modelación"""

# DEFINICIÓN Y ENTRENAMIENTO DE MODELOS

# Definir modelos para comparar su rendimiento
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(probability=True, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Neural Network': MLPClassifier(random_state=42, max_iter=1000),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Entrenar y evaluar cada modelo
results = {}
for name, model in models.items():
    print(f"Entrenando {name}...")
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    results[name] = metrics
    print(f"Métricas de evaluación para {name}:")
    print(metrics, "\n")

"""#Evaluación"""

# COMPARACIÓN DE MODELOS

# Crear un DataFrame con los resultados
results_df = pd.DataFrame(results).T
results_df = results_df[['ROC AUC', 'F1 Score', 'Precision', 'Recall']]

print("Comparación de rendimiento entre modelos:")
display(results_df.sort_values(by='ROC AUC', ascending=False))

# Visualizar los resultados con un gráfico de barras
plt.figure(figsize=(10,6))
sns.barplot(x=results_df.index, y='Recall', data=results_df)
plt.title('Comparación de Recall entre modelos')
plt.ylabel('Recall Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""## Métrica de desempeño más importante
La métrica de desempeño más importante en este caso es el Recall. Esto se debe a que se esta trabajando con un conjunto de datos altamente desequilibrado, en el que el número de empresas No bancarrotas supera significativamente al de empresas en bancarrota. El Recall mide la proporción de casos positivos reales (empresas en bancarrota) que el modelo identifica correctamente. Por lo tanto, esta métrica es clave para garantizar que el modelo sea eficaz en la identificación de la clase minoritaria.

##Por qué no otras métricas?
- Accuracy: En conjuntos de datos desequilibrados, un modelo podría lograr una alta exactitud simplemente prediciendo siempre la clase mayoritaria (no bancarrotas). Esto sería engañoso ya que ignora la capacidad del modelo para detectar las bancarrotas.

- ROC AUC: Aunque ROC AUC es una métrica valiosa, puede presentar una visión demasiado optimista del rendimiento del modelo en configuraciones desequilibradas porque tiene en cuenta la tasa de verdaderos negativos, que es alta debido a la abundancia de casos no bancarrotas.

- F1 Score: Es la media armónica de precisión y recall, proporcionando una métrica única que equilibra ambos. Es útil pero aún podría no capturar los matices de los compromisos en conjuntos de datos altamente desequilibrados.

#Conclusión

Todos los modelos evaluados muestran un buen desempeño, con métricas que superan el 90% en términos de ROC AUC, F1 Score, Precisión y Recall. Sin embargo, dado el desequilibrio que presenta el conjunto de datos, donde las empresas en bancarrota representan la clase minoritaria, y considerando que la métrica de mayor relevancia para este caso es el Recall, el modelo K-Nearest Neighbors (KNN) se destaca como el más adecuado. Este modelo logra un Recall perfecto (1.000000), asegurando la identificación completa de los casos positivos reales (empresas en bancarrota), lo cual es importante en este contexto específico.
"""