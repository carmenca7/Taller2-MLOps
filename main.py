from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Cargamos el pipeline del modelo entrenado que contiene todas las transformaciones
# y el modelo KNN para predicción de bancarrota
model_pipeline = joblib.load('bankruptcy_knn_pipeline.joblib')

# Obtenemos los nombres de las características/variables que el modelo necesita
# Estos nombres se extraen del transformador 'winsorizer' que fue el primer paso del pipeline
feature_names = model_pipeline.named_steps['winsorizer'].lower_bounds_.index.tolist()

# Importamos la función necesaria de pydantic para crear modelos dinámicamente
from pydantic import create_model

# Creamos un diccionario que define los campos del modelo Pydantic
# Cada campo será de tipo float y será requerido (indicado por los ...)
fields = {name: (float, ...) for name in feature_names}

# Creamos dinámicamente el modelo Pydantic que validará los datos de entrada
# Este modelo asegurará que todas las características necesarias estén presentes
# y sean del tipo correcto (float)
BankruptcyFeatures = create_model('BankruptcyFeatures', **fields)

# Inicializamos la aplicación FastAPI
app = FastAPI()

# Definimos el endpoint POST que recibirá las características y devolverá la predicción
@app.post('/predict')
def predict(features: BankruptcyFeatures):
    # Convertimos los datos de entrada en un DataFrame de pandas
    # Esto es necesario porque el pipeline espera un DataFrame como entrada
    input_df = pd.DataFrame([features.dict()])
    
    # Aplicamos las transformaciones del pipeline en orden:
    # 1. Winsorization: para manejar valores extremos
    # 2. Escalado: para normalizar las características
    input_preprocessed = model_pipeline.named_steps['scaler'].transform(
        model_pipeline.named_steps['winsorizer'].transform(input_df)
    )
    
    # Realizamos la predicción usando el modelo KNN
    # prediction: 0 = no bancarrota, 1 = bancarrota
    prediction = model_pipeline.named_steps['knn'].predict(input_preprocessed)
    
    # Calculamos la probabilidad de bancarrota
    # Tomamos solo la probabilidad de la clase positiva (bancarrota)
    probability = model_pipeline.named_steps['knn'].predict_proba(input_preprocessed)[:, 1]
    
    # Devolvemos un diccionario con la predicción y la probabilidad
    # prediction: valor binario (0 o 1)
    # probability: valor entre 0 y 1 que indica la probabilidad de bancarrota
    return {
        'prediction': int(prediction[0]),
        'probability': float(probability[0])
    }