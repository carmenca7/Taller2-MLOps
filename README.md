
# Predicción de Bancarrota con KNN y FastAPI

Un proyecto de machine learning que implementa un modelo K-Nearest Neighbors (KNN) para predecir la probabilidad de bancarrota de empresas utilizando indicadores financieros. El modelo se despliega a través de una API construida con FastAPI y se aloja en Replit, permitiendo a los usuarios calcular la probabilidad de bancarrota mediante un endpoint.

## Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Arquitectura del Proyecto](#arquitectura-del-proyecto)
- [Configuración del Entorno](#configuración-del-entorno)
- [Ejecución de la Aplicación](#ejecución-de-la-aplicación)
- [Uso del Endpoint](#uso-del-endpoint)
- [Despliegue en Replit](#despliegue-en-replit)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)

## Descripción del Proyecto

Este proyecto proporciona una herramienta para predecir la probabilidad de bancarrota de una empresa basada en sus indicadores financieros. Utilizamos el modelo KNN debido a su alto rendimiento en términos de Recall, lo cual es crucial en escenarios donde es importante identificar correctamente los casos positivos.

El proyecto incluye:

- **Preprocesamiento de Datos**: Limpieza, tratamiento de valores atípicos mediante winsorización y balanceo de clases con SMOTE
- **Pipeline del Modelo**: Incluye pasos de preprocesamiento y el modelo KNN entrenado
- **API con FastAPI**: Permite a los usuarios enviar datos y recibir predicciones y probabilidades de bancarrota
- **Despliegue en Replit**: La API se aloja en Replit para un acceso fácil y gratuito

## Arquitectura del Proyecto

- `main.py`: Script principal que contiene la API de FastAPI
- `bankruptcy_knn_pipeline.joblib`: Archivo que contiene el pipeline del modelo KNN entrenado
- `requirements.txt`: Lista de dependencias del proyecto
- `README.md`: Documentación del proyecto

## Configuración del Entorno

### Prerrequisitos

- Python 3.7 o superior
- Dependencias listadas en requirements.txt

### Instalación de Dependencias

Instala las dependencias requeridas usando:

```bash
pip install -r requirements.txt
```

Dependencias incluidas en `requirements.txt`:

```
fastapi
uvicorn
joblib
scikit-learn
imbalanced-learn
pandas
numpy
```

## Ejecución de la Aplicación

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/carmenca7/Taller2-MLOps.git
cd tu_repositorio
```

### Paso 2: Ejecutar la Aplicación

Inicia la aplicación usando Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

La aplicación estará disponible en `http://0.0.0.0:8000`.

## Uso del Endpoint

### Descripción del Endpoint

El endpoint `/predict` recibe un conjunto de indicadores financieros y devuelve una predicción sobre la probabilidad de bancarrota.

### Formato de la Solicitud

- **URL**: `http://<tu-dominio-o-url-replit>/predict`
- **Método HTTP**: POST
- **Encabezados**: `Content-Type: application/json`
- **Cuerpo de la Solicitud**: JSON con los indicadores financieros requeridos

### Ejemplo de Solicitud

```json
{
  "ROA(C) before interest and depreciation before interest": 0.5,
  "ROA(A) before interest and % after tax": 1.2,
  "ROA(B) before interest and depreciation after tax": -0.3,
  "Operating Gross Margin": 0.8,
  "Realized Sales Gross Margin": 1.5,
  "...": "..."
}
```

### Ejemplo de Respuesta

```json
{
  "prediction": 0,
  "probability": 0.35
}
```

- `prediction`: 0 indica que no hay riesgo de bancarrota, 1 indica riesgo de bancarrota
- `probability`: Probabilidad asociada a la predicción (entre 0 y 1)

### Prueba con curl

```bash
curl -X POST "http://<tu-dominio-o-url-replit>/predict" \
     -H "Content-Type: application/json" \
     -d '{"Attr1": 0.5, "Attr2": 1.2, "Attr3": -0.3, "Attr4": 0.8, "Attr5": 1.5, "...": "..."}'
```

## Despliegue en Replit

### Paso 1: Hacer Fork y Configurar el Repositorio de GitHub
1. Haz fork de este repositorio en tu cuenta de GitHub
2. Asegúrate de que tu repositorio contenga todos los archivos necesarios:
   - main.py
   - bankruptcy_knn_pipeline.joblib
   - requirements.txt

### Paso 2: Crear Nuevo Proyecto en Replit
1. Ve a [Replit](https://replit.com) e inicia sesión
2. Haz clic en "+ Create Repl"
3. En lugar de seleccionar una plantilla, elige "Import from GitHub"
4. Selecciona tu repositorio forkeado de la lista o pega su URL
5. Selecciona Python como lenguaje

### Paso 3: Configurar el Entorno
1. Replit detectará automáticamente el entorno Python e instalará las dependencias desde requirements.txt
2. Si es necesario, puedes ejecutar manualmente la instalación de dependencias en la Shell:
```bash
pip install -r requirements.txt
```

### Paso 4: Configurar el Comando de Ejecución
1. En tu Repl, haz clic en el botón "Tools"
2. Selecciona "Secrets"
3. Agrega un nuevo secreto con la clave `REPLIT_RUN_COMMAND` y el valor:
```bash
uvicorn main:app --host=0.0.0.0 --port=8000
```

### Paso 5: Habilitar Always On (Opcional)
Si tienes el Plan Hacker de Replit:
1. Ve a la configuración de tu Repl
2. Busca el interruptor "Always On" y actívalo
3. Esto mantendrá tu API funcionando 24/7

### Paso 6: Conectar con GitHub (Recomendado)
1. En la pestaña "Version Control" de tu Repl, conéctate a GitHub
2. Esto te permitirá:
   - Obtener los últimos cambios de GitHub
   - Hacer commit y push de los cambios a GitHub
   - Mantener el control de versiones de tu despliegue

### Paso 7: Lanzar la Aplicación
1. Haz clic en el botón "Run" en Replit
2. Replit proporcionará una URL donde tu API está alojada
3. Prueba el endpoint de la API usando la URL proporcionada

## Contribuciones

¡Las contribuciones son bienvenidas! Para contribuir a este proyecto:

1. Haz fork del repositorio
2. Crea una nueva rama (`git checkout -b feature/nueva-funcionalidad`)
3. Realiza tus cambios y haz commit (`git commit -am 'Agrega nueva funcionalidad'`)
4. Haz push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.
