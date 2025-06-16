# Heart Disease AI Library

Este proyecto contiene una librería y scripts en Python para el entrenamiento, evaluación y uso de un modelo de regresión logística que predice la presencia de enfermedad cardíaca a partir de datos tabulares.

## Estructura principal

- `ailibrary.py`: Librería con funciones para preprocesar datos, entrenar, guardar, cargar y evaluar modelos de clasificación binaria.
- `app.py`: Script principal que ejecuta el flujo completo: preprocesamiento, entrenamiento, guardado, carga y evaluación del modelo, mostrando métricas y explicaciones.
- `data.csv`: Archivo de datos de entrada (debe estar en la misma carpeta que `app.py`).
- `requirements.txt`: Lista de dependencias necesarias.

## Instalación de dependencias

Se recomienda usar un entorno virtual. Para instalar las librerías necesarias:

```bash
python -m venv ai-env
source ai-env/bin/activate  # En Linux/Mac
ai-env\Scripts\activate    # En Windows
pip install -r requirements.txt
```

## Ejecución del script principal

Asegúrate de que `data.csv` esté en la misma carpeta que `app.py`.

```bash
python app.py
```

El script realizará:
- Preprocesamiento de los datos
- Entrenamiento del modelo
- Guardado y carga del modelo
- Evaluación y muestra de métricas

## Salida y significado de las métricas

El script imprime varias métricas de evaluación:

- **accuracy**: Proporción de predicciones correctas sobre el total de casos.
- **precision**: De los casos predichos como positivos (enfermos), ¿cuántos realmente lo eran? (Evita falsos positivos)
- **recall**: De todos los casos realmente positivos (enfermos), ¿cuántos detectó el modelo? (Evita falsos negativos)
- **f1**: Media armónica entre precisión y recall. Resume ambos en un solo valor.
- **roc_auc**: Capacidad del modelo para distinguir entre clases. 0.5 = azar, 1.0 = perfecto.

También muestra la matriz de confusión:

|           | Pred No | Pred Sí |
|-----------|---------|---------|
| Real No   |   TN    |   FP    |
| Real Sí   |   FN    |   TP    |

- **TN (True Negatives)**: sanos correctamente identificados como sanos.
- **TP (True Positives)**: enfermos correctamente identificados.
- **FP (False Positives)**: sanos que el modelo marcó erróneamente como enfermos.
- **FN (False Negatives)**: enfermos que el modelo no detectó.

En problemas médicos, suele ser prioritario reducir los FN (mejorar el recall), para no dejar enfermos sin detectar.

## Notas
- Puedes modificar el script para ajustar hiperparámetros o probar otros modelos.
- Si tienes dudas sobre el significado de las métricas, revisa la explicación incluida en la salida del script.
