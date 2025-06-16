import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import joblib


def preprocess_data(
    file_path: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Lee el CSV, limpia y codifica los datos y los divide en
    entrenamiento y prueba de forma estratificada.
    Parámetros
    ----------
    file_path : str
        Ruta al archivo CSV con los datos.
    test_size : float, opcional
        Proporción del conjunto de datos que se usará para la prueba (default: 0.2).
    random_state : int, opcional
        Semilla para la aleatoriedad de la división de datos (default: 42).

    Returns
    -------
    X_train : matriz preprocesada de entrenamiento
    X_test  : matriz preprocesada de prueba
    y_train : Series/array con la variable objetivo (train)
    y_test  : Series/array con la variable objetivo (test)
    """
    # Cargar datos
    df = pd.read_csv(file_path)

    # Separar variable objetivo
    y = df["Heart Disease Status"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Heart Disease Status"])

    # Identificar tipos de columnas
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns
    num_cols = X.select_dtypes(exclude=["object", "category", "bool"]).columns

    # Definir pipelines de preprocesamiento
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Asegurarse de que las columnas categóricas no tengan valores nulos
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Crear el preprocesador con ColumnTransformer
    # ColumnTransformer permite aplicar diferentes transformaciones a diferentes columnas
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols)
        ]
    )

    # 5. Dividir datos (estratificado)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # 6. Ajustar preprocesador y transformar
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Entrena un modelo de clasificación binaria.
    Devuelve el modelo entrenado.
    """
    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    )
    clf.fit(X_train, y_train)
    return clf


def test_model(model, X_test, y_test):
    """Evalúa el modelo de clasificación binaria calculando varias métricas y la matriz de confusión.

    Parámetros:
    -----------
    model : sklearn.base.ClassifierMixin
        Modelo de clasificación ya entrenado que implementa los métodos `predict` y `predict_proba`.
    X_test : array-like o pandas.DataFrame de forma (n_samples, n_features)
        Conjunto de datos de prueba (características) sobre el que se evaluará el modelo.
    y_test : array-like o pandas.Series de forma (n_samples,)
        Etiquetas verdaderas correspondientes a X_test.

    Devuelve:
    ---------
    results : dict
        Diccionario con las siguientes métricas:
            - "accuracy": Precisión global del modelo.
            - "precision": Precisión (positive predictive value).
            - "recall": Exhaustividad (sensibilidad).
            - "f1": Puntaje F1.
            - "roc_auc": Área bajo la curva ROC.
            - "confusion_matrix": Matriz de confusión 2x2 en formato [[TN, FP], [FN, TP]].
    """
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)  # [[TN FP] [FN TP]]

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": cm,
    }
    return results


def save_model(model, model_name: str = None) -> str:
    """
    Guarda el modelo entrenado en disco para su posterior uso.

    Parámetros
    ----------
    model : objeto sklearn/base
        Modelo entrenado a guardar.
    model_name : str, opcional
        Ruta y nombre del archivo donde guardar el modelo. Si se omite,
        se usará 'trained_model.joblib' en el directorio actual.

    Devuelve
    -------
    str
        Ruta y nombre del fichero donde se ha almacenado el modelo.
    """
    if model_name is None:
        model_name = "trained_model.joblib"
    joblib.dump(model, model_name)
    return model_name


def load_model(model_name: str):
    """
    Carga un modelo previamente almacenado desde disco.

    Parámetros
    ----------
    model_name : str
        Ruta y nombre del archivo donde se encuentra el modelo guardado.

    Devuelve
    -------
    model : objeto sklearn/base
        Modelo entrenado listo para usar.
    """
    return joblib.load(model_name)


def main():
    print("\n=== Iniciando el proceso de entrenamiento y evaluación del modelo ===\n")
    file_path = "heart_disease.csv"          # ajusta si es necesario
    
    # Preprocesar los datos
    proporcion = 0.25
    aleatoriedad = None
    X_train, X_test, y_train, y_test = preprocess_data(file_path, proporcion, aleatoriedad)
    print(f"Fichero procesado: {file_path}")
    print(f"Datos preprocesados: {X_train.shape[0]} muestras de entrenamiento y {X_test.shape[0]} muestras de prueba.")

    # Entrenar el modelo	
    model = train_model(X_train, y_train)
    print("Modelo entrenado con éxito.")

    # Guardar el modelo entrenado
    model_path = save_model(model)
    print(f"Modelo guardado en: {model_path}")

    # Cargar el modelo desde disco
    loaded_model = load_model(model_path)
    print(f"Modelo {model_path} cargado desde disco.")

    # Evaluar el modelo cargado
    results = test_model(loaded_model, X_test, y_test)
    print("Evaluación del modelo completada.")

    # Métricas principales
    print("=== Métricas del modelo ===")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"{key:>9}: {results[key]:.4f}")

    # Explicación de las métricas
    print("""
Significado de las métricas:
────────────────────────────
• accuracy  : Proporción de predicciones correctas sobre el total de casos.
              (¿Cuántos aciertos tiene el modelo en general?)
• precision : De los casos que el modelo predijo como positivos (enfermos),
              ¿cuántos realmente lo eran? (Evita falsos positivos)
• recall    : De todos los casos realmente positivos (enfermos),
              ¿cuántos detectó el modelo? (Evita falsos negativos)
• f1        : Media armónica entre precisión y recall. Resume ambos en un solo valor.
• roc_auc   : Capacidad del modelo para distinguir entre clases. 0.5 = azar, 1.0 = perfecto.
""")

    # Matriz de confusión en forma tabular
    cm = results["confusion_matrix"]
    cm_df = pd.DataFrame(
        cm,
        index=["Real No", "Real Sí"],
        columns=["Pred No", "Pred Sí"]
    )
    print("\n=== Matriz de confusión ===")
    print(cm_df.to_string())

    # Explicación sencilla
    print("""
¿Qué significa?
────────────────
• Fila = situación REAL   · Columna = predicción del MODELO

┌─────────┬────────┬────────┐
│         │ Pred No│ Pred Sí│
├─────────┼────────┼────────┤
│ Real No │  TN    │  FP    │
│ Real Sí │  FN    │  TP    │
└─────────┴────────┴────────┘

TN (True Negatives) : casos sanos correctamente identificados como sanos.  
TP (True Positives) : personas con enfermedad detectadas correctamente.  
FP (False Positives): sanos que el modelo marcó erróneamente como enfermos.  
FN (False Negatives): enfermos que el modelo dejó escapar (no los detectó).

En un problema médico suele preocuparnos especialmente bajar los FN
(mejorar el 'recall'), porque implican no detectar a alguien con enfermedad. 
""")


if __name__ == "__main__":
    main()
