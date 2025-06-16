from ailibrary import preprocess_data, train_model, test_model, save_model, load_model
import pandas as pd
import os

def main():
    print("\n=== Iniciando el proceso de entrenamiento y evaluación del modelo ===\n")
    current_path = os.path.dirname(os.path.abspath(__file__))
    file_path =  current_path + "/data.csv"

    # Preprocesar los datos
    proporcion = 0.20
    aleatoriedad = 42
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
