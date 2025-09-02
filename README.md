# Proyecto ML: Predicción de Lluvia (WeatherAUS)

Un proyecto para predecir la variable `RainTomorrow` en Australia usando técnicas de Machine Learning: desde exploración de datos hasta evaluación avanzada e interpretabilidad. Generado bajo el Diplomado de Machine Learning en Python de la Universidad EAFIT.

## 📂 Estructura del repositorio
```
ML_Proyecto_WeatherAUS/
├── data/
│   ├── raw/
│   │   └── weatherAUS.csv           # Datos originales descargados
│   └── processed/
│       └── weather_prepared_selected.csv  # Datos filtrados y transformados
├── models/
│   ├── best_pipeline.joblib         # Pipeline final entrenado
│   └── best_metrics.json            # Métricas clave del modelo
├── notebooks/
│   ├── 01_eda.ipynb                 # Análisis exploratorio de datos
│   ├── 02_preprocessing_feature_engineering.ipynb  # Limpieza e ingeniería de atributos
│   ├── 03_modeling.ipynb            # Entrenamiento, validación cruzada y tuning
│   └── 05_final_report_support.ipynb  # Celdas de apoyo para generar reportes finales
├── reports/
│   └── figures/                     # Figuras generadas (ROC, PR, confusión, SHAP, calibración)
├── requirements.txt                 # Dependencias del proyecto
└── README.md                        # Documentación consolidadora (este archivo)
```

## 🛠️ Requisitos e instalación
1. Clonar este repositorio:
   ```bash
   git clone <repo_url>
   cd ML_Proyecto_WeatherAUS
   ```
2. Crear e instalar entorno Python (recomendado 3.8+):
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   pip install -r requirements.txt
   ```

## 🚀 Flujo de trabajo
1. **01_eda.ipynb**: carga y limpieza inicial, análisis de distribuciones, valores faltantes y correlaciones.
2. **02_preprocessing_feature_engineering.ipynb**: imputación, normalización, codificación de categorías, selección de variables y guardado de datos procesados.
3. **03_modeling.ipynb**: definición de pipeline, entrenamientos de baseline y XGBoost, búsqueda de hiperparámetros con validación cruzada estratificada.
4. **04_evaluation_error_analysis.ipynb**: evaluación final en test, cálculo de métricas (ROC AUC, PR AUC), optimización de umbrales (F1, coste y Youden), matriz de confusión, análisis de falsos positivos/negativos, calibración de probabilidades e interpretabilidad (importancias y SHAP).

## 📊 Resultados clave
- **Validación:** ROC AUC = 0.9163
- **Prueba (Test):** ROC AUC = 0.8906, PR AUC = 0.7468
- **Mejor umbral (F1):** ~0.317  → F1-score = 0.7140
- **Comparativa umbrales en Test:**
  - Umbral 0.50:  Precision = 0.8189, Recall = 0.5703, F1 = 0.6723
  - Umbral 0.32:  Precision = 0.6860, Recall = 0.7445, F1 = 0.7140
- **Umbral costo mínimo:** 0.1585 (Coste = 8743)
- **Umbral Youden:** 0.2476 (Youden = 0.6679)

Matriz de confusión (umbral ~0.317):
```
[[TN=19,890 | FP=2,173]
 [FN=1,629  | TP=4,747]]
```

## 🔍 Interpretabilidad y análisis de errores
- **Importancia de características:** Top 20 variables según el modelo (figura en `reports/figures/feature_importances_model_specific.png`).
- **SHAP summary plot:** impacto y dirección de cada variable por instancia (archivo `reports/figures/shap_summary_valid.png`).
- **Calibración:** curva de confiabilidad (`reports/figures/eval_calibration.png`) para verificar la calidad de las probabilidades.

## 👤 Autor y contacto
Federico Gómez – fjgomezc@eafit.edu.co