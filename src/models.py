import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from xgboost import XGBClassifier


def temporal_split(
    df: pd.DataFrame,
    features: list,
    target: str = "target",
) -> tuple:
    """
    Split temporal sin shuffle: train 2008-2024, val 2025, test 2026 - 2026-06-18.

    El split es estrictamente por año para evitar cualquier forma de data leakage.

    Parámetros
    ----------
    df : pd.DataFrame
        Feature matrix con índice DatetimeIndex, salida de
        features.build_feature_matrix().
    features : list
        Nombres de las columnas a usar como features.
    target : str
        Nombre de la columna target.

    Retorna
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    train = df[df.index.year <= 2024]
    val   = df[df.index.year == 2025]
    test  = df[df.index.year >= 2026]

    X_train, y_train = train[features].values, train[target].values
    X_val,   y_val   = val[features].values,   val[target].values
    X_test,  y_test  = test[features].values,  test[target].values

    print(f"[models] Split temporal:")
    print(f"  train: {len(X_train):,} muestras ({train.index[0].date()} → {train.index[-1].date()})")
    print(f"  val:   {len(X_val):,} muestras ({val.index[0].date()} → {val.index[-1].date()})")
    print(f"  test:  {len(X_test):,} muestras ({test.index[0].date()} → {test.index[-1].date()})")

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test) -> tuple:
    """
    Aplica StandardScaler fiteado solo sobre X_train, y lo usa para transformar
    val y test.

    Parámetros
    ----------
    X_train : array
        Features de entrenamiento.
    X_val : array
        Features de validación.
    X_test : array
        Features de test.

    Retorna
    -------
    tuple
        (X_train_scaled, X_val_scaled, X_test_scaled, scaler). El scaler se
        devuelve para guardarlo y usarlo en predict.py sobre el test set externo.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """
    Entrena Logistic Regression con class_weight='balanced'.

    class_weight='balanced' pondera las clases inversamente proporcional a su
    frecuencia, compensando el leve desbalance sin necesidad de resampling.

    Parámetros
    ----------
    X_train : array
        Features de entrenamiento (ya escaladas).
    y_train : array
        Target de entrenamiento.

    Retorna
    -------
    LogisticRegression
        Modelo ya entrenado.
    """
    modelo = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    modelo.fit(X_train, y_train)
    return modelo


def train_xgboost(X_train, y_train, X_val, y_val) -> XGBClassifier:
    """
    Entrena XGBClassifier con scale_pos_weight para manejar desbalance de clases.

    scale_pos_weight = n_negativos / n_positivos — equivalente a class_weight
    en sklearn. Usa early stopping sobre val para evitar overfitting: para si
    el AUC no mejora en 50 rondas.

    Parámetros
    ----------
    X_train : array
        Features de entrenamiento (ya escaladas).
    y_train : array
        Target de entrenamiento.
    X_val : array
        Features de validación (ya escaladas), usadas para early stopping.
    y_val : array
        Target de validación.

    Retorna
    -------
    XGBClassifier
        Modelo ya entrenado.
    """
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())

    modelo = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=n_neg / n_pos,
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=42,
        verbosity=0,
    )
    modelo.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print(f"[models] XGBoost entrenado. Mejor ronda: {modelo.best_iteration}")
    return modelo


def train_mlp(X_train, y_train, X_val, y_val) -> MLPClassifier:
    """
    Entrena MLPClassifier con arquitectura (128, 64), ReLU, Adam, early stopping.

    early_stopping=True reserva el 10% de X_train para validación interna de
    sklearn, pero al pasarle X_val directamente al constructor no es posible
    en sklearn puro. Usamos validation_fraction=0.1 y monitoreamos con el
    X_val externo para reproducibilidad.

    Parámetros
    ----------
    X_train : array
        Features de entrenamiento (ya escaladas).
    y_train : array
        Target de entrenamiento.
    X_val : array
        Features de validación (ya escaladas). No se usa directamente
        dentro del fit (limitación de sklearn), se deja como argumento
        para evaluar después con evaluate_model().
    y_val : array
        Target de validación.

    Retorna
    -------
    MLPClassifier
        Modelo ya entrenado.
    """
    modelo = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
    )
    modelo.fit(X_train, y_train)
    return modelo


def evaluate_model(model, X, y, split_name: str = "test") -> dict:
    """
    Evalúa un modelo y retorna sus métricas como dict.

    Métricas: accuracy, f1 (weighted), roc_auc, precision (weighted),
    recall (weighted).

    Parámetros
    ----------
    model : objeto sklearn-like
        Modelo ya entrenado, con métodos predict() y predict_proba().
    X : array
        Features sobre las que evaluar.
    y : array
        Target verdadero.
    split_name : str
        Nombre del split que se está evaluando (solo para el print).

    Retorna
    -------
    dict
        Métricas calculadas, con la clave 'split' incluida.
    """

    y_prob = model.predict_proba(X)[:, 1]
    umbral = 0.5
    y_pred = (y_prob >= umbral).astype(int)

    metricas = {
        "split":     split_name,
        "accuracy":  round(accuracy_score(y, y_pred), 4),
        "f1":        round(f1_score(y, y_pred, average="weighted"), 4),
        "roc_auc":   round(roc_auc_score(y, y_prob), 4),
        "precision": round(precision_score(y, y_pred, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y, y_pred, average="weighted"), 4),
    }

    print(f"\n── {split_name.upper()} ──")
    for k, v in metricas.items():
        if k != "split":
            print(f"  {k:<12}: {v}")

    return metricas


def save_model(model, scaler, model_name: str, models_dir: str) -> None:
    """
    Guarda el modelo y el scaler con joblib para reutilizarlos en predict.py.

    Parámetros
    ----------
    model : objeto sklearn-like
        Modelo entrenado a guardar.
    scaler : StandardScaler
        Scaler fiteado a guardar junto con el modelo.
    model_name : str
        Nombre con el que se guardan los archivos (sin extensión).
    models_dir : str
        Directorio donde se guardan model_name.pkl y model_name_scaler.pkl.
    """
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model,  os.path.join(models_dir, f"{model_name}.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, f"{model_name}_scaler.pkl"))
    print(f"[models] {model_name} guardado en {models_dir}/")


def load_model(model_name: str, models_dir: str) -> tuple:
    """
    Carga modelo y scaler guardados con joblib.

    Parámetros
    ----------
    model_name : str
        Nombre con el que se guardaron los archivos (sin extensión).
    models_dir : str
        Directorio donde están model_name.pkl y model_name_scaler.pkl.

    Retorna
    -------
    tuple
        (model, scaler)
    """
    model  = joblib.load(os.path.join(models_dir, f"{model_name}.pkl"))
    scaler = joblib.load(os.path.join(models_dir, f"{model_name}_scaler.pkl"))
    return model, scaler


def shap_analysis(xgb_model, X_train, X_test, feature_names: list) -> None:
    """
    Genera análisis SHAP para el modelo XGBoost.

    Produce: (1) un summary plot con la importancia global de cada feature
    (beeswarm), y (2) waterfall plots explicando 3 predicciones individuales
    del test set.

    Parámetros
    ----------
    xgb_model : XGBClassifier
        Modelo XGBoost ya entrenado.
    X_train : array
        Features de entrenamiento (ya escaladas), usadas para el summary plot.
    X_test : array
        Features de test (ya escaladas), usadas para los waterfall plots.
    feature_names : list
        Nombres de las columnas, en el mismo orden que X_train/X_test.
    """
    explainer   = shap.TreeExplainer(xgb_model)
    shap_train  = explainer(pd.DataFrame(X_train, columns=feature_names))
    shap_test   = explainer(pd.DataFrame(X_test,  columns=feature_names))

    # Summary plot
    print("\n── SHAP Summary Plot (importancia global) ──")
    shap.summary_plot(shap_train, pd.DataFrame(X_train, columns=feature_names), show=True)

    # Waterfall para 3 ejemplos del test set
    y_pred_test = xgb_model.predict(X_test)
    y_true_test = None  # se pasa desde el notebook si se quiere filtrar TP/FP/FN

    for i, label in enumerate(["Ejemplo 1", "Ejemplo 2", "Ejemplo 3"]):
        if i >= len(shap_test):
            break
        print(f"\n── Waterfall — {label} (predicción: {y_pred_test[i]}) ──")
        shap.waterfall_plot(shap_test[i], show=True)
