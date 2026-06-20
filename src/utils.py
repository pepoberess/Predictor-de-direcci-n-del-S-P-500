import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc


def dataset_summary(name: str, df: pd.DataFrame) -> None:
    """Imprime resumen estadístico de un DataFrame: dtypes, nulls y valores únicos."""
    summary = pd.DataFrame({
        "dtype":    df.dtypes,
        "non_null": df.notnull().sum(),
        "null":     df.isnull().sum(),
        "null_%":   (df.isnull().sum() / len(df) * 100).round(2),
        "unique":   df.nunique(),
    })
    print(f"\n{'='*55}")
    print(f"  {name}  |  {len(df):,} filas  x  {len(df.columns)} columnas")
    print(f"{'='*55}")
    display(summary)
    print(f"\n── Muestra aleatoria (5 filas) ──")
    display(df.sample(5, random_state=42))


def set_plot_style() -> None:
    """Configura estilo visual consistente para todos los notebooks."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "figure.figsize": (12, 5),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def plot_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    """Grafica la matriz de confusión con etiquetas 'Baja' / 'Sube'."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=["Baja (0)", "Sube (1)"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(f"Matriz de Confusión — {model_name}")
    plt.tight_layout()
    plt.show()


def plot_roc_curves(models_dict: dict, X_test, y_test) -> None:
    """Grafica curvas ROC superpuestas para múltiples modelos.

    Args:
        models_dict: {nombre_modelo: modelo_entrenado}
        X_test: features del conjunto de test (ya escaladas)
        y_test: etiquetas verdaderas del test
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for nombre, modelo in models_dict.items():
        y_score = modelo.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{nombre} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", label="Aleatorio (AUC = 0.500)")
    ax.set_xlabel("Tasa de Falsos Positivos")
    ax.set_ylabel("Tasa de Verdaderos Positivos")
    ax.set_title("Curvas ROC — Comparación de Modelos")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
