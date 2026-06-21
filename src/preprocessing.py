import numpy as np
import pandas as pd


def dataset_summary(name: str, df: pd.DataFrame) -> None:
    """
    Imprime resumen estadístico de un DataFrame: dtypes, nulls y valores únicos.

    Parámetros
    ----------
    name : str
        Nombre descriptivo del dataset, usado en el encabezado impreso.
    df : pd.DataFrame
        DataFrame a resumir.

    Retorna
    -------
    None
        Imprime el resumen directamente.
    """
    summary = pd.DataFrame({
        "dtype":    df.dtypes,
        "non_null": df.notnull().sum(),
        "null":     df.isnull().sum(),
        "null_%":   (df.isnull().sum() / len(df) * 100).round(2),
        "unique":   df.nunique(),
    })
    print(f"\n{'='*55}")
    print(f"  {name}  |  {len(df):,} rows  x  {len(df.columns)} columns")
    print(f"{'='*55}")
    display(summary)
    print(f"\n── Random sample (5 rows) ──")
    display(df.sample(5, random_state=42))
