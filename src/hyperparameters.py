import numpy as np
import pandas as pd
import optuna

from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)


def walk_forward_bayesian_search(
    df: pd.DataFrame,
    features: list,
    target: str = "target",
    n_trials: int = 50,
    val_years: list = None,
) -> tuple:
    """
    Walk-forward expanding window con Bayesian search (Optuna TPE) para XGBoost.

    Estructura de folds:
        Train 2008-2019 / Val 2020
        Train 2008-2020 / Val 2021
        ...
        Train 2008-2023 / Val 2024

    Optimiza AUC-ROC promedio sobre los folds. El test set (2025) no se toca.

    Parámetros
    ----------
    df : pd.DataFrame
        Feature matrix completa con DatetimeIndex.
    features : list
        Columnas a usar como features.
    target : str
        Columna target.
    n_trials : int
        Número de trials Optuna.
    val_years : list
        Años de validación de cada fold. Default [2020, 2021, 2022, 2023, 2024].

    Retorna
    -------
    tuple
        (best_params dict, optuna.Study)
    """
    if val_years is None:
        val_years = [2020, 2021, 2022, 2023, 2024]

    def objective(trial):
        params = dict(
            n_estimators          = 500,
            max_depth             = trial.suggest_int("max_depth", 3, 6),
            learning_rate         = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample             = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree      = trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight      = trial.suggest_int("min_child_weight", 1, 7),
            reg_alpha             = trial.suggest_float("reg_alpha", 0.0, 2.0),
            reg_lambda            = trial.suggest_float("reg_lambda", 0.5, 3.0),
            gamma                 = trial.suggest_float("gamma", 0.0, 2.0),
            eval_metric           = "auc",
            early_stopping_rounds = 30,
            random_state          = 42,
            verbosity             = 0,
        )

        fold_aucs = []
        for val_year in val_years:
            tr = df[df.index.year < val_year]
            vl = df[df.index.year == val_year]

            X_tr, y_tr = tr[features].values, tr[target].values
            X_vl, y_vl = vl[features].values, vl[target].values

            n_neg = int((y_tr == 0).sum())
            n_pos = int((y_tr == 1).sum())

            model = XGBClassifier(**params, scale_pos_weight=n_neg / n_pos)
            model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)

            fold_aucs.append(roc_auc_score(y_vl, model.predict_proba(X_vl)[:, 1]))

        return float(np.mean(fold_aucs))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\n[hyperparameters] Walk-forward Bayesian search — {n_trials} trials, {len(val_years)} folds")
    print(f"  Mejor AUC-CV promedio: {study.best_value:.4f}")
    print(f"  Mejores parámetros:")
    for k, v in best.items():
        print(f"    {k:<22}: {round(v, 5) if isinstance(v, float) else v}")

    return best, study


def train_xgboost_tuned(X_train, y_train, X_val, y_val, params: dict) -> XGBClassifier:
    """
    Entrena XGBoost con los hiperparámetros encontrados por walk_forward_bayesian_search.

    Parámetros
    ----------
    X_train, y_train : array
        Datos de entrenamiento (ya escalados).
    X_val, y_val : array
        Datos de validación (ya escalados), usados para early stopping.
    params : dict
        Hiperparámetros devueltos por walk_forward_bayesian_search.

    Retorna
    -------
    XGBClassifier
        Modelo ya entrenado.
    """
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())

    modelo = XGBClassifier(
        n_estimators          = 500,
        max_depth             = params.get("max_depth", 4),
        learning_rate         = params.get("learning_rate", 0.05),
        subsample             = params.get("subsample", 0.8),
        colsample_bytree      = params.get("colsample_bytree", 0.8),
        min_child_weight      = params.get("min_child_weight", 1),
        reg_alpha             = params.get("reg_alpha", 0.0),
        reg_lambda            = params.get("reg_lambda", 1.0),
        gamma                 = params.get("gamma", 0.0),
        scale_pos_weight      = n_neg / n_pos,
        eval_metric           = "auc",
        early_stopping_rounds = 50,
        random_state          = 42,
        verbosity             = 0,
    )
    modelo.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"[hyperparameters] XGBoost tuneado. Mejor ronda: {modelo.best_iteration}")
    return modelo


# ── MLP ───────────────────────────────────────────────────────────────────────

def walk_forward_bayesian_search_mlp(
    df: pd.DataFrame,
    features: list,
    target: str = "target",
    n_trials: int = 50,
    val_years: list = None,
) -> tuple:
    """
    Walk-forward expanding window con Bayesian search (Optuna TPE) para MLP.

    Espacio de búsqueda sugerido (implementar en el cuerpo de objective):
        hidden_layer_sizes : trial.suggest_categorical → [(64,), (128,64), (256,128), (256,128,64)]
        learning_rate_init : trial.suggest_float(..., 1e-4, 1e-2, log=True)
        alpha              : trial.suggest_float(..., 1e-5, 1e-1, log=True)
        activation         : trial.suggest_categorical → ["relu", "tanh"]

    Parámetros
    ----------
    df : pd.DataFrame
        Feature matrix completa con DatetimeIndex.
    features : list
        Columnas a usar como features.
    target : str
        Columna target.
    n_trials : int
        Número de trials Optuna.
    val_years : list
        Años de validación de cada fold. Default [2020, 2021, 2022, 2023, 2024].

    Retorna
    -------
    tuple
        (best_params dict, optuna.Study)
    """
    if val_years is None:
        val_years = [2020, 2021, 2022, 2023, 2024]

    def objective(trial):
        hidden = trial.suggest_categorical(
            "hidden_layer_sizes",
            [(64,), (128, 64), (256, 128), (256, 128, 64)],
        )
        params = dict(
            hidden_layer_sizes  = hidden,
            activation          = trial.suggest_categorical("activation", ["relu", "tanh"]),
            learning_rate_init  = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            alpha               = trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            solver              = "adam",
            max_iter            = 300,
            early_stopping      = True,
            validation_fraction = 0.1,
            n_iter_no_change    = 15,
            random_state        = 42,
        )

        fold_aucs = []
        for val_year in val_years:
            tr = df[df.index.year < val_year]
            vl = df[df.index.year == val_year]

            X_tr, y_tr = tr[features].values, tr[target].values
            X_vl, y_vl = vl[features].values, vl[target].values

            # StandardScaler fiteado solo en train de cada fold, igual que en producción
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_vl_s = scaler.transform(X_vl)

            model = MLPClassifier(**params)
            model.fit(X_tr_s, y_tr)

            fold_aucs.append(roc_auc_score(y_vl, model.predict_proba(X_vl_s)[:, 1]))

        return float(np.mean(fold_aucs))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\n[hyperparameters] Walk-forward Bayesian search MLP — {n_trials} trials, {len(val_years)} folds")
    print(f"  Mejor AUC-CV promedio: {study.best_value:.4f}")
    print(f"  Mejores parámetros:")
    for k, v in best.items():
        print(f"    {k:<22}: {round(v, 5) if isinstance(v, float) else v}")

    return best, study


def train_mlp_tuned(X_train, y_train, params: dict) -> MLPClassifier:
    """
    Entrena MLP con los hiperparámetros encontrados por walk_forward_bayesian_search_mlp.

    Parámetros
    ----------
    X_train, y_train : array
        Datos de entrenamiento (ya escalados).
    params : dict
        Hiperparámetros devueltos por walk_forward_bayesian_search_mlp.

    Retorna
    -------
    MLPClassifier
        Modelo ya entrenado.
    """
    modelo = MLPClassifier(
        hidden_layer_sizes  = params.get("hidden_layer_sizes", (128, 64)),
        activation          = params.get("activation", "relu"),
        learning_rate_init  = params.get("learning_rate_init", 0.001),
        alpha               = params.get("alpha", 0.0001),
        solver              = "adam",
        max_iter            = 300,
        early_stopping      = True,
        validation_fraction = 0.1,
        n_iter_no_change    = 15,
        random_state        = 42,
    )
    modelo.fit(X_train, y_train)
    print(f"[hyperparameters] MLP tuneado. Épocas: {modelo.n_iter_}")
    return modelo
