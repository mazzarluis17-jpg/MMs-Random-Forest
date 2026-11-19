"""Evaluación fuera de muestra de la estimación humana a partir de la carrera."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from src.analisis import SEMILLA


def _metricas(y, pred):
    return {"MAE": mean_absolute_error(y, pred),
            "RMSE": np.sqrt(mean_squared_error(y, pred)), "R2": r2_score(y, pred)}


def _busqueda(semilla):
    bosque = make_pipeline(
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        RandomForestRegressor(n_estimators=100, random_state=semilla, n_jobs=1),
    )
    return GridSearchCV(
        bosque, {"randomforestregressor__max_depth": [1, 3, None],
                 "randomforestregressor__min_samples_leaf": [1, 3, 6]},
        scoring="neg_mean_absolute_error",
        cv=KFold(4, shuffle=True, random_state=semilla), error_score="raise",
    )


def evaluar(datos, repeticiones=5):
    """Compara tres modelos con CV 5×5 y selección interna del bosque por MAE."""
    if len(datos) < 10 or repeticiones < 1 or datos.carrera.nunique() < 2:
        raise ValueError("Se necesitan al menos diez filas y dos carreras")
    x, y = datos[["carrera"]], datos.prediccion
    predicciones, pliegues, grilla = [], [], []
    for repeticion in range(1, repeticiones + 1):
        exterior = KFold(5, shuffle=True, random_state=SEMILLA + repeticion)
        for pliegue, (entrena, prueba) in enumerate(exterior.split(x), start=1):
            xt, xv = x.iloc[entrena], x.iloc[prueba]
            yt, yv = y.iloc[entrena], y.iloc[prueba]
            busqueda = _busqueda(SEMILLA + 100 * repeticion + pliegue).fit(xt, yt)
            identificador = {"repeticion": repeticion, "pliegue": pliegue}
            for params, score, rank in zip(busqueda.cv_results_["params"],
                                           busqueda.cv_results_["mean_test_score"],
                                           busqueda.cv_results_["rank_test_score"]):
                grilla.append({**identificador, "profundidad": params[
                    "randomforestregressor__max_depth"], "hoja_minima": params[
                    "randomforestregressor__min_samples_leaf"], "MAE_interno": -score,
                    "rango": rank, "seleccionado": params == busqueda.best_params_})
            modelos = {
                "Dummy media": DummyRegressor(strategy="mean").fit(xt, yt),
                "Dummy mediana": DummyRegressor(strategy="median").fit(xt, yt),
                "Bosque aleatorio": busqueda.best_estimator_,
            }
            for nombre, modelo in modelos.items():
                pred = modelo.predict(xv)
                ident = {**identificador, "modelo": nombre}
                pliegues.append({**ident, **_metricas(yv, pred),
                                "MAE_entrenamiento": mean_absolute_error(yt, modelo.predict(xt))})
                predicciones.append(datos.iloc[prueba].assign(
                    **ident, predicha=pred, residuo=yv.to_numpy() - pred,
                    carrera_vista=xv.carrera.isin(xt.carrera).to_numpy(),
                ))
    predicciones = pd.concat(predicciones, ignore_index=True)
    por_repeticion = pd.DataFrame([
        {"modelo": nombre, "repeticion": rep, **_metricas(g.prediccion, g.predicha)}
        for (nombre, rep), g in predicciones.groupby(["modelo", "repeticion"])
    ])
    return predicciones, pd.DataFrame(pliegues), por_repeticion, pd.DataFrame(grilla)


def ajustar_final(datos):
    """Reajusta el bosque sobre todas las filas para inspección posterior."""
    return _busqueda(SEMILLA).fit(datos[["carrera"]], datos.prediccion)
