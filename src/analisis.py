"""Carga validada y análisis estadístico de las estimaciones del frasco."""
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEMILLA = 1837
RAIZ = Path(__file__).resolve().parents[1]
ORIGINAL = RAIZ / "M&Ms - Sheet1.csv"


def cargar(ruta=ORIGINAL):
    """Preserva cada respuesta y devuelve una tabla limpia y el registro de auditoría."""
    with Path(ruta).open(encoding="utf-8-sig", newline="") as archivo:
        lector = csv.reader(archivo)
        if next(lector, None) != ["carrera", "prediccion", "", "Promedio"]:
            raise ValueError("Encabezado inesperado en el CSV")
        filas = list(lector)
    if not filas or any(len(fila) != 4 for fila in filas):
        raise ValueError("CSV vacío o con filas de ancho irregular")
    datos = pd.DataFrame(filas, columns=["carrera", "prediccion", "vacia", "Promedio"])
    if datos["vacia"].str.strip().ne("").any():
        raise ValueError("La columna auxiliar sin encabezado contiene datos")
    carrera = datos["carrera"].str.strip()
    conteos = datos["prediccion"].str.strip()
    if carrera.eq("").any() or not conteos.str.fullmatch(r"[0-9]+").all():
        raise ValueError("Se requieren categorías presentes y conteos enteros positivos")
    conteos = conteos.astype("int64")
    if conteos.le(0).any():
        raise ValueError("Las estimaciones deben ser positivas")
    limpio = pd.DataFrame({"carrera": carrera, "prediccion": conteos})
    coincidencias = limpio.duplicated(keep=False)
    auditoria = {
        "filas": len(limpio),
        "columnas_originales": list(datos.columns),
        "faltantes_por_columna": datos.apply(lambda s: int(s.str.strip().eq("").sum())).to_dict(),
        "etiquetas_con_espacios_recortados": int(carrera.ne(datos["carrera"]).sum()),
        "pares_repetidos_distintos": len(limpio.loc[coincidencias].drop_duplicates()),
        "filas_en_pares_repetidos": int(coincidencias.sum()),
        "repeticiones_adicionales": int(limpio.duplicated().sum()),
        "valores_columna_promedio": datos.loc[datos.Promedio.ne(""), "Promedio"].tolist(),
    }
    limpio.insert(0, "fila_csv", np.arange(2, len(limpio) + 2))
    return limpio, auditoria


def descriptivos(datos):
    """Calcula momentos muestrales, cuartiles lineales y sensibilidad a extremos."""
    y = datos.prediccion
    q1, mediana, q3 = y.quantile([.25, .5, .75])
    iqr = q3 - q1
    inferior, superior = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    extremos = y.lt(inferior) | y.gt(superior)
    resumen = pd.Series(dict(
        n=len(y), media=y.mean(), mediana=mediana, moda=", ".join(y.mode().astype(str)),
        frecuencia_modal=y.value_counts().max(), varianza_muestral=y.var(),
        desviacion_muestral=y.std(), q1=q1, q3=q3, IQR=iqr, minimo=y.min(),
        maximo=y.max(), CV_pct=100 * y.std() / y.mean(),
        asimetria=stats.skew(y, bias=False), exceso_curtosis=stats.kurtosis(y, bias=False),
        cerca_inferior=inferior, cerca_superior=superior, n_extremos=int(extremos.sum()),
    ), name="valor")
    escenarios = {
        "todas": y, "sin_maximo": y.drop(y.idxmax()), "dentro_de_cercas_Tukey": y[~extremos],
    }
    sensibilidad = pd.DataFrame([
        {"escenario": nombre, "n": len(valores), "media": valores.mean(),
         "mediana": valores.median()} for nombre, valores in escenarios.items()
    ])
    return resumen, datos.loc[extremos], sensibilidad


def bootstrap(y, repeticiones=20000):
    """Estimadores colectivos e IC percentiles del 95 % por remuestreo de filas."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size < 2 or not np.isfinite(y).all() or repeticiones < 2:
        raise ValueError("Se requieren un vector finito y al menos dos réplicas")
    rng = np.random.default_rng(SEMILLA)
    replicas = rng.choice(y, (repeticiones, y.size), replace=True)
    k = int(.10 * y.size)

    def estimar(muestras):
        orden = np.sort(muestras, axis=1)
        centro = orden[:, k:y.size - k]
        limitada = np.clip(orden, centro[:, :1], centro[:, -1:])
        return np.column_stack((orden.mean(axis=1), np.median(orden, axis=1),
                                centro.mean(axis=1), limitada.mean(axis=1)))

    inferior, superior = np.quantile(estimar(replicas), [.025, .975], axis=0)
    return pd.DataFrame({
        "estimador": ["Media", "Mediana", "Recortada 10 %", "Winsorizada 10 %"],
        "valor": estimar(y[None, :])[0], "IC95_inferior": inferior, "IC95_superior": superior,
    })


def por_carrera(datos):
    """Resume todos los grupos y compara de forma exploratoria grupos con n ≥ 5."""
    tabla = datos.groupby("carrera").prediccion.agg(
        n="size", media="mean", mediana="median", desviacion="std",
        q1=lambda y: y.quantile(.25), q3=lambda y: y.quantile(.75),
    ).sort_values("n", ascending=False)
    grupos = {nombre: grupo.prediccion.to_numpy()
              for nombre, grupo in datos.groupby("carrera") if len(grupo) >= 5}
    diagnostico = pd.DataFrame([
        {"carrera": nombre, "n": len(y), "Shapiro_W": stats.shapiro(y).statistic,
         "Shapiro_p": stats.shapiro(y).pvalue,
         "asimetria": stats.skew(y, bias=False)}
        for nombre, y in grupos.items()
    ])
    pruebas = []
    if len(grupos) >= 2:
        for nombre, prueba in [("Levene centrado en mediana", stats.levene),
                               ("Kruskal–Wallis", stats.kruskal)]:
            resultado = prueba(*grupos.values())
            pruebas.append({"prueba": nombre, "estadistico": resultado.statistic,
                            "p": resultado.pvalue, "n": sum(map(len, grupos.values())),
                            "grupos": " | ".join(grupos)})
    return tabla, diagnostico, pd.DataFrame(pruebas)
