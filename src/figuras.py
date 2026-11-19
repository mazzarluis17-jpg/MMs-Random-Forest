"""Figuras reproducibles para el informe y el cartel."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from src.analisis import RAIZ, SEMILLA


def exportar(datos, grupos, estimadores, predicciones, metricas):
    """Guarda las figuras en PNG de 300 ppp y PDF vectorial."""
    destino = RAIZ / "graficas"
    destino.mkdir(exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.dpi": 110})

    def guardar(fig, nombre):
        for extension in ("png", "pdf"):
            fig.savefig(destino / f"{nombre}.{extension}", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    y = datos.prediccion
    malla = np.linspace(y.min(), y.max(), 500)
    fig, (ax, caja) = plt.subplots(2, 1, sharex=True, figsize=(9, 5),
                                   height_ratios=[4, 1], layout="constrained")
    ax.hist(y, bins="fd", density=True, color="#a1c9cc", label="Histograma")
    ax.plot(malla, gaussian_kde(y)(malla), color="#134d55", label="Densidad suavizada")
    for nombre, color in [("Media", "#a34e00"), ("Mediana", "#634598")]:
        valor = estimadores.set_index("estimador").loc[nombre, "valor"]
        ax.axvline(valor, color=color, linestyle="--", label=f"{nombre} ({valor:.0f})")
    ax.set(title="Distribución de las estimaciones del frasco", ylabel="Densidad (1/M&M)")
    ax.legend(frameon=False)
    caja.boxplot(y, orientation="horizontal")
    caja.set(xlabel="Estimación (M&Ms)", yticks=[])
    guardar(fig, "01_distribucion")

    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    rng = np.random.default_rng(SEMILLA)
    for i, (carrera, resumen) in enumerate(grupos.iterrows()):
        valores = datos.loc[datos.carrera.eq(carrera), "prediccion"]
        ax.scatter(valores, i + rng.uniform(-.15, .15, len(valores)),
                   color="#256a73", alpha=.64, s=18)
        ax.plot([resumen.q1, resumen.q3], [i, i], color="#134d55", lw=2)
        ax.scatter(resumen.mediana, i, marker="D", color="#a34e00", s=28,
                   label="Mediana" if i == 0 else None)
    ax.set(xscale="log", xlabel="Estimación (M&Ms, escala logarítmica)",
           title="Estimaciones por carrera o grupo",
           yticks=range(len(grupos)),
           yticklabels=[f"{c} (n={n})" for c, n in grupos.n.items()])
    ax.invert_yaxis()
    ax.legend(frameon=False)
    guardar(fig, "02_carreras")

    fig, ejes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, layout="constrained")
    orden = ["Dummy media", "Dummy mediana", "Bosque aleatorio"]
    for ax, metrica in zip(ejes, ["MAE", "RMSE", "R2"]):
        for i, nombre in enumerate(orden):
            valores = metricas.loc[metricas.modelo.eq(nombre), metrica]
            ax.scatter(valores, np.full(len(valores), i), color="#256a73", alpha=.55)
            ax.scatter(valores.mean(), i, color="#a34e00", marker="D",
                       label="Media de las repeticiones" if i == 0 else None)
        ax.set(yticks=range(len(orden)), yticklabels=orden,
               xlabel="R² (sin unidades)" if metrica == "R2" else f"{metrica} (M&Ms)")
    ejes[0].invert_yaxis()
    ejes[-1].legend(frameon=False, loc="best", fontsize=8)
    fig.suptitle("Rendimiento predictivo fuera de muestra")
    guardar(fig, "03_metricas")

    bosque = predicciones.query("modelo == 'Bosque aleatorio' and repeticion == 1")
    fig, (ax, residuos) = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    limite = max(bosque.prediccion.max(), bosque.predicha.max())
    ax.plot([0, limite], [0, limite], linestyle="--", color="#999999", label="Igualdad")
    for vista, g in bosque.groupby("carrera_vista"):
        color = "#256a73" if vista else "#a34e00"
        etiqueta = "Carrera vista" if vista else "Carrera ausente en entrenamiento"
        ax.scatter(g.predicha, g.prediccion, alpha=.65, color=color, label=etiqueta)
        residuos.scatter(g.predicha, g.residuo, alpha=.65, color=color)
    ax.set(title="Observado frente a predicho", xlabel="Predicción (M&Ms)",
           ylabel="Estimación humana (M&Ms)")
    ax.legend(frameon=False, fontsize=8)
    residuos.axhline(0, linestyle="--", color="#999999")
    residuos.set(title="Residuos del bosque", xlabel="Predicción (M&Ms)",
                 ylabel="Observado menos predicho (M&Ms)")
    fig.suptitle("Bosque aleatorio en la primera repetición externa")
    guardar(fig, "04_diagnostico_bosque")
