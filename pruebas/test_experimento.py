"""Comprobaciones de integridad, remuestreo y evaluación."""
import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.analisis import ORIGINAL, bootstrap, cargar
from src.modelado import evaluar


class Experimento(unittest.TestCase):
    def test_original_y_limpieza(self):
        self.assertEqual(hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
                         "cb8cba971e6df0f48217c43219ca81395a6746c57389d4c4998cfc666f363b6d")
        datos, auditoria = cargar()
        self.assertEqual(len(datos), 92)
        self.assertEqual(datos.carrera.nunique(), 14)
        self.assertEqual(auditoria["repeticiones_adicionales"], 16)
        self.assertAlmostEqual(datos.prediccion.mean(), 448.5434783)

    def test_rechaza_conteo_invalido(self):
        with TemporaryDirectory() as directorio:
            ruta = Path(directorio) / "invalido.csv"
            ruta.write_text("carrera,prediccion,,Promedio\nMAC,0,,\n")
            with self.assertRaises(ValueError):
                cargar(ruta)

    def test_bootstrap_reproducible(self):
        datos, _ = cargar()
        estimadores = bootstrap(datos.prediccion, 300)
        assert_frame_equal(estimadores, bootstrap(datos.prediccion, 300))
        self.assertTrue((estimadores.IC95_inferior < estimadores.IC95_superior).all())

    def test_predicciones_externas(self):
        datos, _ = cargar()
        pred, pliegues, metricas, grilla = evaluar(datos, repeticiones=1)
        self.assertEqual(len(pliegues), 15)
        self.assertEqual(len(metricas), 3)
        self.assertEqual(grilla.seleccionado.sum(), 5)
        self.assertTrue(pred.groupby(["modelo", "fila_csv"]).size().eq(1).all())
        self.assertEqual(len(pred), 3 * len(datos))
        np.testing.assert_allclose(pred.residuo, pred.prediccion - pred.predicha)
        unicas = datos.carrera.value_counts().loc[lambda n: n.eq(1)].index
        self.assertFalse(pred.loc[pred.carrera.isin(unicas), "carrera_vista"].any())
        dummies = pred.loc[pred.modelo.str.startswith("Dummy")]
        for (_, nombre), grupo in dummies.groupby(["pliegue", "modelo"]):
            entrenamiento = datos.loc[~datos.fila_csv.isin(grupo.fila_csv), "prediccion"]
            esperado = entrenamiento.median() if nombre == "Dummy mediana" else entrenamiento.mean()
            np.testing.assert_allclose(grupo.predicha, esperado)
