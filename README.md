# Estimación colectiva de M&Ms

Se estudiaron las respuestas registradas para un solo frasco. El objetivo estadístico fue caracterizar la percepción colectiva. El objetivo predictivo fue medir si la carrera ayuda a anticipar una nueva estimación humana en condiciones similares.

## Diagnóstico de los datos

El CSV original `M&Ms - Sheet1.csv` contiene 92 filas y 14 etiquetas de carrera o grupo. Las variables analíticas son `carrera` (categórica nominal) y `prediccion` (entero positivo en M&Ms). Ambas están completas. Se descartaron del archivo procesado una columna vacía y `Promedio`, cuyo único valor coincide con la media calculada. Se retiró un espacio final de una etiqueta y se preservaron las siglas y etiquetas distintas porque su equivalencia no está documentada. Hay 10 pares distintos de carrera y estimación que se repiten, distribuidos en 26 filas. Todas las respuestas permanecen en el análisis.

La inspección inicial encontró el CSV y cuatro directorios vacíos. El conteo físico no está documentado en el proyecto. La [auditoría](resultados/auditoria.json) registra las decisiones y la huella SHA-256 del original.

## Metodología estadística

Se calcularon estadísticas descriptivas de las estimaciones y de cada carrera. Las cercas de Tukey identificaron valores extremos para el análisis de sensibilidad. Se calcularon intervalos percentiles del 95 % para cuatro estimadores mediante 20 000 remuestreos bootstrap de filas. El contraste exploratorio de Kruskal–Wallis se aplicó a grupos con al menos cinco respuestas. Los diagnósticos de normalidad y dispersión se documentan en [las tablas de resultados](resultados/).

## Resultados estadísticos

La mediana y la moda fueron **300 M&Ms**. La moda aparece nueve veces. La mitad central de las respuestas quedó entre **197,5 y 500 M&Ms** y el recorrido total fue de 5 a 3 200. La desviación estándar fue de 499,04 M&Ms, con un coeficiente de variación de 111,26 %. La asimetría corregida de 3,67 confirma una cola derecha marcada.

| Estimador | M&Ms | IC bootstrap del 95 % |
| --- | ---: | ---: |
| Media | 448,54 | 357,09–560,21 |
| Mediana | 300,00 | 250,00–350,00 |
| Media recortada 10 % por cola | 352,43 | 296,54–422,45 |
| Media winsorizada 10 % por cola | 378,76 | 317,67–453,70 |

Al retirar únicamente el máximo en un análisis de sensibilidad, la media bajó a 418,31. Al restringir el cálculo a las 84 respuestas dentro de las cercas de Tukey, la media bajó a 331,38. La mediana fue 300 en los tres escenarios. Para describir la respuesta típica, 300 es la síntesis más robusta de las examinadas en estos escenarios. Los intervalos caracterizan la variabilidad de cada estimador bajo el esquema de remuestreo, con las limitaciones de independencia y representatividad del estudio.

Las medianas de los cuatro grupos comparables descriptivamente abarcaron de 262,5 en Actuaría a 555 en Derecho. Kruskal–Wallis incluyó 73 observaciones de Actuaría, Comunicación, Derecho y MAC, con $H=7,19$ y $p=0,0662$. La evidencia exploratoria fue insuficiente para distinguir sus distribuciones al nivel del 5 %. Shapiro–Wilk detectó desviación de la normalidad en Actuaría y MAC. Levene dio $p=0,7462$, con poca capacidad para descartar diferencias de dispersión. El contraste supone independencia y cubre cuatro de las 14 etiquetas.

## Bosque aleatorio y resultados predictivos

El bosque usó exclusivamente `carrera` y aprendió a predecir `prediccion`, la respuesta humana. Se aplicó codificación one-hot dentro de cada pipeline de scikit-learn. La búsqueda interna seleccionó la profundidad máxima y el tamaño mínimo de hoja por MAE. El esquema anidado separó la selección de hiperparámetros (cuatro pliegues internos) de la evaluación (cinco pliegues externos por repetición, cinco repeticiones). Todas las métricas siguientes son medias de las cinco repeticiones.

| Modelo | MAE (M&Ms) | RMSE (M&Ms) | R² |
| --- | ---: | ---: | ---: |
| Dummy media | 298,91 | 501,98 | −0,023 |
| Dummy mediana | **261,37** | 517,96 | −0,089 |
| Bosque aleatorio | 311,89 | 532,69 | −0,152 |

El bosque obtuvo resultados promedio peores que ambos modelos base en las tres métricas. Frente a la mediana de entrenamiento, su MAE empeoró entre 44,07 y 58,53 M&Ms en las cinco repeticiones. Obtuvo un MAE medio de pliegue de 279,82 en entrenamiento y 311,38 en evaluación. La brecha de 31,56 M&Ms es compatible con sobreajuste. Se seleccionó profundidad máxima de uno en 23 de los 25 pliegues externos. En la primera repetición, el bosque predijo aproximadamente 383 M&Ms para la respuesta de 3 200, lo que dejó un residuo de aproximadamente 2 817 M&Ms. Estos resultados indican que la carrera como única entrada aporta escaso valor predictivo en el esquema estudiado. La [comparación pareada](resultados/comparacion_pareada_mae.csv) y las [métricas por repetición](resultados/metricas_repeticiones.csv) documentan la evaluación.

## Implicaciones para el cartel

- Presentar 300 M&Ms como la respuesta colectiva típica e incluir su intervalo bootstrap y el rango intercuartílico. La [distribución](graficas/01_distribucion.pdf) muestra por qué la media es sensible a la cola superior.
- Mostrar los tamaños de grupo junto a sus estimaciones. Actuaría concentra 40 de las 92 filas y seis categorías tienen una sola respuesta.
- Exponer el resultado predictivo con sus dos referencias. El bosque evaluó respuestas humanas y su desempeño depende de cómo se divide esta muestra. [Comparación de métricas](graficas/03_metricas.pdf)
- Hacer explícitos el único frasco, el conteo desconocido, la selección de participantes y la independencia sin verificar. Documentar el conteo físico permitiría medir el error de cada agregado. Un modelo de conteo real exigiría datos de varios frascos con sus respectivos conteos.

## Limitaciones metodológicas

Se estudió un solo frasco sin un conteo físico documentado. La independencia de las respuestas y la representatividad del reclutamiento siguen sin verificarse. El desequilibrio entre carreras limita el contraste estadístico y la interpretación del rendimiento para grupos pequeños. La carrera es la única entrada del modelo y sus métricas describen predicción de respuestas humanas en esta muestra.

## Reproducción

La ejecución completa requiere una copia autorizada del archivo `M&Ms - Sheet1.csv` en la raíz. Una de sus etiquetas incluye un nombre propio. Tanto ese archivo como los derivados que conservan la etiqueta se mantienen localmente. La copia versionable del notebook contiene el código y la discusión y tiene las salidas de ejecución retiradas. Si solo dispone de esa copia, restaure primero el nombre esperado por el ejecutor.

```bash
cp -n 01_exploracion_y_modelado_publico.ipynb 01_exploracion_y_modelado.ipynb
```

Desde la raíz del proyecto, con Python 3.11 y el CSV disponible, ejecute estos comandos.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s pruebas -v
python reproducir.py
```

El script ejecuta todas las celdas mediante IPython dentro del mismo proceso y guarda las salidas en el notebook. Se usa la semilla 1837 para remuestreo y como base de las semillas de validación. La ejecución regenera los archivos derivados. Los resultados del bosque se calcularon con scikit-learn 1.8.0.

## Estructura del proyecto

| Entregable | Ruta |
| --- | --- |
| Notebook con análisis y referencias | [01_exploracion_y_modelado_publico.ipynb](01_exploracion_y_modelado_publico.ipynb) |
| Código reutilizable | [src/](src/) |
| Ejecutar y guardar salidas del notebook local | [reproducir.py](reproducir.py) |
| Dependencias de Python | [requirements.txt](requirements.txt) |
| Tablas de estadística y evaluación sin etiquetas personales | [resultados/](resultados/) |
| Figuras sin etiquetas personales, en PNG de 300 ppp y PDF vectorial | [graficas/](graficas/) |
| Hiperparámetros finales | [modelos/hiperparametros_finales.json](modelos/hiperparametros_finales.json) |
| Pruebas de funcionamiento | [pruebas/](pruebas/) |

La ejecución local genera también `datos/procesados/estimaciones.csv`, las tablas por carrera y de predicciones individuales, la figura por carrera y el bosque serializado. El [archivo de exclusiones](.gitignore) detalla qué artefactos requieren acceso local.

## Referencias

- Galton, F. (1907). Vox Populi. *Nature*, 75, 450–451. [DOI](https://doi.org/10.1038/075450a0)
- Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife. *The Annals of Statistics*, 7(1), 1–26. [DOI](https://doi.org/10.1214/aos/1176344552)
- Kruskal, W. H. y Wallis, W. A. (1952). Use of Ranks in One-Criterion Variance Analysis. *Journal of the American Statistical Association*, 47(260), 583–621. [DOI](https://doi.org/10.1080/01621459.1952.10483441)
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32. [DOI](https://doi.org/10.1023/A:1010933404324)
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M. y Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12(85), 2825–2830. [Artículo](https://www.jmlr.org/papers/v12/pedregosa11a.html)
- Proyecto scikit-learn. (s. f.). Documentación de scikit-learn 1.8. [RandomForestRegressor](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestRegressor.html), [DummyRegressor](https://scikit-learn.org/1.8/modules/generated/sklearn.dummy.DummyRegressor.html) y [métricas de regresión](https://scikit-learn.org/1.8/modules/model_evaluation.html#regression-metrics).
- Proyecto SciPy. (s. f.). *Statistical functions*. [Documentación de scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html).
