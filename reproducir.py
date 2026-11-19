"""Ejecuta las celdas en IPython y guarda sus salidas en el notebook."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat

raiz = Path(__file__).resolve().parent
ruta = raiz / "01_exploracion_y_modelado.ipynb"


def ejecutar():
    """Procesa todas las celdas en orden y solo guarda el notebook si finalizan."""
    notebook = nbformat.read(ruta, as_version=4)
    os.chdir(raiz)
    with TemporaryDirectory(prefix="mms-") as temporal:
        os.environ.setdefault("MPLCONFIGDIR", temporal)
        os.environ.setdefault("IPYTHONDIR", temporal)
        from IPython.core.interactiveshell import InteractiveShell
        from IPython.utils.capture import capture_output
        from IPython.core.pylabtools import activate_matplotlib
        from matplotlib_inline.backend_inline import configure_inline_support
        from traitlets.config import Config

        shell = InteractiveShell.instance(config=Config({"HistoryManager": {"enabled": False}}))
        backend = "module://matplotlib_inline.backend_inline"
        activate_matplotlib(backend)
        configure_inline_support(shell, backend)
        contador = 0
        for celda in notebook.cells:
            if celda.cell_type != "code":
                continue
            contador += 1
            with capture_output() as captura:
                resultado = shell.run_cell(celda.source, store_history=True)
            if not resultado.success:
                print(captura.stdout, captura.stderr)
                resultado.raise_error()
            celda.execution_count = contador
            celda.outputs = []
            for nombre in ("stdout", "stderr"):
                contenido = getattr(captura, nombre)
                if contenido:
                    celda.outputs.append(nbformat.v4.new_output(
                        "stream", name=nombre, text=contenido))
            for salida in captura.outputs:
                celda.outputs.append(nbformat.v4.new_output(
                    "display_data", data=salida.data, metadata=salida.metadata))
            print(f"Celda {contador} ejecutada")
    nbformat.validate(notebook)
    nbformat.write(notebook, ruta)


if __name__ == "__main__":
    ejecutar()
