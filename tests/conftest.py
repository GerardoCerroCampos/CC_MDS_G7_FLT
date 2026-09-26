"""Registra automáticamente las pruebas en docs/api/pruebas/.

Pytest descubre este archivo al ejecutar tests/test_api.py. No se importa desde
las pruebas y no inicia la API de localhost: las pruebas usan TestClient.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
from uuid import uuid4

import pytest

BASE = Path(__file__).resolve().parents[1]


def pytest_configure(config):
    # Solo asigna una ruta nueva si el usuario no indicó --basetemp.
    # pytest puede vaciar basetemp: nunca se elige una carpeta de datos existente.
    if not config.option.basetemp:
        config.option.basetemp = str(BASE / ('.pytest_tmp_' + uuid4().hex))
    config._registro_api = {
        'inicio': datetime.now(timezone.utc).isoformat(),
        'reloj': time.perf_counter(),
        'fases': [],
        'errores_coleccion': [],
    }


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    resultado = yield
    reporte = resultado.get_result()
    registro = item.config._registro_api
    fase = {'prueba': reporte.nodeid, 'fase': reporte.when,
            'resultado': reporte.outcome, 'duracion_segundos': reporte.duration}
    if reporte.failed or reporte.skipped:
        fase['detalle'] = reporte.longreprtext
    registro['fases'].append(fase)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    registro = config._registro_api
    fallos = terminalreporter.stats.get('error', [])
    registro['errores_coleccion'] = [r.longreprtext for r in fallos if getattr(r, 'when', None) == 'collect']
    conteos = {nombre: len(terminalreporter.stats.get(nombre, []))
               for nombre in ['passed', 'failed', 'error', 'skipped', 'xfailed', 'xpassed', 'warnings']}
    avisos = [str(getattr(r, 'message', r)) for r in terminalreporter.stats.get('warnings', [])]
    duracion = time.perf_counter() - registro.pop('reloj')
    registro.update({
        'fin': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'pytest': pytest.__version__,
        'argumentos_pytest': list(config.invocation_params.args),
        'codigo_salida': int(exitstatus), 'correcto': int(exitstatus) == 0,
        'duracion_segundos': round(duracion, 3), 'resumen': conteos, 'avisos': avisos,
    })
    nombre = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    carpeta = BASE / 'docs/api/pruebas' / nombre
    texto = [
        'REGISTRO AUTOMÁTICO DE PYTEST', '',
        'Inicio UTC: ' + registro['inicio'],
        'Python: ' + registro['python'] + ' | pytest: ' + pytest.__version__,
        'Argumentos: ' + ' '.join(registro['argumentos_pytest']),
        'Código de salida: ' + str(int(exitstatus)),
        'Resultado: ' + ('CORRECTO' if registro['correcto'] else 'REQUIERE REVISIÓN'),
        'Resumen: ' + ', '.join(f'{v} {k}' for k, v in conteos.items() if v),
        f'Duración registrada: {duracion:.3f} segundos', '', 'DETALLE POR FASE', '',
    ]
    for fase in registro['fases']:
        texto.append(f"{fase['prueba']} | {fase['fase']} | {fase['resultado']} | {fase['duracion_segundos']:.4f} s")
        if fase.get('detalle'):
            texto.append(fase['detalle'])
    if registro['errores_coleccion']:
        texto += ['', 'ERRORES DE COLECCIÓN', *registro['errores_coleccion']]
    if avisos:
        texto += ['', 'AVISOS', *avisos]
    try:
        carpeta.mkdir(parents=True, exist_ok=False)
        (carpeta / 'resultado.json').write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding='utf-8')
        (carpeta / 'resultado.txt').write_text('\n'.join(texto) + '\n', encoding='utf-8')
        terminalreporter.write_sep('-', 'Registro guardado en ' + str(carpeta))
    except OSError as error:
        terminalreporter.write_line('ERROR: no se pudo guardar la evidencia: ' + str(error), red=True)
