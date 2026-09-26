"""Desde la raíz: python scripts/registrar_pruebas.py

La API debe estar iniciada en otra terminal. Registra pytest y llamadas reales
sin reemplazar evidencias anteriores. Solo utiliza la biblioteca estándar.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4


def llamar(url, entrada):
    peticion = Request(url, data=json.dumps(entrada).encode('utf-8'),
                       headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urlopen(peticion, timeout=30) as respuesta:
            estado, texto = respuesta.status, respuesta.read().decode('utf-8')
    except HTTPError as error:
        estado, texto = error.code, error.read().decode('utf-8', errors='replace')
    except (URLError, TimeoutError, OSError) as error:
        return None, {'error_conexion': str(error)}
    try:
        return estado, json.loads(texto)
    except ValueError:
        return estado, {'texto': texto}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://localhost:8000')
    args = parser.parse_args()
    base = Path(__file__).resolve().parent.parent
    identificador = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    carpeta = base / 'docs/api/ejecuciones' / identificador
    carpeta.mkdir(parents=True, exist_ok=False)
    # Ruta nueva y dedicada: evita el conflicto de permisos del TEMP de Windows.
    temporal = base / ('.pytest_tmp_' + uuid4().hex)
    comando = [sys.executable, '-m', 'pytest', '-q', 'tests/test_api.py',
               '--basetemp=' + str(temporal)]
    print('Ejecutando pytest...', flush=True)
    try:
        pruebas = subprocess.run(comando, cwd=base, capture_output=True,
                                 text=True, encoding='utf-8', errors='replace', timeout=300)
        codigo = pruebas.returncode
        texto_pytest = pruebas.stdout + '\n' + pruebas.stderr
    except subprocess.TimeoutExpired as error:
        codigo = -1
        salida = error.stdout or b''
        texto_pytest = salida.decode('utf-8', errors='replace') if isinstance(salida, bytes) else salida
        texto_pytest += '\nPytest superó el tiempo máximo de 300 segundos.'
    except OSError as error:
        codigo, texto_pytest = -1, str(error)
    (carpeta / 'pytest.txt').write_text(texto_pytest, encoding='utf-8')

    vuelo = {'AIRLINE': 'AA', 'ORIGIN_AIRPORT': 'LAX', 'DESTINATION_AIRPORT': 'JFK',
             'MONTH': 7, 'DAY': 15, 'DAY_OF_WEEK': 3, 'SCHEDULED_DEPARTURE_MIN': 1080,
             'SCHEDULED_TIME': 320.0, 'DISTANCE': 2475.0}
    casos = [('Individual', '/predict', vuelo, 200),
             ('Lote', '/predict-batch', [vuelo, dict(vuelo, AIRLINE='DL')], 200),
             ('Entrada inválida', '/predict', dict(vuelo, MONTH=13), 422)]
    llamadas = []
    for nombre, ruta, entrada, esperado in casos:
        url = args.url.rstrip('/') + ruta
        estado, respuesta = llamar(url, entrada)
        correcto = estado == esperado
        if nombre == 'Lote':
            correcto = correcto and isinstance(respuesta, list) and len(respuesta) == len(entrada)
        llamadas.append({'caso': nombre, 'fecha_utc': datetime.now(timezone.utc).isoformat(),
                         'metodo': 'POST', 'url': url, 'solicitud': entrada,
                         'http_esperado': esperado, 'http_obtenido': estado,
                         'correcto': correcto, 'respuesta': respuesta})
        print(f'{nombre}: HTTP {estado}, ' + ('correcto' if correcto else 'revisar'), flush=True)
    (carpeta / 'llamadas.json').write_text(json.dumps(llamadas, indent=2, ensure_ascii=False), encoding='utf-8')
    exito = codigo == 0 and all(c['correcto'] for c in llamadas)
    resumen = {'ejecucion': identificador, 'python_pruebas': platform.python_version(),
               'interprete_pruebas': sys.executable, 'api_url': args.url,
               'pytest_comando': comando, 'pytest_exit_code': codigo,
               'llamadas_correctas': sum(c['correcto'] for c in llamadas), 'correcto': exito}
    (carpeta / 'resumen.json').write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding='utf-8')
    md = ['# Evidencia automatizada de API', '', f'Ejecución UTC: {identificador}', '',
          'Resultado: **' + ('CORRECTO' if exito else 'REQUIERE REVISIÓN') + '**', '',
          '## Pytest', '', f'Python: {platform.python_version()}. Código de salida: {codigo}.', '',
          '```text', texto_pytest.strip(), '```', '', '## Llamadas HTTP reales', '']
    for c in llamadas:
        md += ['### ' + c['caso'], '', '`POST ' + c['url'] + '`', '',
               f"HTTP esperado: {c['http_esperado']}. Obtenido: {c['http_obtenido']}.", '',
               'Solicitud:', '', '```json', json.dumps(c['solicitud'], indent=2, ensure_ascii=False),
               '```', '', 'Respuesta:', '', '```json', json.dumps(c['respuesta'], indent=2, ensure_ascii=False), '```', '']
    md += ['## Alcance', '', 'Pytest importa la API del repositorio; las llamadas HTTP consultan el servicio indicado por --url. '
           'Inicia ese servicio desde este mismo proyecto para que ambas comprobaciones correspondan a tu entrega. '
           'Este registro no toma una captura de pantalla de Swagger; esa evidencia se guarda por separado.']
    (carpeta / 'reporte.md').write_text('\n'.join(md), encoding='utf-8')
    print(f'Reporte guardado en: {carpeta / "reporte.md"}')
    print('Agrega .pytest_tmp_*/ a .gitignore; las carpetas temporales no son evidencia.')
    return 0 if exito else 1


if __name__ == '__main__':
    raise SystemExit(main())
