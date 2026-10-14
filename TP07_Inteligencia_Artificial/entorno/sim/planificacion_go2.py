"""Planificador A* para navegación Go2 sobre el mapa 2D del simulador UADE."""
from __future__ import annotations

import heapq
import math

from .grilla import LIBRE


class SinRuta(ValueError):
    """El destino no es alcanzable con el mapa y margen indicados."""


def astar(mapa, inicio=None, destino=None, margen_m: float = 0.0):
    """Busca una ruta de cuatro vecinos y expande obstáculos por ``margen_m``.

    El margen representa el radio de seguridad del robot más una holgura. La
    expansión considera la superficie cuadrada de cada celda ocupada, no solo
    la distancia entre centros, para no dejar pasos diagonales estrechos.
    """
    inicio = tuple(inicio or mapa.inicio)
    destino = tuple(destino or mapa.destino)
    if margen_m < 0:
        raise ValueError("El margen no puede ser negativo.")
    if not mapa.es_transitable(*inicio) or not mapa.es_transitable(*destino):
        raise SinRuta("Inicio o destino no transitable.")

    celda = mapa.tamano_celda
    alcance = int(math.ceil(margen_m / celda + 0.5))
    bloqueadas = set()
    for f in range(mapa.filas):
        for c in range(mapa.columnas):
            if mapa.celda(f, c) == LIBRE:
                continue
            for df in range(-alcance, alcance + 1):
                for dc in range(-alcance, alcance + 1):
                    dx = max(0.0, abs(df) * celda - celda / 2)
                    dy = max(0.0, abs(dc) * celda - celda / 2)
                    if math.hypot(dx, dy) <= margen_m + 1e-9:
                        bloqueadas.add((f + df, c + dc))
    if inicio in bloqueadas or destino in bloqueadas:
        raise SinRuta("El inicio o el destino no respeta el margen de seguridad.")

    def heuristica(p):
        return abs(p[0] - destino[0]) + abs(p[1] - destino[1])

    abiertos = [(heuristica(inicio), 0, inicio)]
    previo = {}
    costo = {inicio: 0}
    cerrados = set()
    while abiertos:
        _, g, actual = heapq.heappop(abiertos)
        if actual in cerrados:
            continue
        if actual == destino:
            ruta = [actual]
            while ruta[-1] in previo:
                ruta.append(previo[ruta[-1]])
            return list(reversed(ruta))
        cerrados.add(actual)
        f, c = actual
        for df, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            vecino = (f + df, c + dc)
            if (vecino in cerrados or vecino in bloqueadas
                    or not mapa.es_transitable(*vecino)):
                continue
            nuevo = g + 1
            if nuevo < costo.get(vecino, math.inf):
                costo[vecino] = nuevo
                previo[vecino] = actual
                heapq.heappush(abiertos, (nuevo + heuristica(vecino), nuevo, vecino))
    raise SinRuta(f"No hay ruta entre {inicio} y {destino}.")


def _despeje_m(mapa, f: float, c: float) -> float:
    """Distancia (m) del punto (fila, columna fraccionarias) al obstáculo más cercano.

    Mide contra la superficie cuadrada de cada celda ocupada. Fuera de la grilla
    cuenta como obstáculo para que un atajo no pueda salirse del mapa.
    """
    celda = mapa.tamano_celda
    if not (-0.5 <= f <= mapa.filas - 0.5 and -0.5 <= c <= mapa.columnas - 0.5):
        return 0.0
    mejor = math.inf
    for ff in range(max(0, int(f) - 3), min(mapa.filas, int(f) + 5)):
        for cc in range(max(0, int(c) - 3), min(mapa.columnas, int(c) + 5)):
            if mapa.celda(ff, cc) == LIBRE:
                continue
            dx = max(0.0, abs(f - ff) - 0.5) * celda
            dy = max(0.0, abs(c - cc) - 0.5) * celda
            mejor = min(mejor, math.hypot(dx, dy))
    return mejor


def _segmento_libre(mapa, a, b, margen_m: float, paso_celdas: float = 0.1) -> bool:
    largo = math.hypot(b[0] - a[0], b[1] - a[1])
    n = max(1, int(math.ceil(largo / paso_celdas)))
    for i in range(n + 1):
        t = i / n
        if _despeje_m(mapa, a[0] + (b[0] - a[0]) * t,
                      a[1] + (b[1] - a[1]) * t) < margen_m - 1e-6:
            return False
    return True


def suavizar_ruta(mapa, ruta, margen_m: float):
    """Reemplaza la escalera de A* por tramos rectos con atajos seguros.

    Recorre la ruta y, desde cada vértice, salta al punto más lejano al que se
    llega en línea recta manteniendo ``margen_m`` de despeje continuo contra
    todos los obstáculos (la misma holgura que usa A*, pero medida a lo largo
    del segmento y no solo en los centros de celda). Devuelve los vértices
    (celdas); el primero y el último son los de la ruta original. Menos giros
    de 90° y menos distancia, sin acercarse más a los obstáculos que A*.
    """
    ruta = [tuple(p) for p in ruta]
    if len(ruta) <= 2:
        return ruta
    vertices = [ruta[0]]
    i = 0
    while i < len(ruta) - 1:
        j = len(ruta) - 1
        while j > i + 1 and not _segmento_libre(mapa, ruta[i], ruta[j], margen_m):
            j -= 1
        vertices.append(ruta[j])
        i = j
    return vertices


def ruta_densa(vertices):
    """Celdas que atraviesan los tramos rectos (para dibujar y para consultas)."""
    vertices = [tuple(v) for v in vertices]
    celdas = [vertices[0]]
    for a, b in zip(vertices, vertices[1:]):
        n = max(1, int(math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) * 2)))
        for k in range(1, n + 1):
            p = (round(a[0] + (b[0] - a[0]) * k / n),
                 round(a[1] + (b[1] - a[1]) * k / n))
            if p != celdas[-1]:
                celdas.append(p)
    return celdas
