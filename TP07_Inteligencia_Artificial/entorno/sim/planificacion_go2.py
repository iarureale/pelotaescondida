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
