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


# =============================================================================
# Planificador de ángulo libre (Theta*) sobre un campo de despeje continuo
# =============================================================================
# A* de 4 vecinos devuelve escaleras de ángulos rectos; recortarlas después no
# recupera la ruta óptima (queda 3-15 % más larga) y siempre parte del centro de
# la celda. Aquí la ruta se busca directamente en el plano, con vértices
# libres (cualquier ángulo) y la misma holgura de seguridad que usa el robot.
RES_CAMPO_M = 0.10            # resolución del campo de despeje
RANGO_CAMPO_M = 1.5           # despeje máximo que se calcula (más allá, "libre")
PASO_RETICULA_M = 0.50        # separación de los nodos de búsqueda
SEGURIDAD_MUESTREO_M = 0.04   # compensa el redondeo del campo al muestrear
ZONA_EXTREMO_M = 0.70         # cerca del inicio/destino se admite menos despeje
# Preferencia por zonas anchas: entre el margen de seguridad (duro) y margen +
# holgura, cada metro cuesta hasta (1 + PENALIZACION_ESTRECHO) metros. Un paso
# estrecho se usa si el rodeo es mayor que ese sobrecosto; nunca por debajo del
# margen duro.
PENALIZACION_ESTRECHO = 0.6
_CACHE_CAMPOS: dict = {}


class CampoDespeje:
    """Distancia al obstáculo más cercano en una grilla fina (numpy).

    Se calcula una vez por mapa (y por conjunto de celdas ocupadas) y permite
    comprobar un segmento entero con una sola consulta vectorizada.
    """

    def __init__(self, mapa):
        import numpy as np
        self.np = np
        self.celda = mapa.tamano_celda
        self.filas, self.columnas = mapa.filas, mapa.columnas
        alto = int(round(self.filas * self.celda / RES_CAMPO_M))
        ancho = int(round(self.columnas * self.celda / RES_CAMPO_M))
        campo = np.full((alto, ancho), RANGO_CAMPO_M, dtype=np.float32)
        # Coordenadas (en celdas) del centro de cada muestra.
        f = (np.arange(alto) + 0.5) * RES_CAMPO_M / self.celda - 0.5
        c = (np.arange(ancho) + 0.5) * RES_CAMPO_M / self.celda - 0.5
        # El borde del mapa cuenta como obstáculo (distancia al borde de la grilla).
        borde_f = np.minimum(f + 0.5, self.filas - 0.5 - f) * self.celda
        borde_c = np.minimum(c + 0.5, self.columnas - 0.5 - c) * self.celda
        campo = np.minimum(campo, np.minimum(borde_f[:, None], borde_c[None, :]))
        radio = int(math.ceil(RANGO_CAMPO_M / RES_CAMPO_M)) + int(math.ceil(self.celda / RES_CAMPO_M))
        for ff in range(self.filas):
            for cc in range(self.columnas):
                if mapa.celda(ff, cc) == LIBRE:
                    continue
                i0 = max(0, int((ff + 0.5) * self.celda / RES_CAMPO_M) - radio)
                i1 = min(alto, int((ff + 0.5) * self.celda / RES_CAMPO_M) + radio + 1)
                j0 = max(0, int((cc + 0.5) * self.celda / RES_CAMPO_M) - radio)
                j1 = min(ancho, int((cc + 0.5) * self.celda / RES_CAMPO_M) + radio + 1)
                dx = np.maximum(0.0, np.abs(f[i0:i1] - ff) - 0.5) * self.celda
                dy = np.maximum(0.0, np.abs(c[j0:j1] - cc) - 0.5) * self.celda
                np.minimum(campo[i0:i1, j0:j1], np.hypot(dx[:, None], dy[None, :]),
                           out=campo[i0:i1, j0:j1])
        self.campo = campo

    def _indices(self, f, c):
        i = ((f + 0.5) * self.celda / RES_CAMPO_M).astype(int)
        j = ((c + 0.5) * self.celda / RES_CAMPO_M).astype(int)
        return i, j

    def en(self, f: float, c: float) -> float:
        i = int((f + 0.5) * self.celda / RES_CAMPO_M)
        j = int((c + 0.5) * self.celda / RES_CAMPO_M)
        if not (0 <= i < self.campo.shape[0] and 0 <= j < self.campo.shape[1]):
            return 0.0
        return float(self.campo[i, j])

    def evaluar(self, a, b, margen_m: float, preferido_m: float,
                tol_a: float | None = None, tol_b: float | None = None):
        """(libre, costo_m) del segmento a-b (celdas).

        ``libre``: despeje >= margen_m en todo el trayecto (más tolerancia cerca de
        los extremos). ``costo_m``: largo en metros ponderado por la estrechez:
        1 + PENALIZACION_ESTRECHO cuando el despeje es el mínimo y 1 desde
        ``preferido_m`` en adelante.
        """
        np = self.np
        largo_m = math.hypot(b[0] - a[0], b[1] - a[1]) * self.celda
        n = max(2, int(math.ceil(largo_m / (RES_CAMPO_M * 0.5))) + 1)
        t = np.linspace(0.0, 1.0, n)
        i, j = self._indices(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        if (i.min() < 0 or j.min() < 0 or i.max() >= self.campo.shape[0]
                or j.max() >= self.campo.shape[1]):
            return False, math.inf
        d = self.campo[i, j].astype(np.float64)
        duro = margen_m + SEGURIDAD_MUESTREO_M
        umbral = np.full(n, duro)
        if tol_a is not None:
            umbral[t * largo_m <= ZONA_EXTREMO_M] = min(duro, tol_a)
        if tol_b is not None:
            cerca = (1 - t) * largo_m <= ZONA_EXTREMO_M
            umbral[cerca] = np.minimum(umbral[cerca], tol_b)
        if not np.all(d >= umbral):
            return False, math.inf
        pref = preferido_m + SEGURIDAD_MUESTREO_M
        ancho = max(1e-6, pref - duro)
        pena = PENALIZACION_ESTRECHO * np.clip((pref - d) / ancho, 0.0, 1.0)
        return True, float(largo_m * (1.0 + pena.mean()))

    def segmento_libre(self, a, b, margen_m: float, tol_a: float | None = None,
                       tol_b: float | None = None) -> bool:
        """¿Se puede ir de a a b (en celdas) con despeje >= margen en todo el trayecto?

        ``tol_a``/``tol_b`` bajan el umbral cerca de los extremos: el robot puede
        estar ya dentro de la zona de margen (p. ej. pegado a un banco) y debe
        poder salir; el resto del trayecto exige el margen completo.
        """
        np = self.np
        largo_m = math.hypot(b[0] - a[0], b[1] - a[1]) * self.celda
        n = max(2, int(math.ceil(largo_m / (RES_CAMPO_M * 0.5))) + 1)
        t = np.linspace(0.0, 1.0, n)
        f = a[0] + (b[0] - a[0]) * t
        c = a[1] + (b[1] - a[1]) * t
        i, j = self._indices(f, c)
        if (i.min() < 0 or j.min() < 0 or i.max() >= self.campo.shape[0]
                or j.max() >= self.campo.shape[1]):
            return False
        d = self.campo[i, j]
        umbral = np.full(n, margen_m + SEGURIDAD_MUESTREO_M, dtype=np.float64)
        if tol_a is not None:
            umbral[t * largo_m <= ZONA_EXTREMO_M] = min(umbral[0], tol_a)
        if tol_b is not None:
            umbral[(1 - t) * largo_m <= ZONA_EXTREMO_M] = np.minimum(
                umbral[(1 - t) * largo_m <= ZONA_EXTREMO_M], tol_b)
        return bool(np.all(d >= umbral))


def campo_de(mapa) -> CampoDespeje:
    clave = (id(mapa), tuple(tuple(f) for f in mapa.grilla))
    campo = _CACHE_CAMPOS.get(clave)
    if campo is None:
        if len(_CACHE_CAMPOS) >= 4:
            _CACHE_CAMPOS.clear()
        campo = _CACHE_CAMPOS[clave] = CampoDespeje(mapa)
    return campo


def _theta_estrella(mapa, campo, inicio, destino, margen_m, preferido_m):
    """Theta* sobre una retícula de ``PASO_RETICULA_M``; inicio/destino exactos.

    Devuelve vértices [(fila, col), ...] en coordenadas de celda fraccionarias o
    None si no hay camino con ese margen.
    """
    import heapq
    from itertools import count
    orden = count()                                     # desempate: nodos 'S'/'G' no son comparables

    celda = mapa.tamano_celda
    paso = PASO_RETICULA_M / celda                      # separación de nodos, en celdas
    ni = int(round((mapa.filas - 1) / paso)) + 1
    nj = int(round((mapa.columnas - 1) / paso)) + 1

    def pos(n):
        return (n[0] * paso, n[1] * paso)

    umbral = margen_m + SEGURIDAD_MUESTREO_M
    libre = {}

    def es_libre(n):
        v = libre.get(n)
        if v is None:
            v = libre[n] = (0 <= n[0] < ni and 0 <= n[1] < nj
                            and campo.en(*pos(n)) >= umbral)
        return v

    d_ini, d_fin = campo.en(*inicio), campo.en(*destino)
    if d_ini <= 0.0 or d_fin <= 0.0:
        return None
    tol_ini, tol_fin = min(margen_m, d_ini), min(margen_m, d_fin)

    def evaluar(a, b, ta=None, tb=None):
        return campo.evaluar(a, b, margen_m, preferido_m, ta, tb)

    def los(a, b, ta=None, tb=None):
        return evaluar(a, b, ta, tb)[0]

    # Si inicio y destino se ven, la ruta es la recta.
    libre_directo, costo_directo = evaluar(inicio, destino, tol_ini, tol_fin)
    if libre_directo and costo_directo <= (math.hypot(destino[0] - inicio[0],
                                                       destino[1] - inicio[1]) * celda) * 1.05:
        return [inicio, destino]            # recta y sin zonas estrechas

    radio_ext = ZONA_EXTREMO_M * 2.0 / celda            # alcance de enlace con los extremos
    vecinos_destino = {}
    for i in range(max(0, int((destino[0] - radio_ext) / paso)), min(ni, int((destino[0] + radio_ext) / paso) + 2)):
        for j in range(max(0, int((destino[1] - radio_ext) / paso)), min(nj, int((destino[1] + radio_ext) / paso) + 2)):
            n = (i, j)
            if es_libre(n) and math.hypot(pos(n)[0] - destino[0], pos(n)[1] - destino[1]) <= radio_ext \
                    and los(pos(n), destino, None, tol_fin):
                vecinos_destino[n] = True

    INICIO, FIN = "S", "G"
    g = {INICIO: 0.0}
    padre = {INICIO: INICIO}
    abiertos = []
    cerrados = set()

    def coord(n):
        return inicio if n == INICIO else destino if n == FIN else pos(n)

    def h(p):
        return math.hypot(destino[0] - p[0], destino[1] - p[1])

    def distancia(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # Sucesores del inicio: nodos libres visibles dentro del radio.
    for i in range(max(0, int((inicio[0] - radio_ext) / paso)), min(ni, int((inicio[0] + radio_ext) / paso) + 2)):
        for j in range(max(0, int((inicio[1] - radio_ext) / paso)), min(nj, int((inicio[1] + radio_ext) / paso) + 2)):
            n = (i, j)
            if es_libre(n) and distancia(pos(n), inicio) <= radio_ext:
                ok, costo = evaluar(inicio, pos(n), tol_ini, None)
                if not ok:
                    continue
                if costo < g.get(n, math.inf):
                    g[n], padre[n] = costo, INICIO
                    heapq.heappush(abiertos, (costo + h(pos(n)), next(orden), n))
    contador = 0
    while abiertos:
        _, _, s = heapq.heappop(abiertos)
        if s in cerrados:
            continue
        if s == FIN:
            ruta, n = [], FIN
            while n != INICIO:
                ruta.append(coord(n))
                n = padre[n]
            ruta.append(inicio)
            return ruta[::-1]
        cerrados.add(s)
        contador += 1
        ps = coord(s)
        p = padre[s]
        pp = coord(p)
        # Enlace directo al destino desde nodos cercanos a él.
        if s in vecinos_destino:
            ok, c_p = (evaluar(pp, destino, tol_ini if p == INICIO else None, tol_fin)
                       if p != s else (False, math.inf))
            if ok:
                cand, par = g[p] + c_p, p
            else:
                cand, par = g[s] + evaluar(ps, destino, None, tol_fin)[1], s
            if cand < g.get(FIN, math.inf):
                g[FIN], padre[FIN] = cand, par
                heapq.heappush(abiertos, (cand, next(orden), FIN))
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                n = (s[0] + di, s[1] + dj) if s != INICIO else None
                if n is None or n in cerrados or not es_libre(n):
                    continue
                pn = pos(n)
                if di != 0 and dj != 0:                 # no cortar esquinas de obstáculos
                    if not (es_libre((s[0] + di, s[1])) and es_libre((s[0], s[1] + dj))):
                        continue
                ok, c_p = evaluar(pp, pn, tol_ini if p == INICIO else None, None)
                if ok:
                    cand, par = g[p] + c_p, p
                else:
                    cand, par = g[s] + evaluar(ps, pn, None, None)[1], s
                if not math.isfinite(cand):
                    continue
                if cand < g.get(n, math.inf):
                    g[n], padre[n] = cand, par
                    heapq.heappush(abiertos, (cand + h(pn), next(orden), n))
    return None


def _afinar(campo, vertices, margen_m, preferido_m):
    """Pasada final de atajos: solo si el atajo cuesta menos que el tramo que reemplaza."""
    if len(vertices) <= 2:
        return vertices
    d_ini, d_fin = campo.en(*vertices[0]), campo.en(*vertices[-1])
    tol_ini, tol_fin = min(margen_m, d_ini), min(margen_m, d_fin)
    ultimo = len(vertices) - 1

    def costo(i, j):
        return campo.evaluar(vertices[i], vertices[j], margen_m, preferido_m,
                             tol_ini if i == 0 else None, tol_fin if j == ultimo else None)

    salida, i = [vertices[0]], 0
    while i < ultimo:
        mejor = i + 1
        for j in range(ultimo, i + 1, -1):
            ok, c = costo(i, j)
            if ok and c <= sum(costo(k, k + 1)[1] for k in range(i, j)) + 1e-9:
                mejor = j
                break
        salida.append(vertices[mejor])
        i = mejor
    return salida


def planificar(mapa, inicio, destino, margen_m: float, holgura_m: float = 0.10):
    """Mejor ruta continua de ``inicio`` a ``destino`` (celdas fraccionarias).

    * Cualquier ángulo: vértices libres, no escalera de celdas.
    * Parte de la posición REAL del robot y llega al punto exacto pedido.
    * Despeje mínimo duro = ``margen_m`` (radio del robot + seguridad). Entre ese
      margen y ``margen_m + holgura_m`` los metros "cuestan" más
      (PENALIZACION_ESTRECHO): prefiere lo ancho salvo que el rodeo sea mayor que
      el sobrecosto. Como último recurso usa el A* de celdas, así que siempre
      devuelve una ruta si existe una.

    Devuelve la lista de vértices [(fila, col), ...]; levanta ``SinRuta`` si no hay.
    """
    inicio = (float(inicio[0]), float(inicio[1]))
    destino = (float(destino[0]), float(destino[1]))
    campo = campo_de(mapa)
    vertices = _theta_estrella(mapa, campo, inicio, destino, margen_m, margen_m + holgura_m)
    if vertices is not None:
        return [tuple(v) for v in _afinar(campo, vertices, margen_m, margen_m + holgura_m)]
    # Último recurso: A* de celdas (garantiza completitud en la grilla).
    ruta = astar(mapa, _celda_libre(mapa, *inicio), _celda_libre(mapa, *destino), margen_m)
    return suavizar_ruta(mapa, ruta, margen_m)


def _celda_libre(mapa, f: float, c: float):
    """Celda transitable más cercana a un punto (fila, col) fraccionario."""
    fi, ci = int(round(f)), int(round(c))
    if 0 <= fi < mapa.filas and 0 <= ci < mapa.columnas and mapa.es_transitable(fi, ci):
        return fi, ci
    libres = [(a, b) for a in range(mapa.filas) for b in range(mapa.columnas)
              if mapa.es_transitable(a, b)]
    if not libres:
        raise SinRuta("El mapa no tiene celdas transitables.")
    return min(libres, key=lambda p: (p[0] - f) ** 2 + (p[1] - c) ** 2)


def recortar_final(mapa, vertices, distancia_m: float):
    """Acorta la ruta ``distancia_m`` por el final (para parar ANTES del objetivo).

    Devuelve los vértices hasta el punto que queda a ``distancia_m`` del final
    medido sobre la ruta; si la ruta es más corta, queda solo el primer vértice.
    """
    vertices = [tuple(v) for v in vertices]
    resto = distancia_m / mapa.tamano_celda
    while len(vertices) >= 2 and resto > 1e-9:
        (ax, ay), (bx, by) = vertices[-2], vertices[-1]
        largo = math.hypot(bx - ax, by - ay)
        if largo > resto:
            t = (largo - resto) / largo
            vertices[-1] = (ax + (bx - ax) * t, ay + (by - ay) * t)
            return vertices
        resto -= largo
        vertices.pop()
    return vertices


def largo_ruta_m(mapa, vertices) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(vertices, vertices[1:])) * mapa.tamano_celda
