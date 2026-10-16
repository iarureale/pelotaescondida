"""Compara la ruta del planificador contra la ruta ÓPTIMA continua (referencia).

Referencia: Dijkstra con 16 vecinos sobre una grilla fina (0,1 m) y el mismo
despeje mínimo duro que usa el robot, a longitud pura (sin preferencia por zonas anchas). Es lenta
(solo para medir); el planificador del simulador debe acercarse a ella.
Uso (desde entorno/):  python herramientas/benchmark_rutas.py
Requiere scipy (solo para esta medición).
"""
import math, os, sys, time
RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, RAIZ + "/entorno")
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from sim.grilla import cargar, LIBRE
from sim import planificacion_go2 as P

MARGEN = 0.39 + 0.04        # despeje duro del planificador (margen + muestreo); la holgura es solo preferencia
RES = 0.1


def campo_despeje(mapa):
    """Distancia (m) de cada punto de la grilla fina al obstáculo más cercano."""
    c = mapa.tamano_celda
    ys = (np.arange(int(mapa.filas * c / RES)) + 0.5) * RES / c - 0.5      # coord. en celdas (fila)
    xs = (np.arange(int(mapa.columnas * c / RES)) + 0.5) * RES / c - 0.5
    F, C = np.meshgrid(ys, xs, indexing="ij")
    d = np.full(F.shape, np.inf)
    for f in range(mapa.filas):
        for col in range(mapa.columnas):
            if mapa.celda(f, col) == LIBRE:
                continue
            dx = np.maximum(0, np.abs(F - f) - 0.5) * c
            dy = np.maximum(0, np.abs(C - col) - 0.5) * c
            d = np.minimum(d, np.hypot(dx, dy))
    return d


def optimo(mapa, d, a, b, margen=MARGEN):
    h, w = d.shape
    libre = d >= margen
    idx = -np.ones((h, w), dtype=np.int64)
    idx[libre] = np.arange(libre.sum())
    filas, cols, pesos = [], [], []
    vecinos = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)
               if (dy, dx) != (0, 0) and math.gcd(abs(dy), abs(dx)) == 1]   # 16 direcciones
    for dy, dx in vecinos:
        ys0, ys1 = max(0, -dy), min(h, h - dy)
        xs0, xs1 = max(0, -dx), min(w, w - dx)
        o = idx[ys0:ys1, xs0:xs1]; t = idx[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
        ok = (o >= 0) & (t >= 0)
        filas.append(o[ok]); cols.append(t[ok]); pesos.append(np.full(ok.sum(), math.hypot(dy, dx) * RES))
    n = int(libre.sum())
    g = coo_matrix((np.concatenate(pesos), (np.concatenate(filas), np.concatenate(cols))), shape=(n, n)).tocsr()
    def nodo(p):                       # celda (fila, col) -> índice libre más cercano
        y = int(round((p[0] + 0.5) * mapa.tamano_celda / RES - 0.5)); x = int(round((p[1] + 0.5) * mapa.tamano_celda / RES - 0.5))
        for r in range(0, 8):
            for yy in range(y - r, y + r + 1):
                for xx in range(x - r, x + r + 1):
                    if 0 <= yy < h and 0 <= xx < w and idx[yy, xx] >= 0:
                        return idx[yy, xx]
    s, t = nodo(a), nodo(b)
    dist = dijkstra(g, directed=True, indices=s)
    return dist[t]


def largo(v, mapa):
    return sum(math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(v, v[1:])) * mapa.tamano_celda


if __name__ == "__main__":
    mapa = cargar(RAIZ + "/mi_desarrollo/mapas/patio_uade_go2.json")
    d = campo_despeje(mapa)
    bola = tuple(mapa.escenario["pelota_oculta"])
    casos = [("inicio -> pelota", tuple(mapa.inicio), bola), ("(21,15) -> pelota", (21, 15), bola),
             ("(12,8) -> pelota", (12, 8), bola), ("(3,3) -> (20,3)", (3, 3), (20, 3)),
             ("(21,29) -> (3,29)", (21, 29), (3, 29)), ("(20,3) -> (4,26)", (20, 3), (4, 26)),
             ("(8,28) -> (18,5)", (8, 28), (18, 5))]
    total_opt = total_pl = 0.0
    print(f"{'caso':>20} | {'óptima':>7} | {'planificador':>12} | {'exceso':>7} | {'ms':>6}")
    for nombre, a, b in casos:
        opt = optimo(mapa, d, a, b)
        t = time.perf_counter()
        ruta = P.planificar(mapa, a, b, 0.39, 0.10)
        ms = (time.perf_counter() - t) * 1000
        pl = largo(ruta, mapa)
        total_opt += opt; total_pl += pl
        print(f"{nombre:>20} | {opt:6.1f}m | {pl:11.1f}m | {100*(pl/opt-1):6.1f}% | {ms:6.0f}")
    antes = sum(largo(P.suavizar_ruta(mapa, P.astar(mapa, a, b, 0.39), 0.49), mapa) for _, a, b in casos)
    print(f"{'TOTAL':>20} | {total_opt:6.1f}m | {total_pl:11.1f}m | {100*(total_pl/total_opt-1):6.1f}%")
    print(f"A* de 4 vecinos + atajos (pipeline anterior): {antes:.1f} m ({100*(antes/total_opt-1):+.1f}% sobre el óptimo)")
