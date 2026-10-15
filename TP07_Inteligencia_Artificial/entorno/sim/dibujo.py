"""Dibuja la grilla y ruta en la ventana 3D.

Usa las geometrías del visor para celdas libres y ruta. En el modo A* las
cajas de obstáculo se compilan además como geometrías MJCF con colisiones.
Los mapas didácticos conservan el dibujo superpuesto.
"""

from __future__ import annotations

import math

import numpy as np

# Colores (RGBA). Pensados para que se distingan tambien en captura de pantalla.
COLOR_LIBRE = np.array([0.86, 0.88, 0.91, 0.35], dtype=np.float32)
COLOR_OBSTACULO = np.array([0.72, 0.18, 0.18, 0.90], dtype=np.float32)
COLOR_PROHIBIDA = np.array([0.85, 0.65, 0.10, 0.70], dtype=np.float32)
COLOR_INICIO = np.array([0.20, 0.65, 0.32, 0.75], dtype=np.float32)
COLOR_DESTINO = np.array([0.15, 0.45, 0.85, 0.75], dtype=np.float32)
COLOR_RUTA = np.array([0.95, 0.75, 0.10, 0.95], dtype=np.float32)

ALTURA_PISO = 0.005      # las baldosas, casi al ras
ALTURA_OBSTACULO = 0.35  # los obstaculos, visibles pero sin tapar al robot
IDENTIDAD = np.eye(3).flatten()


def dibujar(user_scn, mapa, mostrar_ruta: bool = True,
           mostrar_obstaculos: bool = True,
           mostrar_baldosas: bool = True) -> int:
    """Vuelca la grilla en la escena del visor. Devuelve cuantos geoms uso.

    ``mostrar_baldosas=False`` omite las baldosas por celda: en el patio UADE su
    cara superior (z=1,0 cm) queda bajo la plaza (z=1,8 cm), no se ven y solo
    cuestan ~720 geoms por cuadro. La ruta dinámica se dibuja aparte con
    ``actualizar_ruta_pendiente``.
    """
    import mujoco

    n = 0
    media = mapa.tamano_celda / 2.0

    def _geom(tipo, tamano, pos, color):
        nonlocal n
        if n >= user_scn.maxgeom:
            return
        mujoco.mjv_initGeom(
            user_scn.geoms[n], tipo,
            np.array(tamano, dtype=np.float64),
            np.array(pos, dtype=np.float64),
            IDENTIDAD, color)
        n += 1

    for fila in range(mapa.filas):
        for columna in range(mapa.columnas):
            x, y = mapa.a_mundo(fila, columna)
            valor = mapa.celda(fila, columna)

            if valor == 1:      # el modo A* usa cajas MJCF con colision real
                if mostrar_obstaculos:
                    _geom(mujoco.mjtGeom.mjGEOM_BOX,
                          [media * 0.92, media * 0.92, ALTURA_OBSTACULO / 2],
                          [x, y, ALTURA_OBSTACULO / 2], COLOR_OBSTACULO)
                continue

            if valor == 2:      # zona prohibida: baldosa marcada, sin volumen
                color = COLOR_PROHIBIDA
            elif (fila, columna) == mapa.inicio:
                color = COLOR_INICIO
            elif (fila, columna) == mapa.destino:
                color = COLOR_DESTINO
            else:
                color = COLOR_LIBRE

            if mostrar_baldosas:
                _geom(mujoco.mjtGeom.mjGEOM_BOX,
                      [media * 0.92, media * 0.92, ALTURA_PISO],
                      [x, y, ALTURA_PISO], color)

    # La ruta calculada, como esferas encadenadas sobre el piso.
    if mostrar_ruta and mapa.ruta:
        for fila, columna in mapa.ruta:
            x, y = mapa.a_mundo(int(fila), int(columna))
            _geom(mujoco.mjtGeom.mjGEOM_SPHERE,
                  [mapa.tamano_celda * 0.10] * 3, [x, y, 0.05], COLOR_RUTA)

    user_scn.ngeom = n
    return n


# --- Ruta dinámica: solo lo que FALTA recorrer ---------------------------------
Z_TRAZO = 0.06           # sobre el césped (cara superior 3,2 cm) y la plaza (1,8 cm)
RADIO_TRAZO = 0.025
PASO_TRAZO_M = 0.50      # distancia entre inicios de trazos
LARGO_TRAZO_M = 0.30
DESVANECER_M = 1.2       # el camino aparece de a poco desde el robot hacia adelante


def _acumulado(polilinea):
    cum = [0.0]
    for (ax, ay), (bx, by) in zip(polilinea, polilinea[1:]):
        cum.append(cum[-1] + math.hypot(bx - ax, by - ay))
    return cum


def _punto(polilinea, cum, s):
    s = max(0.0, min(cum[-1], s))
    for k in range(len(cum) - 1):
        if s <= cum[k + 1] or k == len(cum) - 2:
            largo = cum[k + 1] - cum[k]
            t = 0.0 if largo < 1e-9 else (s - cum[k]) / largo
            (ax, ay), (bx, by) = polilinea[k], polilinea[k + 1]
            return ax + (bx - ax) * t, ay + (by - ay) * t
    return polilinea[-1]


def trazos_ruta(polilinea, progreso_m: float):
    """Trazos [(puntos, opacidad)] de la ruta que falta, con fundido progresivo.

    Los trazos están anclados a la ruta (no al robot): a medida que el robot
    avanza (``progreso_m``) cada trazo se acorta y desaparece al pasar sobre él,
    y los que están más cerca del robot que ``DESVANECER_M`` salen translúcidos.
    No queda nada dibujado detrás del robot.
    """
    if len(polilinea) < 2:
        return []
    cum = _acumulado(polilinea)
    total = cum[-1]
    if not progreso_m < total - 0.02:      # llegó (o inf/NaN): no queda nada por mostrar
        return []
    trazos = []
    k = max(0, int((progreso_m - LARGO_TRAZO_M) // PASO_TRAZO_M))
    while k * PASO_TRAZO_M < total:
        a = k * PASO_TRAZO_M
        b = min(total, a + LARGO_TRAZO_M)
        k += 1
        if b <= progreso_m + 0.02:
            continue
        a = max(a, progreso_m)
        t = max(0.0, min(1.0, ((a + b) / 2 - progreso_m) / DESVANECER_M))
        opacidad = t * t * (3 - 2 * t)
        if opacidad < 0.03:
            continue
        puntos = [_punto(polilinea, cum, a)]
        puntos += [tuple(v) for v, c in zip(polilinea, cum) if a < c < b]   # esquinas
        puntos.append(_punto(polilinea, cum, b))
        trazos.append((puntos, opacidad))
    return trazos


def actualizar_ruta_pendiente(user_scn, n_base: int, polilinea, progreso_m: float) -> int:
    """Escribe los trazos pendientes a partir de ``n_base`` y fija ``ngeom``.

    Con ``polilinea`` vacía borra la ruta (solo quedan las ``n_base`` geoms
    estáticas). Reutiliza los mismos slots de geometría cada cuadro: no se
    acumulan objetos. Devuelve el total de geoms en uso.
    """
    import mujoco

    n = n_base
    for puntos, opacidad in (trazos_ruta(polilinea, progreso_m) if polilinea else []):
        color = COLOR_RUTA.copy()
        color[3] = 0.95 * opacidad
        for (ax, ay), (bx, by) in zip(puntos, puntos[1:]):
            if n >= user_scn.maxgeom:
                break
            geom = user_scn.geoms[n]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                np.zeros(3), np.zeros(3), IDENTIDAD, color)
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, RADIO_TRAZO,
                                 np.array([ax, ay, Z_TRAZO]), np.array([bx, by, Z_TRAZO]))
            n += 1
    user_scn.ngeom = n
    return n
