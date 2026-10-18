"""Dibuja la grilla y ruta en la ventana 3D.

Usa las geometrías del visor para celdas libres y ruta. En el modo A* las
cajas de obstáculo se compilan además como geometrías MJCF con colisiones.
Los mapas didácticos conservan el dibujo superpuesto.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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
@dataclass(frozen=True)
class EstiloRuta:
    """Aspecto de la ruta. Cambiar el estilo no toca la lógica de navegación.

    forma        "flecha" (chevrones ">" planos orientados con la ruta) o "trazo"
                 (rayitas); ambas sirven para cualquier polilínea, recta o curva.
    paso_m       distancia entre marcas a lo largo de la ruta
    largo_m      largo de cada brazo de la flecha / de cada trazo
    ancho_m      ancho del brazo / del trazo
    apertura_rad semiángulo de la punta de la flecha
    z            altura sobre el piso (sobre el césped, cara superior 3,2 cm)
    color        RGBA base
    desvanecer_m longitud del fundido cerca del robot
    radio_esquina_m  radio con que se redondean las esquinas al DIBUJAR (0 = sin redondeo)
    """
    forma: str = "flecha"
    paso_m: float = 0.80
    largo_m: float = 0.40
    ancho_m: float = 0.075
    apertura_rad: float = math.radians(32)
    z: float = 0.045
    color: tuple = (1.0, 0.72, 0.05, 0.95)
    desvanecer_m: float = 1.5
    radio_esquina_m: float = 0.9


ESTILO_FLECHAS = EstiloRuta()
ESTILO_TRAZOS = EstiloRuta(forma="trazo", paso_m=0.50, largo_m=0.30, ancho_m=0.05,
                           z=0.06, desvanecer_m=1.2, radio_esquina_m=0.0)


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


def _rumbo(polilinea, cum, s, ventana=0.15):
    """Dirección (rad) de la ruta en el arco ``s``."""
    (x0, y0) = _punto(polilinea, cum, s - ventana)
    (x1, y1) = _punto(polilinea, cum, s + ventana)
    return math.atan2(y1 - y0, x1 - x0)


def suavizar_polilinea(polilinea, radio_m: float = 0.9, muestras: int = 6):
    """Redondea cada esquina con una curva de radio acotado; conserva los extremos.

    Solo es para dibujar: el seguidor recorta las esquinas unos 0,8-1 m por su
    punto de mira, y la línea redondeada se parece más a lo que hace el robot que
    el ángulo recto de A*. El recorte nunca supera la mitad de cada tramo, así
    que los tramos cortos (diagonales de atajo) no se deforman.
    """
    pol = [tuple(p) for p in polilinea]
    if radio_m <= 0 or len(pol) < 3:
        return pol
    salida = [pol[0]]
    for i in range(1, len(pol) - 1):
        (ax, ay), (vx, vy), (bx, by) = pol[i - 1], pol[i], pol[i + 1]
        l1, l2 = math.hypot(vx - ax, vy - ay), math.hypot(bx - vx, by - vy)
        if l1 < 1e-9 or l2 < 1e-9:
            continue
        d = min(radio_m, 0.5 * l1, 0.5 * l2)
        p0 = (vx - (vx - ax) / l1 * d, vy - (vy - ay) / l1 * d)
        p1 = (vx + (bx - vx) / l2 * d, vy + (by - vy) / l2 * d)
        for k in range(muestras + 1):                    # Bézier cuadrática p0 -V- p1
            t = k / muestras
            salida.append(((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * vx + t * t * p1[0],
                           (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * vy + t * t * p1[1]))
    salida.append(pol[-1])
    return salida


def proyectar(polilinea, x: float, y: float, minimo_m: float = 0.0) -> float:
    """Arco (m) del punto de la polilínea más cercano a (x, y), sin retroceder.

    ``minimo_m`` evita que la proyección salte hacia atrás en cruces o cuando la
    ruta pasa cerca de sí misma: solo se consideran puntos con arco >= minimo_m - 0,5.
    """
    cum = _acumulado(polilinea)
    mejor, mejor_d = max(0.0, minimo_m), float("inf")
    for k, ((ax, ay), (bx, by)) in enumerate(zip(polilinea, polilinea[1:])):
        largo = cum[k + 1] - cum[k]
        if cum[k + 1] < minimo_m - 0.5 or largo < 1e-9:
            continue
        t = ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / (largo * largo)
        t = max(0.0, min(1.0, t))
        d = math.hypot(x - (ax + (bx - ax) * t), y - (ay + (by - ay) * t))
        if d < mejor_d:
            mejor_d, mejor = d, max(minimo_m - 0.5, cum[k] + t * largo)
    return mejor


def _opacidad(distancia_m: float, desvanecer_m: float) -> float:
    t = max(0.0, min(1.0, distancia_m / desvanecer_m))
    return t * t * (3 - 2 * t)


def trazos_ruta(polilinea, progreso_m: float, estilo: EstiloRuta = ESTILO_TRAZOS):
    """Trazos [(puntos, opacidad)] de la ruta que falta, con fundido progresivo.

    Los trazos están anclados a la ruta (no al robot): a medida que el robot
    avanza (``progreso_m``) cada trazo se acorta y desaparece al pasar sobre él,
    y los más cercanos al robot que ``desvanecer_m`` salen translúcidos. No queda
    nada dibujado detrás del robot.
    """
    if len(polilinea) < 2:
        return []
    cum = _acumulado(polilinea)
    total = cum[-1]
    if not progreso_m < total - 0.02:      # llegó (o inf/NaN): no queda nada por mostrar
        return []
    trazos = []
    k = max(0, int((progreso_m - estilo.largo_m) // estilo.paso_m))
    while k * estilo.paso_m < total:
        a = k * estilo.paso_m
        b = min(total, a + estilo.largo_m)
        k += 1
        if b <= progreso_m + 0.02:
            continue
        a = max(a, progreso_m)
        opacidad = _opacidad((a + b) / 2 - progreso_m, estilo.desvanecer_m)
        if opacidad < 0.03:
            continue
        puntos = [_punto(polilinea, cum, a)]
        puntos += [tuple(v) for v, c in zip(polilinea, cum) if a < c < b]   # esquinas
        puntos.append(_punto(polilinea, cum, b))
        trazos.append((puntos, opacidad))
    return trazos


def flechas_ruta(polilinea, progreso_m: float, estilo: EstiloRuta = ESTILO_FLECHAS):
    """Flechas [(x, y, rumbo, opacidad)] de la ruta que falta.

    Cada flecha está anclada a la ruta (arco = k * paso). Desaparece por
    completo cuando el robot pasa sobre ella y aparece con fundido a medida que
    queda más lejos por delante; la última queda cerca del destino.
    """
    if len(polilinea) < 2:
        return []
    cum = _acumulado(polilinea)
    total = cum[-1]
    if not progreso_m < total - 0.05:
        return []
    flechas = []
    k = max(0, int(progreso_m // estilo.paso_m))
    while True:
        s = (k + 0.5) * estilo.paso_m
        k += 1
        if s > total - 0.10:
            break
        if s <= progreso_m + 0.05:
            continue
        opacidad = _opacidad(s - progreso_m, estilo.desvanecer_m)
        if opacidad < 0.04:
            continue
        x, y = _punto(polilinea, cum, s)
        flechas.append((x, y, _rumbo(polilinea, cum, s), opacidad))
    return flechas


def _agregar_geom(user_scn, n, tipo, tamano, pos, mat, rgba):
    import mujoco
    if n >= user_scn.maxgeom:
        return n
    mujoco.mjv_initGeom(user_scn.geoms[n], tipo, np.asarray(tamano, dtype=np.float64),
                        np.asarray(pos, dtype=np.float64), np.asarray(mat, dtype=np.float64),
                        np.asarray(rgba, dtype=np.float32))
    return n + 1


def actualizar_ruta_pendiente(user_scn, n_base: int, polilinea, progreso_m: float,
                              estilo: EstiloRuta = ESTILO_FLECHAS) -> int:
    """Escribe la ruta pendiente a partir de ``n_base`` y fija ``ngeom``.

    ``polilinea`` y ``progreso_m`` deben estar en la MISMA polilínea (la que se
    dibuja; ver ``suavizar_polilinea`` y ``proyectar``). Con polilínea vacía
    borra la ruta (solo quedan las ``n_base`` geoms estáticas). Reutiliza los
    mismos slots cada cuadro: no se acumulan objetos. Devuelve el total en uso.
    """
    import mujoco

    n = n_base
    if not polilinea:
        user_scn.ngeom = n
        return n
    if estilo.forma == "flecha":
        for x, y, rumbo, opacidad in flechas_ruta(polilinea, progreso_m, estilo):
            escala = 0.55 + 0.45 * opacidad           # también se achican al acercarse
            largo, ancho = estilo.largo_m * escala, estilo.ancho_m * escala
            cuerpo = largo * math.cos(estilo.apertura_rad)
            punta = (x + 0.5 * cuerpo * math.cos(rumbo), y + 0.5 * cuerpo * math.sin(rumbo))
            color = list(estilo.color)
            color[3] = estilo.color[3] * opacidad
            for lado in (1.0, -1.0):                    # los dos brazos del chevrón ">"
                psi = rumbo + math.pi + lado * estilo.apertura_rad
                c, s_ = math.cos(psi), math.sin(psi)
                centro = (punta[0] + 0.5 * largo * c, punta[1] + 0.5 * largo * s_, estilo.z)
                n = _agregar_geom(user_scn, n, mujoco.mjtGeom.mjGEOM_BOX,
                                  [largo / 2, ancho / 2, 0.004], centro,
                                  [c, -s_, 0, s_, c, 0, 0, 0, 1], color)
    else:
        for puntos, opacidad in trazos_ruta(polilinea, progreso_m, estilo):
            color = np.array(estilo.color, dtype=np.float32)
            color[3] = estilo.color[3] * opacidad
            for (ax, ay), (bx, by) in zip(puntos, puntos[1:]):
                if n >= user_scn.maxgeom:
                    break
                geom = user_scn.geoms[n]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                    np.zeros(3), np.zeros(3), IDENTIDAD, color)
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                     estilo.ancho_m / 2,
                                     np.array([ax, ay, estilo.z]), np.array([bx, by, estilo.z]))
                n += 1
    user_scn.ngeom = n
    return n
