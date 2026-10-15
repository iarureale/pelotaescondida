"""Seguimiento en lazo cerrado de una ruta A* en el mundo cinemático UADE."""
from __future__ import annotations

import math
import threading
import time


class NavegadorAstar:
    """Controla velocidad y giro usando la pose publicada por ``Mundo``.

    En este paquete la pose es ground truth del simulador cinemático, no una
    estimación sensorial ni un estado de un Go2 con dinámica física.
    """

    def __init__(self, mundo, mapa, ruta, tolerancia_m: float = 0.12,
                 velocidad_m_s: float | None = None):
        if velocidad_m_s is None:
            velocidad_m_s = (getattr(mundo, "velocidad_autonoma_max", None)
                             or mundo.perfil.velocidad_max)
        if not ruta:
            raise ValueError("La ruta no puede estar vacía.")
        if tolerancia_m <= 0 or velocidad_m_s <= 0:
            raise ValueError("La tolerancia y la velocidad deben ser positivas.")
        self.mundo = mundo
        self.mapa = mapa
        # Si el robot ya entró en la celda objetivo durante una actualización
        # dinámica, A* devuelve un nodo. Aún puede faltarle llegar al centro.
        # Conserva los vértices donde cambia el rumbo. Así el Go2 corre por
        # tramos largos en vez de frenar ante cada celda consecutiva de A*.
        ruta_control = _comprimir_tramos_rectos(ruta)
        celdas_objetivo = ruta_control[1:] if len(ruta_control) > 1 else ruta_control
        self.puntos = [mapa.a_mundo(*p) for p in celdas_objetivo]
        p_inicio = mundo.leer()
        # Polilínea fija de la ruta vigente (posición inicial + vértices): es la
        # que se dibuja y la que se sigue, así que ambas coinciden siempre.
        self._polilinea = [(p_inicio["x"], p_inicio["y"])] + list(self.puntos)
        self.tolerancia_m = tolerancia_m
        limite_autonomo = getattr(mundo, "velocidad_autonoma_max", None)
        self.velocidad_m_s = min(velocidad_m_s,
                                 limite_autonomo or mundo.perfil.velocidad_max)
        self.indice = 0
        self.estado = "esperando"
        self._hilo = None
        self._t_previo = 0.0
        self._v = 0.0
        self._w = 0.0
        p0 = mundo.leer()
        self._origen = (p0["x"], p0["y"])
        self._ref_t = time.monotonic()
        self._ref_xy = (p0["x"], p0["y"])

    # --- Parámetros del seguimiento (ver NAVEGACION_GO2.md, "Seguimiento") ---
    LOOKAHEAD_M = 0.80        # "zanahoria" sobre el tramo: mantiene al robot en la línea
    TOLERANCIA_ESQUINA_M = 0.30
    GANANCIA_GIRO = 2.0
    ERROR_GIRO_EN_SITIO = 0.90   # rad; por encima gira sin avanzar
    ACEL_LINEAL = 1.5            # m/s²  (rampa de arranque/frenado)
    ACEL_ANGULAR = 5.0           # rad/s² (evita golpes de giro)
    VENTANA_ATASCO_S = 6.0
    PROGRESO_MIN_M = 0.20

    def polilinea(self):
        """Vértices (x, y) de la ruta que sigue el robot, desde donde arrancó."""
        return list(self._polilinea)

    def progreso_m(self) -> float:
        """Metros recorridos sobre la polilínea (proyección de la pose actual)."""
        if self.estado == "completado" or self.indice >= len(self._polilinea) - 1:
            return float("inf")
        pose = self.mundo.leer()
        k = min(self.indice, len(self._polilinea) - 2)
        recorrido = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                        for a, b in zip(self._polilinea[:k], self._polilinea[1:k + 1]))
        (ax, ay), (bx, by) = self._polilinea[k], self._polilinea[k + 1]
        largo = math.hypot(bx - ax, by - ay)
        if largo < 1e-9:
            return recorrido
        t = ((pose["x"] - ax) * (bx - ax) + (pose["y"] - ay) * (by - ay)) / largo
        return recorrido + max(0.0, min(largo, t))

    def _limites(self):
        w = (getattr(self.mundo, "velocidad_angular_autonoma_max", None)
             or self.mundo.perfil.velocidad_angular_max)
        return self.velocidad_m_s, w

    def paso(self) -> str:
        """Una orden corta y continua. Se llama en CADA ciclo del visor (~50 Hz).

        Sigue el tramo anterior->objetivo con un punto de mira (pure pursuit
        simplificado): el robot converge a la línea de la grilla en lugar de
        cortar en diagonal hacia la esquina, gira de forma progresiva
        (avanza mientras corrige) y solo gira en el sitio si el error es
        grande. Velocidades con rampa: sin saltos ni vibraciones.
        """
        if self.estado in ("completado", "error", "detenido", "atascado"):
            return self.estado
        self.estado = "navegando"
        ahora = time.monotonic()
        dt = min(0.2, max(1e-3, ahora - self._t_previo)) if self._t_previo else 0.02
        self._t_previo = ahora
        pose = self.mundo.leer()
        px, py, yaw = pose["x"], pose["y"], pose["yaw"]
        vmax, wmax = self._limites()

        while self.indice < len(self.puntos):
            tx, ty = self.puntos[self.indice]
            es_final = self.indice == len(self.puntos) - 1
            tol = self.tolerancia_m if es_final else self.TOLERANCIA_ESQUINA_M
            if math.hypot(tx - px, ty - py) <= tol:
                self._origen = (tx, ty)
                self.indice += 1
                continue
            break
        else:
            self.mundo.detener()
            self.estado = "completado"
            return self.estado

        ox, oy = self._origen if self._origen else (px, py)
        dist = math.hypot(tx - px, ty - py)
        # Punto de mira: proyección sobre el tramo + LOOKAHEAD, sin pasar del objetivo.
        sx, sy = tx - ox, ty - oy
        largo = math.hypot(sx, sy)
        if largo > 1e-6:
            ux, uy = sx / largo, sy / largo
            proy = (px - ox) * ux + (py - oy) * uy
            mx = ox + ux * min(largo, max(0.0, proy) + self.LOOKAHEAD_M)
            my = oy + uy * min(largo, max(0.0, proy) + self.LOOKAHEAD_M)
        else:
            mx, my = tx, ty
        error = (math.atan2(my - py, mx - px) - yaw + math.pi) % (2 * math.pi) - math.pi

        w_deseada = max(-wmax, min(wmax, self.GANANCIA_GIRO * error))
        if abs(error) > self.ERROR_GIRO_EN_SITIO:
            v_deseada = 0.0
        else:
            alineacion = max(0.0, math.cos(error)) ** 2
            # Frena para la esquina (giro de 90°) y para el destino final.
            v_frenado = (max(0.06, 0.8 * dist) if es_final
                         else 0.35 + 1.4 * dist)
            v_deseada = min(vmax, v_frenado) * alineacion
        # Rampas de aceleración.
        dv = self.ACEL_LINEAL * dt
        dw = self.ACEL_ANGULAR * dt
        self._v = max(self._v - 3.0 * dt if v_deseada < self._v else self._v,
                      min(self._v + dv, v_deseada))
        self._w = max(self._w - dw, min(self._w + dw, w_deseada))
        self.mundo.mover_autonomo(self._v, 0.0, self._w, duracion=0.50)

        # Detección de atasco: comanda avanzar pero no se mueve.
        if (ahora - self._ref_t) >= self.VENTANA_ATASCO_S:
            if math.hypot(px - self._ref_xy[0], py - self._ref_xy[1]) < self.PROGRESO_MIN_M:
                self.mundo.detener()
                self.estado = "atascado"
                return self.estado
            self._ref_t, self._ref_xy = ahora, (px, py)
        return self.estado

    def iniciar(self, detener: threading.Event, periodo_s: float = 0.05):
        if self._hilo is not None and self._hilo.is_alive():
            raise RuntimeError("El navegador ya está activo.")

        def ejecutar():
            try:
                while not detener.is_set():
                    estado = self.paso()
                    if estado != "navegando":
                        return
                    time.sleep(periodo_s)
            except Exception:
                self.mundo.detener()
                self.estado = "error"
                raise
            finally:
                if self.estado != "completado":
                    self.mundo.detener()
                    if self.estado == "navegando":
                        self.estado = "detenido"

        self._hilo = threading.Thread(target=ejecutar, name="go2-astar", daemon=True)
        self._hilo.start()
        return self._hilo


def _comprimir_tramos_rectos(ruta):
    """Devuelve extremos y esquinas de una ruta ortogonal, sin crear atajos."""
    if len(ruta) <= 2:
        return list(ruta)
    resultado = [ruta[0]]
    direccion = (ruta[1][0] - ruta[0][0], ruta[1][1] - ruta[0][1])
    for i in range(1, len(ruta) - 1):
        siguiente = (ruta[i + 1][0] - ruta[i][0],
                     ruta[i + 1][1] - ruta[i][1])
        if siguiente != direccion:
            resultado.append(ruta[i])
        direccion = siguiente
    resultado.append(ruta[-1])
    return resultado
