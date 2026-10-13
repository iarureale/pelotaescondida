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
        self.tolerancia_m = tolerancia_m
        limite_autonomo = getattr(mundo, "velocidad_autonoma_max", None)
        self.velocidad_m_s = min(velocidad_m_s,
                                 limite_autonomo or mundo.perfil.velocidad_max)
        self.indice = 0
        self.estado = "esperando"
        self._hilo = None

    def paso(self) -> str:
        """Actualiza una orden corta. Expuesto para integrar y probar por ciclo."""
        if self.estado in ("completado", "error", "detenido"):
            return self.estado
        self.estado = "navegando"
        pose = self.mundo.leer()
        while self.indice < len(self.puntos):
            tx, ty = self.puntos[self.indice]
            dx, dy = tx - pose["x"], ty - pose["y"]
            distancia = math.hypot(dx, dy)
            if distancia <= self.tolerancia_m:
                self.indice += 1
                continue
            objetivo = math.atan2(dy, dx)
            error = (objetivo - pose["yaw"] + math.pi) % (2 * math.pi) - math.pi
            if abs(error) > 0.20:
                limite_giro = (getattr(self.mundo, "velocidad_angular_autonoma_max", None)
                               or self.mundo.perfil.velocidad_angular_max)
                vyaw = max(-limite_giro, min(limite_giro, 1.6 * error))
                self.mundo.mover_autonomo(0.0, 0.0, vyaw, duracion=0.15)
            else:
                vx = min(self.velocidad_m_s, max(0.06, distancia * 0.8))
                self.mundo.mover_autonomo(vx, 0.0, 0.0, duracion=0.15)
            return self.estado
        self.mundo.detener()
        self.estado = "completado"
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
