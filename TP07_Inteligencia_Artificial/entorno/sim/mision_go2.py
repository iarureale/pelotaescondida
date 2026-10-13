"""Máquina de estados para buscar pelota con cámara y A* en el parque MuJoCo."""
from __future__ import annotations

import copy
import math
import time

from .camara_go2 import SeguidorPelota
from .navegador_astar import NavegadorAstar
from .planificacion_go2 import SinRuta, astar


BACKFLIP_CINEMATICO_DISPONIBLE = True


class MisionParque:
    """Explora waypoints conocidos; la celda de la pelota no se usa al planificar."""

    def __init__(self, mundo, mapa, margen_m: float, timeout_s: float = 300.0):
        escenario = getattr(mapa, "escenario", {}) or {}
        if not escenario.get("pelota_oculta"):
            raise ValueError("El mapa de misión debe declarar una pelota oculta.")
        self.mundo = mundo
        self.mapa = mapa
        self.margen_m = margen_m
        self.timeout_s = timeout_s
        self.waypoints = [tuple(p) for p in escenario.get("busqueda", [])]
        if not self.waypoints:
            self.waypoints = [tuple(mapa.destino)]
        self.indice_waypoint = 0
        self.estado = "INICIANDO"
        self.motivo = ""
        self.seguidor = SeguidorPelota(mundo)
        self.navegador = None
        self.ruta = []
        self._ultima_replanificacion = 0.0
        self._bloqueo_desde = None
        self._ultimo_estado = "INICIANDO"
        self._ultimo_xy = (mundo.leer()["x"], mundo.leer()["y"])
        self._inicio = time.monotonic()
        self._confirmaciones_pelota = 0
        self._pelota_perdida_desde = None
        self._escaneo_desde = None
        self._persona_visible_previa = False
        self.metricas = {"distancia_m": 0.0, "replanificaciones": 0,
                         "detecciones_camara": 0, "obstaculos_evitados": 0,
                         "colisiones": 0, "tiempo_busqueda_s": 0.0}
        self._planificar([])

    def _celda_pose(self, x, y):
        celda = self.mapa.tamano_celda
        f = int(round(-y / celda))
        c = int(round(x / celda))
        if not self.mapa.es_transitable(f, c):
            transitables = [(fi, ci) for fi in range(self.mapa.filas)
                            for ci in range(self.mapa.columnas)
                            if self.mapa.es_transitable(fi, ci)]
            if not transitables:
                raise SinRuta("El mapa no tiene celdas transitables.")
            f, c = min(transitables,
                       key=lambda p: (p[0] - f) ** 2 + (p[1] - c) ** 2)
        return f, c

    def _celda_deteccion(self, pose, objeto):
        """Proyecta rango y bearing medidos por cámara usando pose del simulador."""
        # MuJoCo entrega depth axial y el eje horizontal de esta cámara está
        # invertido respecto del eje Y del robot (imagen derecha = robot -Y).
        frente = 0.30 + objeto.distancia_m
        lateral = -objeto.distancia_m * math.tan(objeto.bearing_rad)
        yaw = pose["yaw"]
        x = pose["x"] + frente * math.cos(yaw) - lateral * math.sin(yaw)
        y = pose["y"] + frente * math.sin(yaw) + lateral * math.cos(yaw)
        return self._celda_pose(x, y)

    def _mapa_temporal(self, detecciones, pose):
        mapa = copy.deepcopy(self.mapa)
        for objeto in detecciones:
            # El circuito estático ya forma parte del mapa previo. Solo las
            # personas cambian de posición; agregarlas evita marcar dos veces
            # las mismas cajas, árboles y bancos por su superficie visible.
            if objeto.categoria != "persona":
                continue
            try:
                f, c = self._celda_deteccion(pose, objeto)
            except SinRuta:
                continue
            mapa.grilla[f][c] = 1
        return mapa

    def _planificar(self, detecciones, pose=None):
        pose = pose or self.mundo.leer()
        temporal = self._mapa_temporal(detecciones, pose)
        inicio = self._celda_pose(pose["x"], pose["y"])
        objetivo = self.waypoints[self.indice_waypoint]
        ruta = astar(temporal, inicio=inicio, destino=objetivo,
                     margen_m=self.margen_m)
        self.navegador = NavegadorAstar(self.mundo, temporal, ruta)
        self.ruta = ruta
        self.mapa.ruta = ruta
        self._ultima_replanificacion = time.monotonic()
        self._bloqueo_desde = None
        self.motivo = ""
        self.metricas["replanificaciones"] += 1

    def _esperar_ruta(self, ahora, mensaje):
        self.mundo.detener()
        if self._bloqueo_desde is None:
            self._bloqueo_desde = ahora
        self._ultima_replanificacion = ahora
        if ahora - self._bloqueo_desde >= 10.0:
            self.estado = "ERROR"
            self.motivo = "Camino bloqueado durante 10 s; no hay ruta segura."
        else:
            self.estado = "ESPERANDO_OBSTACULO"
            self.motivo = mensaje

    def paso(self, detecciones):
        if self.estado in ("FINALIZADA", "ERROR", "BACKFLIP_NO_DISPONIBLE",
                           "BACKFLIP_OMITIDO", "BACKFLIP_SIMULADO"):
            return self.estado
        ahora = time.monotonic()
        pose = self.mundo.leer()
        dx, dy = pose["x"] - self._ultimo_xy[0], pose["y"] - self._ultimo_xy[1]
        self.metricas["distancia_m"] += math.hypot(dx, dy)
        self._ultimo_xy = (pose["x"], pose["y"])
        self.metricas["tiempo_busqueda_s"] = ahora - self._inicio
        self.metricas["detecciones_camara"] += len(detecciones)
        if self.metricas["tiempo_busqueda_s"] > self.timeout_s:
            self.mundo.detener()
            self.estado = "ERROR"
            self.motivo = "Se agotó el tiempo límite de búsqueda."
            return self.estado
        if self.estado == "VERIFICANDO_MANIOBRA":
            self._verificar_backflip(detecciones)
            return self.estado
        if (self.estado == "ESPERANDO_OBSTACULO"
                and ahora - self._ultima_replanificacion < 0.5):
            return self.estado

        personas_cercanas = [d for d in detecciones
                             if d.categoria == "persona" and d.distancia_m < 1.0]
        persona_visible = any(d.categoria == "persona" for d in detecciones)
        if persona_visible and not self._persona_visible_previa:
            ruta_actual = set(self.ruta)
            for persona in detecciones:
                if persona.categoria != "persona":
                    continue
                try:
                    if self._celda_deteccion(pose, persona) in ruta_actual:
                        self.metricas["obstaculos_evitados"] += 1
                        break
                except SinRuta:
                    pass
        self._persona_visible_previa = persona_visible
        if personas_cercanas:
            self.mundo.detener()
            self.estado = "ESPERANDO_PERSONA"
            self.motivo = "Persona detectada a menos de 1 m; robot detenido."
            return self.estado

        pelota = next((d for d in detecciones if d.categoria == "pelota"), None)
        # La misión define zonas de búsqueda. Una etiqueta de segmentación de
        # un solo frame fuera de esas zonas puede aparecer por artefactos de
        # rasterización en el borde del frustum; no debe cancelar A* ni mandar
        # al Go2 a perseguir un píxel espurio. El mapa declara desde qué
        # waypoint se acepta el candidato.
        indice_busqueda = int((getattr(self.mapa, "escenario", {}) or {})
                              .get("detectar_pelota_desde_waypoint", 0))
        if pelota is not None and self.indice_waypoint < indice_busqueda:
            pelota = None
        if self.estado == "ESCANEANDO" and pelota is None:
            duracion = float((getattr(self.mapa, "escenario", {}) or {})
                             .get("escaneo_duracion_s", 7.0))
            if ahora - (self._escaneo_desde or ahora) < duracion:
                self.mundo.mover(0.0, 0.0, 0.45, 0.25)
                return self.estado
            self.mundo.detener()
            self._escaneo_desde = None
            if self.indice_waypoint + 1 < len(self.waypoints):
                self.indice_waypoint += 1
                try:
                    self._planificar(detecciones, pose)
                    self.estado = "EXPLORANDO"
                    self.motivo = "Sector escaneado; avanzando al siguiente waypoint."
                except SinRuta:
                    self.estado = "ERROR"
                    self.motivo = "No se puede alcanzar el siguiente sector de búsqueda."
            else:
                self.estado = "ERROR"
                self.motivo = "Se recorrieron los sectores sin detectar la pelota."
            return self.estado
        if pelota is not None:
            self._pelota_perdida_desde = None
            self._confirmaciones_pelota += 1
            if self._confirmaciones_pelota < 2:
                self.mundo.detener()
                self.estado = "DETECTANDO_PELOTA"
                return self.estado
            self.estado = "ACERCANDOSE"
            estado_seguidor = self.seguidor.paso(pelota)
            if estado_seguidor == "distancia_objetivo":
                self.mundo.detener()
                self.estado = "VERIFICANDO_MANIOBRA"
            return self.estado
        self._confirmaciones_pelota = 0
        if self.estado == "ESCANEANDO":
            duracion = max(0.5, float((getattr(self.mapa, "escenario", {}) or {})
                                      .get("escaneo_duracion_s", 7.0)))
            if ahora - self._escaneo_desde < duracion:
                self.mundo.set_velocidad(0.0, 0.0, 0.45, duracion=0.15)
                self.motivo = "Barrido visual del sector, rotando en el lugar."
                return self.estado
            self.mundo.detener()
            self._escaneo_desde = None
            try:
                self._planificar(detecciones, pose)
                self.estado = "NAVEGANDO"
            except SinRuta:
                self.estado = "ERROR"
                self.motivo = "No se puede alcanzar el siguiente sector de búsqueda."
                return self.estado
        if self.estado in ("ACERCANDOSE", "DETECTANDO_PELOTA"):
            if self._pelota_perdida_desde is None:
                self._pelota_perdida_desde = ahora
            if ahora - self._pelota_perdida_desde < 1.5:
                self.mundo.set_velocidad(0.0, 0.0, 0.25, duracion=0.15)
                self.estado = "DETECTANDO_PELOTA"
                self.motivo = "Pelota perdió visibilidad; escaneo lento para reacquirir."
                return self.estado
            self.mundo.detener()
            self.estado = "NAVEGANDO"
            self.motivo = "Pelota no reacquirida; reanudo exploración del mapa."
            self._pelota_perdida_desde = None
            return self.estado

        if self.estado in ("ACERCANDOSE", "ESPERANDO_PERSONA",
                           "ESPERANDO_OBSTACULO"):
            self.estado = "REPLANIFICANDO"
            try:
                self._planificar(detecciones, pose)
            except SinRuta:
                self._esperar_ruta(ahora,
                                   "Obstáculo temporal bloquea el siguiente sector.")
                return self.estado
            self.estado = "REPLANIFICANDO"

        elif ahora - self._ultima_replanificacion >= 0.5:
            try:
                self._planificar(detecciones, pose)
                if any(d.categoria == "persona" for d in detecciones):
                    self.metricas["obstaculos_evitados"] += 1
                    self.estado = "REPLANIFICANDO"
                elif self.estado == "REPLANIFICANDO":
                    self.estado = "NAVEGANDO"
            except SinRuta:
                self._esperar_ruta(ahora,
                                   "Obstáculo temporal bloquea el siguiente sector.")
                return self.estado

        if self.navegador and self.navegador.paso() == "completado":
            self.mundo.detener()
            if self.indice_waypoint + 1 < len(self.waypoints):
                self.indice_waypoint += 1
                indice_busqueda = int((getattr(self.mapa, "escenario", {}) or {})
                                      .get("detectar_pelota_desde_waypoint", 0))
                if self.indice_waypoint >= indice_busqueda:
                    self._escaneo_desde = ahora
                    self.estado = "ESCANEANDO"
                    self.motivo = "Barrido visual del sector antes de continuar la ruta."
                    return self.estado
                try:
                    self._planificar(detecciones, pose)
                    self.estado = "EXPLORANDO"
                except SinRuta:
                    self.estado = "ERROR"
                    self.motivo = "No se puede alcanzar el siguiente sector de búsqueda."
            else:
                self.estado = "ERROR"
                self.motivo = "Se recorrieron los sectores sin detectar la pelota."
            return self.estado
        if self.estado in ("INICIANDO", "REPLANIFICANDO"):
            self.estado = "NAVEGANDO"
        elif self.estado != "EXPLORANDO":
            self.estado = "NAVEGANDO"
        return self.estado

    def _verificar_backflip(self, detecciones):
        self.mundo.detener()
        persona_cerca = any(d.categoria == "persona" and d.distancia_m < 2.5
                            for d in detecciones)
        objetos_cerca = any(d.categoria not in ("pelota", "persona")
                            and d.distancia_m < 2.0 for d in detecciones)
        if self.mundo.leer()["moviendose"] or persona_cerca or objetos_cerca:
            self.estado = "BACKFLIP_OMITIDO"
            self.motivo = ("Backflip omitido: hay objetos/personas cerca o el "
                           "robot todavía se mueve. La misión termina detenido.")
        elif BACKFLIP_CINEMATICO_DISPONIBLE:
            self.mundo.gesto("backflip_visual", duracion=1.8)
            self.estado = "BACKFLIP_SIMULADO"
            self.motivo = ("Pelota encontrada. Ejecutando backflip visual "
                           "cinemático; no envía una orden al robot físico.")
        else:
            # Punto de integración intencional: no hay controlador compatible
            # validado en este simulador para ejecutar esta acción.
            self.estado = "ERROR"
            self.motivo = "No hay adaptador seguro de backflip conectado."

    def registrar_colision(self):
        self.metricas["colisiones"] += 1
        self.mundo.detener()
        if self.estado not in ("FINALIZADA", "BACKFLIP_NO_DISPONIBLE",
                               "BACKFLIP_OMITIDO"):
            self.estado = "ESPERANDO_PERSONA"
            self.motivo = "Contacto físico detectado; movimiento detenido."
