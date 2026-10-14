"""Máquina de estados para buscar pelota con cámara y A* en el parque MuJoCo."""
from __future__ import annotations

import copy
import math
import time

from .camara_go2 import SeguidorPelota
from .navegador_astar import NavegadorAstar
from .planificacion_go2 import SinRuta, astar, ruta_densa, suavizar_ruta


BACKFLIP_CINEMATICO_DISPONIBLE = True


class MisionParque:
    """Explora waypoints conocidos; la celda de la pelota no se usa al planificar."""

    # Una pelota a más de esta distancia se alcanza con A*; más cerca, con el
    # control reactivo de aproximación final.
    DISTANCIA_FINAL_M = 2.5
    # Cuadros consecutivos para aceptar una pelota lejana (evita falsos
    # positivos de un solo píxel en el borde del frustum).
    CONFIRMACIONES_LEJOS = 4
    # Tope de seguridad (s) sin ver la pelota, medido desde la última vez que se
    # la vio, antes de olvidarla aunque siga viajando. En el caso normal la
    # memoria se descarta al llegar a la última posición y completar un barrido.
    PERDIDA_MAX_S = 90.0
    # Holgura extra (m) que deben respetar los atajos en línea recta: absorbe el
    # error de seguimiento (~5 cm) para que el robot nunca roce el margen de A*.
    HOLGURA_ATAJO_M = 0.10
    # Duración (s) del barrido de 360° en la última posición conocida. El perfil
    # limita el giro a 0,5 rad/s: 2π/14 s = 0,45 rad/s completa la vuelta.
    BARRIDO_PELOTA_S = 14.0

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
        self._barrido_pelota_desde = None
        self._pelota_muestras = []
        self._pelota_est = None
        self._destino_pelota = None
        self._t_control = 0.0
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

    @staticmethod
    def _xy_deteccion(pose, objeto):
        """Proyecta rango y bearing medidos por cámara a metros de mundo."""
        # MuJoCo entrega depth axial y el eje horizontal de esta cámara está
        # invertido respecto del eje Y del robot (imagen derecha = robot -Y).
        frente = 0.30 + objeto.distancia_m
        lateral = -objeto.distancia_m * math.tan(objeto.bearing_rad)
        yaw = pose["yaw"]
        x = pose["x"] + frente * math.cos(yaw) - lateral * math.sin(yaw)
        y = pose["y"] + frente * math.sin(yaw) + lateral * math.cos(yaw)
        return x, y

    def _celda_deteccion(self, pose, objeto):
        """Celda de la grilla donde cae una detección de cámara."""
        return self._celda_pose(*self._xy_deteccion(pose, objeto))

    # ---------- memoria de la pelota ----------
    def _registrar_pelota(self, pose, pelota):
        """Promedia las últimas estimaciones: una sola lectura lejana es ruidosa."""
        self._pelota_muestras.append(self._xy_deteccion(pose, pelota))
        del self._pelota_muestras[:-6]
        n = len(self._pelota_muestras)
        self._pelota_est = (sum(p[0] for p in self._pelota_muestras) / n,
                            sum(p[1] for p in self._pelota_muestras) / n)

    def _planificar_hacia_pelota(self, pose):
        """A* hacia la celda libre más cercana a la pelota estimada."""
        if self._pelota_est is None:
            raise SinRuta("Sin estimación de la pelota.")
        destino = self._celda_pose(*self._pelota_est)
        inicio = self._celda_pose(pose["x"], pose["y"])
        ruta = astar(self.mapa, inicio=inicio, destino=destino,
                     margen_m=self.margen_m)
        vertices = suavizar_ruta(self.mapa, ruta, self.margen_m + self.HOLGURA_ATAJO_M)
        self.navegador = NavegadorAstar(self.mundo, self.mapa, vertices)
        self.ruta = ruta_densa(vertices)      # lo que se dibuja = lo que se sigue
        self.mapa.ruta = self.ruta
        self._destino_pelota = destino
        self._ultima_replanificacion = time.monotonic()
        self.metricas["replanificaciones"] += 1

    def _olvidar_pelota(self):
        self._barrido_pelota_desde = None
        self._pelota_est = None
        self._pelota_muestras = []
        self._destino_pelota = None
        self._confirmaciones_pelota = 0
        self._pelota_perdida_desde = None

    def control(self):
        """Lazo rápido (cada ciclo del visor): sólo ejecuta el seguimiento de ruta.

        Separado de ``paso`` (que depende del render de la cámara, ~10 Hz o
        menos) para que las órdenes de velocidad no venzan entre capturas y el
        robot no avance a tirones.
        """
        ahora = time.monotonic()
        if ahora - self._t_control < 0.005:
            return          # ya se ejecutó en este ciclo
        if (self.navegador is not None and self.estado in (
                "NAVEGANDO", "EXPLORANDO", "REPLANIFICANDO",
                "APROXIMANDO_PELOTA")):
            self._t_control = ahora
            self.navegador.paso()

    def _mapa_temporal(self, detecciones, pose):
        if not any(o.categoria == "persona" for o in detecciones):
            return self.mapa      # mapa estático: sin copia profunda por replan
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
        vertices = suavizar_ruta(temporal, ruta, self.margen_m + self.HOLGURA_ATAJO_M)
        self.navegador = NavegadorAstar(self.mundo, temporal, vertices)
        self.ruta = ruta_densa(vertices)
        self.mapa.ruta = self.ruta
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

    def paso(self, detecciones, pose_captura=None):
        if self.estado in ("FINALIZADA", "ERROR", "BACKFLIP_NO_DISPONIBLE",
                           "BACKFLIP_OMITIDO", "BACKFLIP_SIMULADO"):
            return self.estado
        ahora = time.monotonic()
        pose = self.mundo.leer()
        # Pose del robot EN EL INSTANTE del cuadro: con la cámara en otro hilo el
        # robot avanzó desde la captura y proyectar con la pose actual corría la
        # pelota estimada hasta ~0,9 m a 3 m/s.
        pose_cam = pose_captura or pose
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
                    if self._celda_deteccion(pose_cam, persona) in ruta_actual:
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
            self._barrido_pelota_desde = None
            self._registrar_pelota(pose_cam, pelota)
            self._confirmaciones_pelota += 1
            # Histéresis: una vez en el tramo final no se vuelve a A* por una
            # lectura de rango que oscila alrededor del umbral.
            umbral = self.DISTANCIA_FINAL_M + (
                0.7 if self.estado in ("ACERCANDOSE", "DETECTANDO_PELOTA") else 0.0)
            cerca = pelota.distancia_m <= umbral
            necesarias = 2 if cerca else self.CONFIRMACIONES_LEJOS
            if self._confirmaciones_pelota >= necesarias:
                if cerca:
                    # Tramo final: control reactivo a la vista de la pelota.
                    self.estado = "ACERCANDOSE"
                    self.motivo = "Pelota a la vista; aproximación final."
                    estado_seguidor = self.seguidor.paso(pelota)
                    if estado_seguidor == "distancia_objetivo":
                        self.mundo.detener()
                        self.estado = "VERIFICANDO_MANIOBRA"
                    return self.estado
                # Pelota lejana confirmada: ir con A* (esquiva obstáculos) en
                # lugar de perseguirla en línea recta y perderla tras un árbol.
                refrescar = (self.estado != "APROXIMANDO_PELOTA"
                             or self.navegador is None
                             or self._celda_pose(*self._pelota_est)
                             != self._destino_pelota
                             and ahora - self._ultima_replanificacion > 2.0)
                if refrescar:
                    try:
                        self._planificar_hacia_pelota(pose)
                    except SinRuta:
                        pelota = None
                    else:
                        self.motivo = "Pelota detectada; acercándome con A*."
                if pelota is not None:
                    self.estado = "APROXIMANDO_PELOTA"
                    return self.estado
            elif cerca:
                self.mundo.detener()
                self.estado = "DETECTANDO_PELOTA"
                return self.estado
            # Lejos y aún sin confirmar: sigue la ruta sin frenar. Si ya iba hacia
            # la pelota, sigue ese camino (no es un fin de waypoint ni una pérdida).
            if self.estado == "APROXIMANDO_PELOTA":
                return self.estado
        else:
            self._confirmaciones_pelota = 0

        if self.estado == "ESCANEANDO":
            duracion = max(0.5, float((getattr(self.mapa, "escenario", {}) or {})
                                      .get("escaneo_duracion_s", 7.0)))
            if ahora - (self._escaneo_desde or ahora) < duracion:
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
        if pelota is None and self.estado in (
                "ACERCANDOSE", "DETECTANDO_PELOTA", "APROXIMANDO_PELOTA"):
            # Pelota fuera de vista (tapada por un árbol, borde del frustum...).
            if self._pelota_perdida_desde is None:
                self._pelota_perdida_desde = ahora
            perdida = ahora - self._pelota_perdida_desde
            if self._pelota_est is not None and perdida < self.PERDIDA_MAX_S:
                # Conserva la memoria: sigue hacia la última posición conocida
                # en vez de oscilar entre "buscar" y "perseguir".
                if (self.estado != "APROXIMANDO_PELOTA"
                        or self.navegador is None):
                    try:
                        self._planificar_hacia_pelota(pose)
                    except SinRuta:
                        self._olvidar_pelota()
                    else:
                        self.estado = "APROXIMANDO_PELOTA"
                        self.motivo = ("Pelota fuera de vista; voy a la última "
                                       "posición conocida.")
                        return self.estado
                if self.estado == "APROXIMANDO_PELOTA":
                    if self.navegador.estado == "atascado":
                        pass                       # cae a retomar la ruta
                    elif self.navegador.estado != "completado":
                        # Todavía viajando: la pelota puede quedar fuera del
                        # cuadro durante el trayecto (árboles, giros). No
                        # corre ningún tiempo de espera hasta llegar.
                        return self.estado
                    else:
                        # Llegó a la última posición conocida sin verla: barrido
                        # completo de 360° ahí mismo antes de rendirse.
                        if self._barrido_pelota_desde is None:
                            self._barrido_pelota_desde = ahora
                            self.mundo.detener()
                        if ahora - self._barrido_pelota_desde < self.BARRIDO_PELOTA_S:
                            self.mundo.set_velocidad(
                                0.0, 0.0, 2 * math.pi / self.BARRIDO_PELOTA_S,
                                duracion=0.15)
                            self.motivo = ("En la última posición conocida; "
                                           "barrido de 360° para reubicarla.")
                            return self.estado
            self.mundo.detener()
            self._olvidar_pelota()
            self.estado = "REPLANIFICANDO"
            self.motivo = "Pelota no reacquirida; retomo la ruta de búsqueda."
            try:
                self._planificar(detecciones, pose)
            except SinRuta:
                self.estado = "ERROR"
                self.motivo = "No se puede alcanzar el siguiente sector de búsqueda."
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

        elif (any(d.categoria == "persona" for d in detecciones)
              or self.navegador is None
              or self.navegador.estado == "atascado") \
                and ahora - self._ultima_replanificacion >= 0.5:
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

        self.control()
        if self.navegador and self.navegador.estado == "completado":
            self.mundo.detener()
            indice_busqueda = int((getattr(self.mapa, "escenario", {}) or {})
                                  .get("detectar_pelota_desde_waypoint", 0))
            if self.indice_waypoint >= indice_busqueda:
                # Llegó al waypoint declarado: barrido visual ahí mismo (también
                # en el último). El índice avanza UNA sola vez, al terminar el
                # barrido. Antes avanzaba al llegar y otra vez tras el barrido,
                # y el robot se saltaba un waypoint de cada dos.
                self._escaneo_desde = ahora
                self.estado = "ESCANEANDO"
                self.motivo = "Barrido visual del sector antes de continuar la ruta."
                return self.estado
            if self.indice_waypoint + 1 < len(self.waypoints):
                self.indice_waypoint += 1
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
