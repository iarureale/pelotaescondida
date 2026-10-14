"""Levanta el simulador OFICIAL de Unitree, listo para el TP.

Que hace, en orden:

  1. Inyecta nuestra configuracion (robot, dominio, interfaz) donde el codigo
     oficial la espera. El bridge oficial hace `import config` y ramifica segun
     config.ROBOT para elegir los mensajes unitree_hg (G1) o unitree_go (Go2),
     asi que hay que dejarsela puesta ANTES de importarlo.
  2. Carga la escena oficial del robot elegido.
  3. Levanta el UnitreeSdk2Bridge OFICIAL: publica rt/lowstate y escucha
     rt/lowcmd, igual que contra el robot real.
  4. Levanta el servicio "sport", que el simulador oficial NO trae. Sin esto
     LocoClient.Move() da timeout: el controlador que hace caminar al G1 corre
     en la PC interna del robot y Unitree no lo publica.
  5. Corre el bucle y la ventana.

Sobre la fisica: el bucle usa mj_forward, no mj_step. El G1 es un humanoide y
SE CAE SOLO sin un controlador de locomocion; con fisica real se desploma antes
de que el alumno pruebe nada. La marcha de las patas es cosmetica.
"""

from __future__ import annotations

import math
import os
import sys
import time
import types



def _rellenar(msg, campo: str, valor) -> None:
    """Escribe un valor en todos los slots de una secuencia del IDL.

    Los campos Sequence[int] del IDL vienen con largo fijo: no se pueden
    reasignar con una lista, hay que escribir slot por slot.
    """
    try:
        seq = getattr(msg, campo)
    except AttributeError:
        return
    try:
        for i in range(len(seq)):
            seq[i] = int(valor())
    except (TypeError, ValueError, IndexError):
        pass


def preparar_config(robot, repo: str, domain: int, interfaz: str):
    """Deja un modulo `config` en sys.modules para el codigo oficial."""
    cfg = types.ModuleType("config")
    cfg.ROBOT = robot.clave
    # OJO: hay que pedirle la ruta al robot, NO armarla con un join.
    #
    # `ruta_escena()` GENERA la escena limpia del Go2 si todavia no existe
    # (`scene_uade.xml`, que no es un archivo del repo oficial). Armando la
    # ruta a mano, en una maquina donde nunca se genero, el simulador moria con
    # "No encuentro la escena oficial". En la de Teo no se veia porque el
    # archivo habia quedado escrito desde la primera vez.
    ruta = robot.ruta_escena()
    if not ruta:
        raise FileNotFoundError(
            f"No encuentro el modelo de {robot.clave} ni pude generar su "
            f"escena. Fijate que este 'entorno/sim/unitree_mujoco/"
            f"unitree_robots/{robot.clave}/' dentro de tu carpeta.")
    cfg.ROBOT_SCENE = ruta
    cfg.DOMAIN_ID = domain
    cfg.INTERFACE = interfaz
    cfg.USE_JOYSTICK = 0            # sin joystick: nadie tiene uno en clase
    cfg.JOYSTICK_TYPE = "xbox"
    cfg.JOYSTICK_DEVICE = 0
    cfg.PRINT_SCENE_INFORMATION = False
    cfg.ENABLE_ELASTIC_BAND = False  # no hace falta: no corremos fisica
    cfg.SIMULATE_DT = 0.005
    cfg.VIEWER_DT = 0.02
    sys.modules["config"] = cfg
    return cfg


class SimuladorOficial:
    def __init__(self, mundo, robot, repo: str, domain: int = 0,
                 interfaz: str = "lo", verboso: bool = True, mapa=None):
        self.mundo = mundo
        self.robot = robot
        self.mapa = mapa
        self._ultima_ruta = None
        self.repo = repo
        self.verboso = verboso

        self.cfg = preparar_config(robot, repo, domain, interfaz)
        if not os.path.exists(self.cfg.ROBOT_SCENE):
            raise FileNotFoundError(
                f"No encuentro la escena oficial:\n  {self.cfg.ROBOT_SCENE}")

        # El bridge oficial vive en el repo de Unitree.
        ruta_oficial = os.path.join(repo, "simulate_python")
        if ruta_oficial not in sys.path:
            sys.path.insert(0, ruta_oficial)

        import mujoco

        self.mj = mujoco
        self.seguir_pelota = bool(getattr(self.mundo, "seguir_pelota", False))
        self.mision_parque = bool(getattr(self.mundo, "mision_parque", False))
        self.obstaculos_fisicos = bool(
            self.robot.clave == "go2" and self.mapa is not None
            and (self.mapa.ruta or self.seguir_pelota or self.mision_parque))
        escena_temporal = []
        ruta_modelo = self.cfg.ROBOT_SCENE
        if self.obstaculos_fisicos:
            from .escena_navegacion import crear_escena
            ruta_modelo, escena_temporal = crear_escena(
                ruta_modelo, self.mapa, incluir_bola=True)
        try:
            self.model = mujoco.MjModel.from_xml_path(ruta_modelo)
        finally:
            if escena_temporal:
                from .escena_navegacion import limpiar_escena
                limpiar_escena(escena_temporal)
        self.data = mujoco.MjData(self.model)
        self._inicio_escenario = time.monotonic()
        self._personas_mocap = []
        if self.obstaculos_fisicos:
            for indice, persona in enumerate(
                    getattr(self.mapa, "escenario", {}).get("personas", [])):
                nombre = f"nav_person_{indice}"
                cuerpo = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, nombre)
                if cuerpo >= 0:
                    mocap = int(self.model.body_mocapid[cuerpo])
                    if mocap >= 0:
                        celdas = persona.get("trayectoria", [persona["celda"]])
                        puntos = [self.mapa.a_mundo(*p) for p in celdas]
                        self._personas_mocap.append((
                            mocap, puntos,
                            max(0.05, float(persona.get("velocidad_m_s", 0.45))),
                            bool(persona.get("loop", True))))
        self.camara = None
        self.vista_camara = None
        self.ultima_percepcion = []
        self._ultima_captura_camara = 0.0
        self.seguidor_pelota = None
        self.mision = None
        self._pedido_reinicio = False
        if self.mision_parque:
            self.mision = self._crear_mision()
        if self.seguir_pelota:
            from .camara_go2 import SeguidorPelota
            self.seguidor_pelota = SeguidorPelota(self.mundo)
        self.model.opt.timestep = self.cfg.SIMULATE_DT
        self.model.opt.gravity[:] = 0.0   # sin fisica, el robot no se cae

        self.bridge = None
        self.servicio = None

        # Telemetria para el TP05. Sin esto, el dashboard del alumno grafica
        # lineas planas: la cinematica no produce torque ni temperatura.
        from .telemetria import Telemetria

        self.telemetria = Telemetria(self.model.nu)
        self._qpos_previo = None

    # ---------- transporte ----------
    def iniciar_transporte(self):
        """Punto de enganche: `SimuladorLocal` lo reemplaza por el socket.

        Los bucles de `correr_*` llaman aca y no directo a `iniciar_dds`, asi
        el modo local reusa TODO el resto -- MuJoCo, el visor, la pose, la
        telemetria, la grilla -- sin duplicar una linea.
        """
        self.iniciar_dds()

    def iniciar_dds(self):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        ChannelFactoryInitialize(self.cfg.DOMAIN_ID, self.cfg.INTERFACE)

        # Bridge OFICIAL: rt/lowstate, rt/lowcmd, rt/sportmodestate
        from unitree_sdk2py_bridge import UnitreeSdk2Bridge

        self.bridge = UnitreeSdk2Bridge(self.model, self.data)
        if self.verboso:
            print("  [DDS] Bridge oficial activo: rt/lowstate, rt/lowcmd")

        # El bridge oficial publica SOLO en rt/lowstate. El robot real publica
        # ademas en rt/lf/lowstate, y varios laboratorios escuchan ahi -- el
        # TP05 entre ellos. Sin este espejo, el dashboard del alumno no recibe
        # nada contra el simulador y si contra el robot: la diferencia
        # aparecería recien el dia de la visita.
        self._espejo_lowstate()

        # Servicio de locomocion, que el oficial NO trae.
        if self.robot.clave == "g1":
            from .servicio_sport import ServicioSportSimulado

            self.servicio = ServicioSportSimulado(self.mundo, verboso=self.verboso)
            if self.verboso:
                print("  [DDS] Servicio de locomocion activo: LocoClient responde")
        else:
            from .servicio_sport_go2 import ServicioSportGo2

            self.servicio = ServicioSportGo2(self.mundo, verboso=self.verboso)
            if self.verboso:
                print("  [DDS] Servicio de locomocion activo: SportClient responde")

    def _espejo_lowstate(self) -> None:
        """Republica el LowState en rt/lf/lowstate, como hace el robot real."""
        import threading

        from unitree_sdk2py.core.channel import ChannelPublisher

        if self.robot.clave == "g1":
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        else:
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

        pub = ChannelPublisher("rt/lf/lowstate", LowState_)
        pub.Init()
        self._pub_espejo = pub

        def bucle():
            while True:
                low = getattr(self.bridge, "low_state", None)
                if low is not None:
                    # La inclinacion se aplica ACA, justo antes de publicar: el
                    # bridge oficial reescribe el cuaternion desde sensordata en
                    # su propio hilo, asi que el ultimo en escribir tiene que
                    # ser el espejo. El adaptador del TP05 calcula roll y pitch
                    # a partir del cuaternion, no de rpy.
                    roll, pitch = getattr(self, "_inclinacion", (0.0, 0.0))
                    yaw = getattr(self, "_yaw", 0.0)
                    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
                    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
                    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
                    try:
                        q = low.imu_state.quaternion
                        q[0] = cr * cp * cy + sr * sp * sy
                        q[1] = sr * cp * cy - cr * sp * sy
                        q[2] = cr * sp * cy + sr * cp * sy
                        q[3] = cr * cp * sy - sr * sp * cy
                        low.imu_state.rpy[0] = roll
                        low.imu_state.rpy[1] = pitch
                        low.imu_state.rpy[2] = yaw
                        pub.Write(low)
                    except Exception:
                        pass
                time.sleep(0.02)   # 50 Hz, como el bridge oficial

        threading.Thread(target=bucle, daemon=True).start()
        if self.verboso:
            print("  [DDS] Espejo activo: rt/lf/lowstate (lo usa el TP05)")
        self._publicar_bateria()

    def _publicar_bateria(self) -> None:
        """Publica el estado de la bateria en rt/lf/bmsstate.

        En el G1 la bateria NO viaja dentro del LowState_: `unitree_hg` no
        tiene `bms_state` (eso es del `unitree_go`). El robot real la manda en
        un BmsState_ por su propio topico, y el dashboard del TP05 se quedaba
        con el panel de bateria en cero para siempre.

        La bateria es INVENTADA -- el simulador es cinematico y no consume
        nada real -- pero baja despacio mientras el robot se mueve, que es lo
        que hace falta para que un grafico de SOC tenga algo que mostrar.
        """
        import threading

        from unitree_sdk2py.core.channel import ChannelPublisher

        if self.robot.clave == "g1":
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import BmsState_
            from unitree_sdk2py.idl.default import (
                unitree_hg_msg_dds__BmsState_ as _bms_vacio)
        else:
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import BmsState_
            from unitree_sdk2py.idl.default import (
                unitree_go_msg_dds__BmsState_ as _bms_vacio)

        # BmsState_() NO se puede construir vacio: el IDL exige los 13 campos
        # como argumentos posicionales. Hay que pedirle uno a la fabrica de
        # `idl.default` y despues mutarlo. Es la misma trampa que con
        # MotorState_; si se construye a mano, el TypeError queda tapado por el
        # try/except del hilo y la bateria no se publica NUNCA, sin un error.
        try:
            msg = _bms_vacio()
            pub = ChannelPublisher("rt/lf/bmsstate", BmsState_)
            pub.Init()
        except Exception as exc:
            if self.verboso:
                print(f"  [DDS] No pude abrir rt/lf/bmsstate: {exc}")
            return
        self._pub_bms = pub

        def bucle_bms():
            aviso_dado = False
            while True:
                try:
                    soc = int(self.telemetria.bateria)
                    msg.soc = soc
                    msg.soh = 98
                    # Corriente negativa = descarga, como reporta el robot.
                    moviendose = bool(self.mundo.leer().get("moviendose"))
                    msg.current = -1200 if moviendose else -300
                    _rellenar(msg, "cell_vol", lambda: 3700 + (soc - 50) * 4)
                    _rellenar(msg, "temperature", lambda: 30)
                    pub.Write(msg)
                except Exception as exc:
                    if not aviso_dado and self.verboso:
                        aviso_dado = True
                        print(f"  [DDS] Fallo la publicacion de bateria: "
                              f"{type(exc).__name__}: {exc}")
                time.sleep(0.5)   # 2 Hz: una bateria no cambia mas rapido

        threading.Thread(target=bucle_bms, daemon=True).start()
        if self.verboso:
            print("  [DDS] Bateria activa: rt/lf/bmsstate (lo usa el TP05)")

    # ---------- misión ----------
    def _crear_mision(self):
        from .mision_go2 import MisionParque
        from .robots import HUELLA
        largo, ancho = HUELLA[self.robot.clave]
        margen = math.hypot(largo / 2, ancho / 2) + 0.05
        return MisionParque(self.mundo, self.mapa, margen)

    def reiniciar_mision(self):
        """Vuelve al inicio y repite el recorrido desde cero (tecla R).

        El mundo es cinemático y determinista: misma pose inicial + misma
        lista de waypoints = mismo recorrido.
        """
        if not self.mision_parque:
            return
        self.mundo.detener()
        self.mundo.reiniciar()          # vuelve a la pose fijada con situar()
        self.mapa.ruta = []
        self._inicio_escenario = time.monotonic()
        self._estado_mision_reportado = None
        self._metricas_mision_reportadas = False
        self._contacto_reportado = False
        self._ultima_captura_camara = 0.0
        self.ultima_percepcion = []
        self.mision = self._crear_mision()
        print("\n  [MISION] Reiniciada: el Go2 vuelve al inicio y repite el recorrido.")

    def _tecla(self, keycode):
        """Callback del visor (hilo de la GUI): sólo levanta una bandera."""
        if keycode in (ord("R"), ord("r")):
            self._pedido_reinicio = True

    # ---------- pose ----------
    def _escribir_pose(self):
        e = self.mundo.leer()
        q = self.data.qpos

        q[0] = e["x"]
        q[1] = e["y"]
        q[2] = self.robot.altura + e.get("altura", 0.0)
        media_yaw = e["yaw"] / 2.0
        fase_marcha = e["fase"] * e.get(
            "cadencia_marcha", self.robot.multiplicador_cadencia_marcha)
        pitch_carrera = (self.robot.inclinacion_marcha_rad * math.sin(fase_marcha)
                         if e["moviendose"] else 0.0)
        if e["accion"] == "backflip_visual":
            # Gesto de presentación cinemático: arco y pitch completos de 360°.
            # No integra fuerzas ni se envía por SportClient al robot real.
            progreso = e.get("gesto_progreso", 0.0)
            pitch = -2.0 * math.pi * progreso
            media_pitch = pitch / 2.0
            q[2] += 0.58 * math.sin(math.pi * progreso) ** 2
            q[3] = math.cos(media_yaw) * math.cos(media_pitch)
            q[4] = -math.sin(media_yaw) * math.sin(media_pitch)
            q[5] = math.cos(media_yaw) * math.sin(media_pitch)
            q[6] = math.sin(media_yaw) * math.cos(media_pitch)
        else:
            if e["moviendose"]:
                q[2] += self.robot.rebote_marcha_m * (
                    0.5 - 0.5 * math.cos(2.0 * fase_marcha))
            media_pitch = pitch_carrera / 2.0
            q[3] = math.cos(media_yaw) * math.cos(media_pitch)
            q[4] = -math.sin(media_yaw) * math.sin(media_pitch)
            q[5] = math.cos(media_yaw) * math.sin(media_pitch)
            q[6] = math.sin(media_yaw) * math.cos(media_pitch)

        art = q[7:]
        art[:] = 0.0
        for idx, valor in self.robot.pose_de_pie.items():
            if idx < len(art):
                art[idx] = valor

        if not e.get("de_pie", True):
            for idx, valor in self.robot.pose_sentado.items():
                if idx < len(art):
                    art[idx] = valor
        elif e["accion"] in ("saludando", "besando"):
            for idx, valor in self.robot.saludo.items():
                if idx < len(art):
                    art[idx] = valor
        elif e["moviendose"]:
            f = fase_marcha
            for idx, amplitud, desfase in self.robot.marcha:
                if idx < len(art):
                    art[idx] += amplitud * math.sin(f + desfase)

        # Velocidad de las articulaciones por diferencia finita. Es REAL: las
        # articulaciones se estan moviendo de verdad, aunque no haya fisica.
        # De aca salen tambien el gyro y la velocidad de motor que publica el
        # bridge oficial.
        ahora = time.perf_counter()
        if self._qpos_previo is not None:
            dt = ahora - self._qpos_previo[0]
            if dt > 1e-6:
                self.data.qvel[6:] = (q[7:] - self._qpos_previo[1]) / dt
                self.data.qvel[:6] = 0.0
            else:
                self.data.qvel[:] = 0.0
        else:
            self.data.qvel[:] = 0.0
        self._qpos_previo = (ahora, q[7:].copy())

        self._actualizar_personas()
        self.mj.mj_forward(self.model, self.data)
        if self.obstaculos_fisicos:
            self._detener_si_hay_contacto()
        self._completar_telemetria(e)

    def _actualizar_personas(self):
        """Mueve personas mocap por rutas declaradas en el escenario JSON."""
        if not self._personas_mocap:
            return
        transcurrido = time.monotonic() - self._inicio_escenario
        for mocap, puntos, velocidad, loop in self._personas_mocap:
            if len(puntos) < 2:
                if puntos:
                    self.data.mocap_pos[mocap, :2] = puntos[0]
                continue
            tramos = []
            largo = 0.0
            for inicio, fin in zip(puntos, puntos[1:]):
                distancia = math.hypot(fin[0] - inicio[0], fin[1] - inicio[1])
                tramos.append((inicio, fin, distancia, largo))
                largo += distancia
            if largo <= 1e-9:
                continue
            recorrido = transcurrido * velocidad
            restante = recorrido % largo if loop else min(recorrido, largo)
            for inicio, fin, distancia, acumulado in tramos:
                if restante <= acumulado + distancia:
                    fraccion = (restante - acumulado) / max(distancia, 1e-9)
                    self.data.mocap_pos[mocap, 0] = inicio[0] + (fin[0] - inicio[0]) * fraccion
                    self.data.mocap_pos[mocap, 1] = inicio[1] + (fin[1] - inicio[1]) * fraccion
                    self.data.mocap_pos[mocap, 2] = 0.0
                    break

    def _paso_seguimiento_visual(self):
        if time.monotonic() - self._ultima_captura_camara < 0.10:
            return
        self._ultima_captura_camara = time.monotonic()
        try:
            if self.camara is None:
                from .camara_go2 import CamaraGo2
                self.camara = CamaraGo2(self.model)
            frame, profundidad, segmentacion = self.camara.capturar_sensores(self.data)
            if self.vista_camara is None:
                from .camara_go2 import VistaCamaraTk
                self.vista_camara = VistaCamaraTk()
                self.vista_camara.iniciar()
            from .camara_go2 import (DeteccionPelota, detectar_objetos_segmentados,
                                     validar_pelota_con_rgb)
            percibidas = detectar_objetos_segmentados(
                segmentacion, profundidad, self.model, self.mj)
            self.ultima_percepcion, descartadas = validar_pelota_con_rgb(
                percibidas, frame)
            if descartadas:
                self._pelotas_descartadas = getattr(self, "_pelotas_descartadas", 0) + descartadas
                if self._pelotas_descartadas in (1, 10, 100, 1000):
                    print(f"\n  [CAMARA] Pelota de la segmentación descartada por no "
                          f"coincidir con la imagen RGB (total {self._pelotas_descartadas}).")
            if self.mision is not None:
                estado = self.mision.paso(self.ultima_percepcion)
                estado_mostrar = f"MISION: {estado}"
                if estado != getattr(self, "_estado_mision_reportado", None):
                    print(f"\n  [MISION] {estado}: {self.mision.motivo}")
                    self._estado_mision_reportado = estado
                    if (estado in ("FINALIZADA", "ERROR", "BACKFLIP_SIMULADO",
                                   "BACKFLIP_NO_DISPONIBLE",
                                   "BACKFLIP_OMITIDO")
                            and not getattr(self, "_metricas_mision_reportadas", False)):
                        print(f"  [MISION] Métricas: {self.mision.metricas}")
                        self._metricas_mision_reportadas = True
            elif self.seguidor_pelota is not None:
                pelota = next((obj for obj in self.ultima_percepcion
                               if obj.categoria == "pelota"), None)
                deteccion = (None if pelota is None else DeteccionPelota(
                    pelota.centro_x, pelota.centro_y, pelota.bearing_rad,
                    pelota.distancia_m, pelota.pixeles))
                self.seguidor_pelota.paso(deteccion)
                estado_mostrar = f"SEGUIDOR: {self.seguidor_pelota.estado}"
                if deteccion is None and not getattr(self, "_pelota_perdida_reportada", False):
                    print("\n  [CAMARA] Pelota no visible; Go2 detenido por seguridad.")
                    self._pelota_perdida_reportada = True
                elif deteccion is not None:
                    self._pelota_perdida_reportada = False
            pose_actual = self.mundo.leer()
            estado_mostrar += (f" | Go2 {pose_actual['accion']}: "
                               f"{pose_actual['velocidad_m_s']:.1f} m/s")
            self.vista_camara.publicar(
                frame, self.ultima_percepcion, estado=estado_mostrar)
        except Exception as exc:
            self.mundo.detener()
            if self.mision is not None:
                self.mision.estado = "ERROR"
                self.mision.motivo = str(exc)
            elif self.seguidor_pelota is not None:
                self.seguidor_pelota.estado = "error"
            print(f"\n  [CAMARA] Error de captura/detección: {exc}; Go2 detenido.")
            self.seguidor_pelota = None
            self.mision = None
    def _detener_si_hay_contacto(self):
        """Detiene el mundo si la pose cinemática entra en un obstáculo MJCF."""
        for indice in range(self.data.ncon):
            contacto = self.data.contact[indice]
            pares = ((int(contacto.geom1), int(contacto.geom2)),
                     (int(contacto.geom2), int(contacto.geom1)))
            for obstaculo, otro in pares:
                nombre = self.mj.mj_id2name(
                    self.model, self.mj.mjtObj.mjOBJ_GEOM, obstaculo)
                if (nombre and (nombre.startswith("nav_obstacle_")
                                or nombre.startswith("nav_person_"))
                        and int(self.model.geom_bodyid[otro]) != 0):
                    self.mundo.detener()
                    if self.mision is not None:
                        if not getattr(self, "_contacto_reportado", False):
                            self.mision.registrar_colision()
                    if not getattr(self, "_contacto_reportado", False):
                        print(f"\n  [SEGURIDAD] Contacto con {nombre}; Go2 detenido.")
                    self._contacto_reportado = True
                    return
        self._contacto_reportado = False
    # ---------- bucles ----------
    def correr_con_ventana(self):
        """Abre la ventana 3D. Si no se puede, sigue en consola.

        Que el import de MuJoCo funcione NO garantiza que se pueda abrir una
        ventana: hace falta un driver con OpenGL 3.3. En una maquina virtual sin
        GPU, o por escritorio remoto, `launch_passive` muere con

            GLFWError: (65542) WGL: The driver does not appear to support OpenGL
            ERROR: could not create window

        y hasta ahora eso mataba el simulador entero. No tiene por que: **el TP
        se puede hacer completo sin ver el robot**, y perder la ventana es mucho
        mejor que perder la clase.
        """
        try:
            self._correr_con_ventana()
        except Exception as exc:                              # noqa: BLE001
            print()
            print("*" * 62)
            print("  NO SE PUDO ABRIR LA VENTANA 3D")
            print("*" * 62)
            print(f"  Motivo: {type(exc).__name__}: {str(exc).splitlines()[0]}")
            print()
            print("  Casi siempre es una de estas:")
            print("    - una maquina virtual sin GPU configurada")
            print("    - una sesion por escritorio remoto")
            print("    - drivers de video sin soporte de OpenGL 3.3")
            # El caso de conda es aparte y da un error muy particular:
            # conda trae su propio glfw y su propio libstdc++, que chocan con
            # los del sistema. La pista es "EGL" en el mensaje.
            if "EGL" in str(exc) and os.environ.get("CONDA_PREFIX"):
                print("    - **conda**: trae su propio glfw y choca con el del")
                print("      sistema. Es lo mas probable en tu caso.")
                print()
                print("  Si queres la ventana 3D, proba con el Python del")
                print("  sistema en vez del de conda:")
                print("      conda deactivate")
                print("      python3 -m pip install --user mujoco")
                print()
            print("  EL SIMULADOR SIGUE FUNCIONANDO, en modo consola:")
            print("  el programa del alumno corre igual y el recorrido del")
            print("  robot se dibuja en texto. El TP se puede hacer completo.")
            print("*" * 62)
            print()
            self.correr_sin_ventana()

    def _correr_con_ventana(self):
        import mujoco.viewer

        with mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False,
            key_callback=self._tecla
        ) as v:
            seguir_go2 = getattr(self.mundo, "velocidad_autonoma_max", None) is not None
            proximo_estado_go2 = 0.0
            v.cam.distance = 3.5 if self.robot.tipo == "humanoide" else 2.6
            v.cam.elevation = -20
            if self.mapa is not None:
                from .dibujo import dibujar
                # Los obstáculos físicos ya están en el MJCF cuando se usa A*.
                dibujar(v.user_scn, self.mapa,
                            mostrar_obstaculos=not self.obstaculos_fisicos)
                v.cam.distance = max(
                    3.0, (1.25 if getattr(self.mapa, "escenario", {}).get("patio_uade")
                          else 0.9) * max(self.mapa.filas, self.mapa.columnas)
                    * self.mapa.tamano_celda)
                cx = (self.mapa.columnas - 1) / 2 * self.mapa.tamano_celda
                cy = -(self.mapa.filas - 1) / 2 * self.mapa.tamano_celda
                v.cam.lookat[:] = [cx, cy, 0.0]
                if seguir_go2:
                    # El patio nuevo es mucho mayor que la escena limpia. Una
                    # vista fija deja al robot diminuto y puede parecer quieto.
                    # Cámara de seguimiento cercana: con el Go2 de 0,7 m, a
                    # 10-18 m ocupaba ~30 px y parecía "perderse". A 6 m se lo
                    # ve con claridad y el patio sigue a la vista; la rueda
                    # del mouse permite alejar o acercar.
                    v.cam.distance = 6.0
                    v.cam.elevation = -30
                    v.cam.azimuth = -40
                    pose0 = self.mundo.leer()
                    cam_xy = [pose0["x"], pose0["y"]]
            self.iniciar_transporte()
            periodo = self.cfg.VIEWER_DT
            t_cam = time.perf_counter()
            if 'cam_xy' not in dir():
                cam_xy = [0.0, 0.0]
            while v.is_running():
                inicio = time.perf_counter()
                if self._pedido_reinicio:
                    self._pedido_reinicio = False
                    self.reiniciar_mision()
                    cam_xy = [self.mundo.leer()["x"], self.mundo.leer()["y"]]
                self.mundo.avanzar()
                self._escribir_pose()
                if self.mision is not None:
                    self.mision.control()    # lazo rápido de seguimiento de ruta
                if seguir_go2:
                    pose = self.mundo.leer()
                    # Seguimiento suavizado: filtra el rebote de la marcha y
                    # evita que la vista "salte" con cada ajuste de rumbo.
                    k = 1.0 - math.exp(-(inicio - t_cam) / 0.25)
                    t_cam = inicio
                    cam_xy[0] += (pose["x"] - cam_xy[0]) * k
                    cam_xy[1] += (pose["y"] - cam_xy[1]) * k
                    v.cam.lookat[:] = [cam_xy[0], cam_xy[1], 0.25]
                    ahora = time.monotonic()
                    if ahora >= proximo_estado_go2:
                        print(f"  [GO2] {pose['accion']} | "
                              f"{pose['velocidad_m_s']:.1f} m/s | "
                              f"x={pose['x']:.1f}, y={pose['y']:.1f}")
                        proximo_estado_go2 = ahora + 2.0
                if self.seguidor_pelota is not None or self.mision is not None:
                    self._paso_seguimiento_visual()
                self._volcar_avisos()
                if self.mapa is not None and self._ruta_cambio():
                    from .dibujo import dibujar
                    dibujar(v.user_scn, self.mapa,
                            mostrar_obstaculos=not self.obstaculos_fisicos)
                v.sync()
                resto = periodo - (time.perf_counter() - inicio)
                if resto > 0:
                    time.sleep(resto)

    def correr_sin_ventana(self):
        """El bucle cuando no hay ventana 3D. Dibuja el recorrido en texto.

        Antes esto no mostraba NADA: la consola quedaba muda mientras el robot
        se movia. El alumno veia su programa imprimir "Avanzando..." y del otro
        lado, silencio -- y no tenia forma de saber si el simulador lo estaba
        escuchando. Ver `consola.py`.
        """
        from .consola import VistaConsola

        vista = VistaConsola(verboso=self.verboso) if self.verboso else None
        self.iniciar_transporte()
        while True:
            if self._pedido_reinicio:
                self._pedido_reinicio = False
                self.reiniciar_mision()
            self.mundo.avanzar()
            self._escribir_pose()
            if self.mision is not None:
                self.mision.control()
            if self.seguidor_pelota is not None or self.mision is not None:
                self._paso_seguimiento_visual()
            self._volcar_avisos()
            if vista is not None:
                vista.actualizar(self.mundo.leer())
            time.sleep(self.cfg.VIEWER_DT)

    def _completar_telemetria(self, estado) -> None:
        """Rellena lo que la cinematica no produce. Ver telemetria.py."""
        import numpy as np

        nu = self.model.nu
        moviendose = bool(estado.get("moviendose"))
        fase = (float(estado.get("fase", 0.0))
                * float(estado.get("cadencia_marcha",
                                   self.robot.multiplicador_cadencia_marcha)))

        # Esfuerzo por motor: cuanto se esta moviendo cada articulacion.
        vel = np.abs(self.data.qvel[6:6 + nu])
        esfuerzos = np.clip(vel / 2.0, 0.0, 1.0).tolist()

        # Las piernas cargan mas peso que los brazos.
        cargas = [1.0 if i < nu // 2 else 0.45 for i in range(nu)]

        self.telemetria.actualizar(esfuerzos, moviendose)
        torques = self.telemetria.torque(esfuerzos, cargas)

        # El bridge oficial lee el torque de sensordata (mjSENS_JOINTACTFRC),
        # que sin fisica queda en cero. Se escribe ahi para no tocar el bridge
        # ni el laboratorio: la ruta de la telemetria queda igual.
        base_torque = 2 * nu
        for i, t in enumerate(torques):
            if base_torque + i < len(self.data.sensordata):
                self.data.sensordata[base_torque + i] = t

        # La inclinacion se guarda siempre: la usan los dos transportes.
        self._inclinacion = self.telemetria.inclinacion(fase, moviendose)
        self._yaw = float(estado.get("yaw", 0.0))
        self._moviendose = moviendose
        self._fase = fase

        self._publicar_telemetria(estado, nu, fase, moviendose)

    def _publicar_telemetria(self, estado, nu, fase, moviendose) -> None:
        """Escribe la telemetria en el LowState_ que esta por salir por DDS.

        Solo aplica al transporte DDS. En el modo local no hay LowState_: la
        telemetria se sirve como diccionario por el socket.
        """
        low = getattr(self.bridge, "low_state", None)
        if low is None:
            return
        temps = self.telemetria.temperaturas
        for i in range(min(nu, len(low.motor_state))):
            m = low.motor_state[i]
            try:
                if hasattr(m.temperature, "__len__"):
                    m.temperature[0] = int(temps[i])
                    if len(m.temperature) > 1:
                        m.temperature[1] = int(temps[i])
                else:
                    m.temperature = int(temps[i])
            except (TypeError, ValueError, IndexError):
                pass

        # La inclinacion la aplica el espejo al publicar, no aca: el bridge
        # oficial reescribe el cuaternion desde sensordata en su propio hilo,
        # DESPUES de esto, y se perderia.

        if hasattr(low, "foot_force"):
            for i, f in enumerate(self.telemetria.fuerzas_de_pata(fase, 4, moviendose)):
                if i < len(low.foot_force):
                    low.foot_force[i] = f
        if hasattr(low, "bms_state"):
            try:
                low.bms_state.soc = int(self.telemetria.bateria)
            except (TypeError, ValueError):
                pass
        if hasattr(low, "power_v"):
            try:
                low.power_v = 28.0 + (self.telemetria.bateria / 100.0) * 4.0
            except (TypeError, ValueError):
                pass

    _ultima_ruta = None

    def _ruta_cambio(self) -> bool:
        actual = tuple(tuple(p) for p in self.mapa.ruta) if self.mapa.ruta else ()
        if actual != self._ultima_ruta:
            self._ultima_ruta = actual
            return True
        return False

    def _volcar_avisos(self):
        if self.mundo.avisos and self.verboso:
            for aviso in self.mundo.avisos:
                print(f"  [LIMITE] {aviso}")
            self.mundo.avisos.clear()
