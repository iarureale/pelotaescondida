"""Captura MuJoCo y seguimiento visual básico de la pelota roja del escenario."""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct
import zlib


@dataclass(frozen=True)
class DeteccionPelota:
    centro_x: float
    centro_y: float
    bearing_rad: float
    distancia_m: float
    pixeles: int


@dataclass(frozen=True)
class ObjetoPercibido:
    """Detección a partir de máscara de cámara y profundidad MuJoCo."""
    nombre: str
    categoria: str
    centro_x: float
    centro_y: float
    bearing_rad: float
    distancia_m: float
    pixeles: int
    x_min: int = 0
    x_max: int = 0
    y_min: int = 0
    y_max: int = 0


def detectar_pelota(frame, diametro_m: float = 0.20,
                    fovy_deg: float = 60.0, pixeles_minimos: int = 20):
    """Segmenta el marcador rojo del render RGB y estima bearing y distancia.

    La distancia usa el diámetro conocido del marcador y su tamaño aparente;
    es una aproximación pinhole, útil en este escenario controlado.
    """
    import numpy as np

    imagen = np.asarray(frame)
    if imagen.ndim != 3 or imagen.shape[2] < 3:
        raise ValueError("La imagen debe tener formato RGB de tres canales.")
    alto, ancho = imagen.shape[:2]
    rgb = imagen[:, :, :3].astype(np.int16, copy=False)
    rojo, verde, azul = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    mascara = (rojo > 145) & (rojo > verde * 1.45) & (rojo > azul * 1.45)
    ys, xs = np.nonzero(mascara)
    if len(xs) < pixeles_minimos:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    centro_x = (x0 + x1) / 2.0
    centro_y = (y0 + y1) / 2.0
    diametro_px = max(1.0, ((x1 - x0 + 1) + (y1 - y0 + 1)) / 2.0)
    focal_px = alto / (2.0 * math.tan(math.radians(fovy_deg) / 2.0))
    distancia = diametro_m * focal_px / diametro_px
    bearing = math.atan2(centro_x - (ancho - 1) / 2.0, focal_px)
    return DeteccionPelota(centro_x, centro_y, bearing, distancia, len(xs))


class CamaraGo2:
    """Renderer de MuJoCo para la cámara fija al cuerpo del Go2."""

    def __init__(self, model, ancho: int = 640, alto: int = 480,
                 nombre: str = "go2_front_camera"):
        import mujoco

        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, nombre) < 0:
            raise ValueError(f"El modelo no tiene la cámara {nombre!r}.")
        self.renderer = mujoco.Renderer(model, height=alto, width=ancho)
        self.nombre = nombre

    def capturar(self, data):
        self.renderer.update_scene(data, camera=self.nombre)
        return self.renderer.render()

    def capturar_sensores(self, data):
        """Entrega RGB, profundidad métrica y segmentación por geometría.

        La etiqueta de segmentación es un canal del render de MuJoCo. Las
        detecciones se filtran por sus píxeles; no se consultan poses de
        objetos para decidir si son visibles ni para estimar su distancia.
        """
        self.renderer.update_scene(data, camera=self.nombre)
        self.renderer.enable_depth_rendering()
        profundidad = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        segmentacion = self._renderizar_ids_geometria()
        self.renderer.update_scene(data, camera=self.nombre)
        rgb = self.renderer.render().copy()
        return rgb, profundidad, segmentacion

    def _renderizar_ids_geometria(self):
        """Renderiza ID-color y lo convierte a (geom_id, tipo) por píxel.

        En MuJoCo 3.15, Renderer.render() remapea IDs globales contra la
        cantidad de geoms visibles en la cámara. Si esos IDs tienen huecos,
        un objeto visible puede exceder la tabla y lanzar IndexError. El
        buffer ID-color conserva el ID global; lo decodificamos directamente.
        """
        import mujoco
        import numpy as np

        escena = self.renderer.scene
        flags_originales = escena.flags.copy()
        try:
            escena.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = True
            escena.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = True
            self.renderer.disable_segmentation_rendering()
            self.renderer.disable_depth_rendering()
            raw = self.renderer.render().copy()
        finally:
            np.copyto(escena.flags, flags_originales)

        rgb = raw.astype(np.uint32, copy=False)
        ids = (rgb[:, :, 0] + rgb[:, :, 1] * (2 ** 8)
               + rgb[:, :, 2] * (2 ** 16)).astype(np.int64) - 1
        segmentacion = np.full((*ids.shape, 2), -1, dtype=np.int32)
        visibles = (ids >= 0) & (ids < self.renderer._model.ngeom)
        segmentacion[:, :, 0][visibles] = ids[visibles].astype(np.int32)
        segmentacion[:, :, 1][visibles] = int(mujoco.mjtObj.mjOBJ_GEOM)
        return segmentacion

    def close(self):
        self.renderer.close()


_TABLAS_CATEGORIAS: dict = {}


def _tabla_categorias(model, mujoco):
    """(nombres, categorias, es_objeto) por geom_id, calculada una vez por modelo.

    El modelo no cambia durante una misión; recorrer los ~400 geoms con
    mj_id2name en cada cuadro era trabajo repetido.
    """
    import numpy as np

    clave = id(model)
    cacheada = _TABLAS_CATEGORIAS.get(clave)
    if cacheada is not None and cacheada[0] is model:
        return cacheada[1]
    nombres = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
               for i in range(model.ngeom)]
    categorias = [_categoria_geometria(n) for n in nombres]
    es_objeto = np.array([c is not None for c in categorias], dtype=bool)
    if len(_TABLAS_CATEGORIAS) >= 4:
        _TABLAS_CATEGORIAS.pop(next(iter(_TABLAS_CATEGORIAS)))
    tabla = (nombres, categorias, es_objeto)
    _TABLAS_CATEGORIAS[clave] = (model, tabla)
    return tabla


def detectar_objetos_segmentados(segmentacion, profundidad, model, mujoco,
                                 fovy_deg: float = 60.0,
                                 pixeles_minimos: int = 8):
    """Detecta objetos etiquetados en píxeles y obtiene rango del depth map.

    Una sola pasada sobre los píxeles etiquetados (agrupados por geom_id) en
    lugar de una máscara de imagen completa por cada geom: mismo resultado,
    costo proporcional a los píxeles de objetos y no a ngeom x píxeles.
    """
    import numpy as np

    seg = np.asarray(segmentacion)
    depth = np.asarray(profundidad)
    if seg.ndim != 3 or seg.shape[2] < 2 or depth.shape[:2] != seg.shape[:2]:
        raise ValueError("Segmentación/depth incompatibles con la imagen de cámara.")
    alto, ancho = seg.shape[:2]
    focal_px = alto / (2.0 * math.tan(math.radians(fovy_deg) / 2.0))
    tipo_geom = int(mujoco.mjtObj.mjOBJ_GEOM)
    nombres, categorias, es_objeto = _tabla_categorias(model, mujoco)

    ids = seg[:, :, 0].ravel()
    de_geom = seg[:, :, 1].ravel() == tipo_geom
    candidato = de_geom & (ids >= 0) & (ids < len(es_objeto))
    candidato[candidato] = es_objeto[ids[candidato]]
    indices = np.flatnonzero(candidato)
    if not len(indices):
        return []
    ids_c = ids[indices]
    orden = np.argsort(ids_c, kind="stable")
    indices, ids_c = indices[orden], ids_c[orden]
    inicios = np.flatnonzero(np.r_[True, ids_c[1:] != ids_c[:-1]])
    fines = np.r_[inicios[1:], len(ids_c)]
    depth_plano = depth.reshape(-1)

    detecciones = []
    for a, b in zip(inicios, fines):
        if b - a < pixeles_minimos:
            continue
        geom_id = int(ids_c[a])
        px = indices[a:b]
        distancias = depth_plano[px]
        validas = distancias[np.isfinite(distancias) & (distancias > 0)]
        if not len(validas):
            continue
        xs, ys = px % ancho, px // ancho
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        detecciones.append(ObjetoPercibido(
            nombre=nombres[geom_id], categoria=categorias[geom_id],
            centro_x=float(cx), centro_y=float(cy),
            bearing_rad=math.atan2(cx - (ancho - 1) / 2.0, focal_px),
            distancia_m=float(np.median(validas)),
            pixeles=int(b - a), x_min=x0, x_max=x1, y_min=y0, y_max=y1))
    return detecciones


def validar_pelota_con_rgb(detecciones, rgb, diametro_m: float = 0.20,
                           fovy_deg: float = 60.0):
    """Descarta "pelotas" de la segmentación que la imagen RGB no respalda.

    La segmentación decodifica el ID de geometría desde un buffer de color. En
    algunas GPU/drivers los bordes de objetos cercanos (p. ej. el propio cuerpo
    del Go2) salen mezclados y pueden decodificar exactamente el ID de la
    pelota: una "pelota" falsa a pocos centímetros de la cámara. La pelota real
    es una esfera roja de 0,20 m, así que se exige (1) píxeles rojos dentro de
    su caja y (2) que el diámetro aparente sea coherente con la profundidad.
    Devuelve (detecciones_validas, cantidad_descartada).
    """
    import numpy as np

    imagen = np.asarray(rgb)
    alto = imagen.shape[0]
    focal_px = alto / (2.0 * math.tan(math.radians(fovy_deg) / 2.0))
    validas, descartadas = [], 0
    for obj in detecciones:
        if obj.categoria != "pelota":
            validas.append(obj)
            continue
        caja = imagen[obj.y_min:obj.y_max + 1, obj.x_min:obj.x_max + 1, :3]
        caja = caja.astype(np.int16, copy=False)
        r, g, b = caja[:, :, 0], caja[:, :, 1], caja[:, :, 2]
        rojos = int(np.count_nonzero((r > 120) & (r > g * 1.3) & (r > b * 1.3)))
        diametro_px = ((obj.x_max - obj.x_min + 1) + (obj.y_max - obj.y_min + 1)) / 2.0
        esperado_m = diametro_m * focal_px / max(1.0, diametro_px)
        coherente = 0.5 <= esperado_m / max(1e-6, obj.distancia_m) <= 2.0
        if rojos >= max(3, 0.2 * obj.pixeles) and coherente:
            validas.append(obj)
        else:
            descartadas += 1
    return validas, descartadas


def _categoria_geometria(nombre: str | None) -> str | None:
    if not nombre:
        return None
    if nombre == "nav_target_ball":
        return "pelota"
    if nombre.startswith("nav_person_"):
        return "persona"
    if nombre.startswith("nav_obstacle_arbol"):
        return "arbol"
    if nombre.startswith("nav_obstacle_arbusto"):
        return "arbusto"
    if nombre.startswith("nav_obstacle_banco"):
        return "banco"
    if nombre.startswith("nav_obstacle_poste"):
        return "poste"
    if nombre.startswith("nav_obstacle_valla"):
        return "valla"
    if nombre.startswith("nav_obstacle_"):
        return "obstaculo"
    return None


class SeguidorPelota:
    """Aproximación final continua a la pelota; si la pierde, velocidad cero.

    Avanza MIENTRAS corrige el rumbo (arco) y solo gira en el lugar si el error
    angular es grande; la velocidad se frena de forma proporcional a la distancia
    que falta y se suaviza con rampas. Las órdenes duran 0,4 s para no vencer
    entre cuadros de cámara lentos (antes 0,15 s: el robot arrancaba y paraba).
    """

    ACEL_M_S2 = 1.8           # aceleración al arrancar
    DECEL_M_S2 = 3.0          # al frenar
    GIRO_EN_SITIO_RAD = 0.9   # por encima, gira sin avanzar

    def __init__(self, mundo, distancia_objetivo_m: float = 0.75):
        self.mundo = mundo
        self.distancia_objetivo_m = distancia_objetivo_m
        self.estado = "esperando"
        self.ultima_deteccion = None
        self._v = None
        self._t = None

    def _velocidad_actual(self) -> float:
        return float(getattr(self.mundo, "vx", 0.0))

    def _rampa(self, objetivo: float) -> float:
        import time
        ahora = time.monotonic()
        dt = 0.1 if self._t is None else max(0.02, min(0.3, ahora - self._t))
        self._t = ahora
        v = self._velocidad_actual() if self._v is None else self._v
        paso = (self.ACEL_M_S2 if abs(objetivo) > abs(v) else self.DECEL_M_S2) * dt
        v = max(v - paso, min(v + paso, objetivo))
        self._v = v
        return v

    def paso(self, deteccion: DeteccionPelota | None) -> str:
        self.ultima_deteccion = deteccion
        if deteccion is None:
            self.mundo.detener()
            self.estado = "objetivo_perdido"
            self._v = None
            return self.estado
        self.estado = "siguiendo"
        bearing = deteccion.bearing_rad
        error = deteccion.distancia_m - self.distancia_objetivo_m
        limite_giro = (getattr(self.mundo, "velocidad_angular_autonoma_max", None)
                       or self.mundo.perfil.velocidad_angular_max)
        limite = (getattr(self.mundo, "velocidad_autonoma_max", None)
                  or self.mundo.perfil.velocidad_max)
        # Pixel a la derecha => bearing positivo; girar a la derecha = yaw negativo.
        giro = max(-limite_giro, min(limite_giro, -1.8 * bearing)) if abs(bearing) > 0.04 else 0.0
        if abs(error) <= 0.12 and abs(bearing) <= 0.12:
            self.mundo.detener()
            self._v = None
            self.estado = "distancia_objetivo"
            return self.estado
        if error > 0.12:
            frenado = min(limite, max(0.12, 1.5 * error))
            alineacion = max(0.0, 1.0 - abs(bearing) / self.GIRO_EN_SITIO_RAD)
            objetivo = frenado * alineacion
        elif error < -0.12:
            objetivo = -0.06
        else:
            objetivo = 0.0                     # a distancia: solo alinear el rumbo
        v = self._rampa(objetivo)
        self.mundo.mover_autonomo(v, 0.0, giro, duracion=0.4)
        return self.estado


class VistaCamaraTk:
    """Vista auxiliar opcional; usa Tkinter estándar y recibe el último frame."""

    # La captura y la detección siguen a 640x480; solo se achica lo que se
    # MUESTRA (factor 2 -> 320x240). Menos píxeles que codificar a PNG en el
    # hilo del visor por cada cuadro y una ventana más chica en pantalla.
    FACTOR_VISTA = 2
    MAX_DETALLES = 4          # detecciones más cercanas escritas en la barra

    def __init__(self, factor: int | None = None):
        import queue
        self.factor = max(1, int(factor or self.FACTOR_VISTA))
        self._queue = queue.Queue(maxsize=1)
        self._thread = None
        self._activa = False
        self.error = None

    def iniciar(self):
        import threading
        if self._thread is not None:
            return
        self._activa = True
        self._thread = threading.Thread(target=self._bucle, name="go2-camera-preview",
                                        daemon=True)
        self._thread.start()

    def _bucle(self):
        try:
            import tkinter as tk
            root = tk.Tk()
            root.title("Go2 — cámara simulada")
            ancho_vista, alto_vista = 640 // self.factor, 480 // self.factor
            root.geometry(f"{ancho_vista + 20}x{alto_vista + 90}")
            label = tk.Label(root, text="Esperando imagen de la cámara")
            label.pack(fill="both", expand=True)
            estado = tk.Label(root, text="", anchor="w", justify="left",
                              wraplength=ancho_vista)
            estado.pack(fill="x")

            def actualizar():
                if not self._activa:
                    root.destroy()
                    return
                try:
                    contenido = self._queue.get_nowait()
                except Exception:
                    contenido = None
                if contenido is not None:
                    try:
                        datos_png, texto_estado = contenido
                        imagen = tk.PhotoImage(data=datos_png, format="PNG")
                        label.configure(image=imagen, text="")
                        label.image = imagen
                        estado.configure(text=texto_estado)
                    except Exception as exc:
                        self.error = exc
                        label.configure(text=f"Error al mostrar cámara:\n{exc}")
                        print(f"\n  [CAMARA] Error al mostrar el frame: {exc}")
                root.after(40, actualizar)

            root.protocol("WM_DELETE_WINDOW", lambda: self._cerrar(root))
            root.after(40, actualizar)
            root.mainloop()
        except Exception as exc:
            self.error = exc
            self._activa = False

    def publicar(self, frame, detecciones=(), estado=""):
        if not self._activa:
            return
        import base64
        import numpy as np
        imagen = np.asarray(frame, dtype=np.uint8)[:, :, :3]
        colores = {"pelota": (255, 40, 30), "persona": (255, 220, 30),
                   "arbol": (40, 255, 90), "arbusto": (40, 255, 90),
                   "banco": (30, 220, 255), "poste": (30, 220, 255),
                   "valla": (30, 220, 255), "obstaculo": (255, 80, 220)}
        for deteccion in detecciones:
            x0 = max(0, min(imagen.shape[1] - 1, int(deteccion.x_min)))
            x1 = max(0, min(imagen.shape[1] - 1, int(deteccion.x_max)))
            y0 = max(0, min(imagen.shape[0] - 1, int(deteccion.y_min)))
            y1 = max(0, min(imagen.shape[0] - 1, int(deteccion.y_max)))
            if x1 < x0 or y1 < y0:
                continue
            color = colores.get(deteccion.categoria, (255, 255, 255))
            grosor = 2 * self.factor     # queda de 2 px tras reducir
            imagen[y0:min(y0 + grosor, y1 + 1), x0:x1 + 1] = color
            imagen[max(y1 - grosor + 1, y0):y1 + 1, x0:x1 + 1] = color
            imagen[y0:y1 + 1, x0:min(x0 + grosor, x1 + 1)] = color
            imagen[y0:y1 + 1, max(x1 - grosor + 1, x0):x1 + 1] = color
        if self.factor > 1:
            imagen = np.ascontiguousarray(imagen[::self.factor, ::self.factor])
        alto, ancho = imagen.shape[:2]
        filas = b"".join(b"\x00" + imagen[fila].tobytes()
                         for fila in range(alto))

        def chunk(tipo, datos):
            contenido = tipo + datos
            return (struct.pack(">I", len(datos)) + contenido
                    + struct.pack(">I", zlib.crc32(contenido) & 0xffffffff))

        png = (b"\x89PNG\r\n\x1a\n"
               + chunk(b"IHDR", struct.pack(">IIBBBBB", ancho, alto, 8, 2,
                                               0, 0, 0))
               + chunk(b"IDAT", zlib.compress(filas, 1))
               + chunk(b"IEND", b""))
        contenido = base64.b64encode(png).decode("ascii")
        try:
            self._queue.get_nowait()
        except Exception:
            pass
        try:
            cercanas = sorted(detecciones, key=lambda d: d.distancia_m)
            detalle = ", ".join(
                f"{d.categoria} {d.distancia_m:.1f} m"
                for d in cercanas[:self.MAX_DETALLES])
            texto = " | ".join(part for part in (estado, detalle) if part)
            self._queue.put_nowait((contenido, texto))
        except Exception:
            pass

    def _cerrar(self, root):
        self._activa = False
        root.destroy()
