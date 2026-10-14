"""Generación efímera de escenas Go2 con obstáculos físicos de MuJoCo."""
from __future__ import annotations

import os
import re
import tempfile
import warnings

# Texturas de banderas (PNG 14:9). Un objeto "bandera" del mapa elige una con
# el campo "textura"; si el archivo no existe se usa tela lisa y se avisa.
DIR_BANDERAS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "assets", "banderas")


def crear_escena(base_xml: str, mapa, incluir_bola: bool = False) -> tuple[str, list[str]]:
    """Crea MJCF junto al modelo para que los assets relativos sigan resolviendo.

    Los archivos temporales se eliminan después de compilar el MjModel. La
    escena original de Unitree y el XML limpio del paquete no se modifican.
    """
    directorio = os.path.dirname(os.path.abspath(base_xml))
    go2_xml = os.path.join(directorio, "go2.xml")
    with open(go2_xml, encoding="utf-8-sig") as archivo:
        robot_xml = archivo.read()
    raiz = re.search(r'(<body\s+name="base_link"(?=[\s/>])[^>]*>)', robot_xml)
    if not raiz:
        raise ValueError("No se encontró base_link en el modelo oficial Go2.")
    camara = ('\n      <camera name="go2_front_camera" pos="0.30 0 0.06" '
              'quat="0.5 0.5 -0.5 -0.5" fovy="60"/>')
    robot_xml = robot_xml[:raiz.end()] + camara + robot_xml[raiz.end():]
    robot_tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix="_go2_nav.xml", prefix=".",
        dir=directorio, delete=False)
    with robot_tmp:
        robot_tmp.write(robot_xml)
    temporales = [robot_tmp.name]

    with open(base_xml, encoding="utf-8-sig") as archivo:
        scene_xml = archivo.read()
    # Mejora la resolución offscreen del visor y de la cámara del robot sin
    # modificar la escena limpia ni los modelos oficiales.
    # znear mayor (0,05·extent) y zfar acotado: con znear=0,01·extent el buffer
    # de profundidad de 24 bits no resolvía las capas del piso (2-4 mm entre
    # baldosas, juntas y césped) a 15-30 m y parpadeaban. Menos "haze" para que
    # el fondo no se lave a esa misma distancia.
    scene_xml = scene_xml.replace(
        "<visual>", '<visual>\n    <quality shadowsize="2048"/>\n'
        '    <map znear="0.05" zfar="150" haze="0.12"/>', 1)
    scene_xml = scene_xml.replace(
        '<global azimuth="-130" elevation="-20"/>',
        '<global azimuth="-132" elevation="-24" offwidth="1280" '
        'offheight="720"/>', 1)
    if (getattr(mapa, "escenario", None) or {}).get("parque"):
        scene_xml = _mejorar_iluminacion(scene_xml)
    scene_xml = scene_xml.replace('<include file="go2.xml"/>',
                                  f'<include file="{os.path.basename(robot_tmp.name)}"/>')
    media = mapa.tamano_celda / 2
    obstaculos = []
    escenario = getattr(mapa, "escenario", {}) or {}
    assets_banderas, banderas_disponibles = _assets_banderas(escenario)
    celdas_decoradas = set()
    for fila in range(mapa.filas):
        for columna in range(mapa.columnas):
            if mapa.celda(fila, columna) != 1:
                continue
            celda = (fila, columna)
            decorado = next((obj for obj in escenario.get("objetos", [])
                             if tuple(obj.get("celda", ())) == celda), None)
            if decorado:
                celdas_decoradas.add(celda)
                obstaculos.extend(_geometria_decorado(
                    mapa, decorado, celda, banderas_disponibles))
                continue
            x, y = mapa.a_mundo(fila, columna)
            obstaculos.append(
                f'    <geom name="nav_obstacle_{fila}_{columna}" type="box" '
                f'size="{media:.5f} {media:.5f} 0.40" '
                f'pos="{x:.5f} {y:.5f} 0.40" '
                'rgba="0.72 0.18 0.18 1" contype="1" conaffinity="1"/>')
    pelota = escenario.get("pelota_oculta")
    if pelota is not None or incluir_bola:
        # En el mapa de misión, la celda de la pelota solo se usa para ubicar
        # la geometría en la escena: el planificador no recibe esa posición.
        celda_pelota = tuple(pelota) if pelota is not None else mapa.destino
        x, y = mapa.a_mundo(*celda_pelota)
        obstaculos.append(
            f'    <geom name="nav_target_ball" type="sphere" size="0.10" '
            f'pos="{x:.5f} {y:.5f} 0.10" rgba="0.95 0.10 0.08 1" '
            'contype="0" conaffinity="0"/>')
    obstaculos.extend(_superficies_escenario(mapa, escenario))
    obstaculos.extend(_personas_mjcf(mapa, escenario))
    obstaculos.extend(_personas_sentadas_mjcf(mapa, escenario))
    cierre = scene_xml.rfind("</worldbody>")
    if cierre < 0:
        for temporal in temporales:
            os.unlink(temporal)
        raise ValueError("La escena MJCF no contiene <worldbody>.")
    scene_xml = (scene_xml[:cierre] + "\n" + "\n".join(obstaculos)
                 + "\n  " + scene_xml[cierre:])
    if assets_banderas:
        fin_modelo = scene_xml.rfind("</mujoco>")
        scene_xml = (scene_xml[:fin_modelo] + "  <asset>\n"
                     + "\n".join(assets_banderas) + "\n  </asset>\n"
                     + scene_xml[fin_modelo:])
    scene_tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix="_go2_course.xml", prefix=".",
        dir=directorio, delete=False)
    with scene_tmp:
        scene_tmp.write(scene_xml)
    temporales.append(scene_tmp.name)
    return scene_tmp.name, temporales


# Retoques puramente visuales de la escena limpia (no tocan colisiones,
# coordenadas ni la percepción, que usa segmentación y profundidad): luz de sol
# inclinada con sombras en lugar de una luz cenital que quemaba el piso, cielo
# claro, horizonte neblinoso y suelo exterior verde mate sin reflejos (el
# reflejo del piso obligaba a renderizar la escena dos veces).
_AJUSTES_VISUALES = (
    ('<headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>',
     '<headlight diffuse="0.20 0.20 0.20" ambient="0.45 0.45 0.45" specular="0 0 0"/>'),
    ('<rgba haze="0.15 0.25 0.35 1"/>', '<rgba haze="0.74 0.82 0.90 1"/>'),
    ('rgb1="0.3 0.5 0.7" rgb2="0 0 0"', 'rgb1="0.36 0.58 0.86" rgb2="0.86 0.91 0.96"'),
    ('rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"',
     'rgb1="0.34 0.44 0.30" rgb2="0.31 0.41 0.28" markrgb="0.34 0.44 0.30"'),
    ('reflectance="0.2"', 'reflectance="0"'),
    ('<light pos="0 0 1.5" dir="0 0 -1" directional="true"/>',
     '<light name="sol" pos="0 0 12" dir="0.35 0.45 -1" directional="true" '
     'castshadow="true" diffuse="0.80 0.77 0.68" specular="0.05 0.05 0.05"/>'),
)


def _mejorar_iluminacion(scene_xml: str) -> str:
    """Aplica los ajustes que encuentre; una plantilla distinta no falla."""
    for viejo, nuevo in _AJUSTES_VISUALES:
        scene_xml = scene_xml.replace(viejo, nuevo, 1)
    return scene_xml


def _assets_banderas(escenario: dict) -> tuple[list[str], set[str]]:
    """<texture>/<material> para cada textura de bandera con archivo presente."""
    assets, disponibles = [], set()
    for objeto in escenario.get("objetos", []):
        nombre = objeto.get("textura") if objeto.get("tipo") == "bandera" else None
        if not nombre or nombre in disponibles:
            continue
        if not re.fullmatch(r"[A-Za-z0-9_]+", nombre):
            raise ValueError(f"Nombre de textura de bandera inválido: {nombre!r}")
        archivo = os.path.join(DIR_BANDERAS, f"bandera_{nombre}.png")
        if not os.path.isfile(archivo):
            warnings.warn(f"Falta la textura {archivo}; la bandera "
                          f"{nombre!r} se dibuja como tela lisa.")
            continue
        disponibles.add(nombre)
        ruta = archivo.replace("\\", "/")
        assets.append(f'    <texture name="tex_bandera_{nombre}" type="2d" '
                      f'file="{ruta}"/>')
        assets.append(f'    <material name="mat_bandera_{nombre}" '
                      f'texture="tex_bandera_{nombre}" specular="0" '
                      'shininess="0"/>')
    return assets, disponibles


def _geometria_decorado(mapa, objeto: dict, celda: tuple[int, int],
                        banderas_disponibles=frozenset()) -> list[str]:
    """Primitivas con nombres estables; todos los obstáculos tienen colisión."""
    tipo = objeto.get("tipo")
    fila, columna = celda
    x, y = mapa.a_mundo(fila, columna)
    nombre = f"nav_obstacle_{tipo}_{fila}_{columna}"
    medio = mapa.tamano_celda / 2
    solido = 'contype="1" conaffinity="1"'
    sin_colision = 'contype="0" conaffinity="0"'
    if tipo == "arbol":
        # Escala de las fotos: árboles de ~3 m en grandes macetas redondas.
        return [
            f'<geom name="{nombre}_maceta" type="cylinder" size="0.36 0.19" '
            f'pos="{x} {y} 0.20" rgba="0.27 0.20 0.15 1" {solido}/>',
            f'<geom name="{nombre}_tronco" type="cylinder" size="0.07 0.95" '
            f'pos="{x} {y} 1.02" rgba="0.34 0.20 0.10 1" {solido}/>',
            f'<geom name="{nombre}_copa" type="sphere" size="0.62" '
            f'pos="{x} {y} 2.25" rgba="0.14 0.42 0.16 1" {sin_colision}/>',
            f'<geom name="{nombre}_copa_alta" type="sphere" size="0.42" '
            f'pos="{x - 0.25} {y + 0.10} 2.78" '
            f'rgba="0.22 0.52 0.20 1" {sin_colision}/>',
            f'<geom name="{nombre}_copa_lateral" type="sphere" size="0.40" '
            f'pos="{x + 0.30} {y - 0.10} 2.55" '
            f'rgba="0.18 0.47 0.18 1" {sin_colision}/>',
        ]
    if tipo == "arbusto":
        return [f'<geom name="{nombre}" type="sphere" size="0.34" '
                f'pos="{x} {y} 0.30" rgba="0.18 0.43 0.18 1" {solido}/>']
    if tipo == "banco":
        return [
            f'<geom name="{nombre}_asiento" type="box" '
            f'size="{medio * 0.75} 0.16 0.05" pos="{x} {y} 0.43" '
            f'rgba="0.38 0.22 0.11 1" {solido}/>',
            f'<geom name="{nombre}_respaldo" type="box" '
            f'size="{medio * 0.75} 0.045 0.20" pos="{x} {y + 0.13} 0.66" '
            f'rgba="0.38 0.22 0.11 1" {solido}/>',
            f'<geom name="{nombre}_patas" type="box" '
            f'size="0.05 0.19 0.20" pos="{x} {y} 0.20" '
            f'rgba="0.17 0.17 0.18 1" {solido}/>',
        ]
    if tipo == "poste":
        # Farola negra de ~3,6 m con pantalla cuadrada y globo de luz.
        return [
            f'<geom name="{nombre}_base" type="cylinder" size="0.07 1.70" '
            f'pos="{x} {y} 1.72" rgba="0.12 0.12 0.13 1" {solido}/>',
            f'<geom name="{nombre}_zocalo" type="box" size="0.16 0.16 0.04" '
            f'pos="{x} {y} 0.05" rgba="0.12 0.12 0.13 1" {sin_colision}/>',
            f'<geom name="{nombre}_pantalla" type="box" size="0.30 0.30 0.05" '
            f'pos="{x} {y} 3.52" rgba="0.14 0.14 0.15 1" {sin_colision}/>',
            f'<geom name="{nombre}_luz" type="sphere" size="0.17" '
            f'pos="{x} {y} 3.36" rgba="0.97 0.93 0.78 1" {sin_colision}/>',
        ]
    if tipo == "columna":
        # Pilar de la pérgola: sólido y presente en la grilla (antes era solo
        # decorativo y la ruta de búsqueda lo atravesaba).
        return [
            f'<geom name="{nombre}_fuste" type="cylinder" size="0.11 1.50" '
            f'pos="{x} {y} 1.50" rgba="0.76 0.75 0.68 1" {solido}/>',
            f'<geom name="{nombre}_base" type="cylinder" size="0.17 0.05" '
            f'pos="{x} {y} 0.05" rgba="0.62 0.61 0.56 1" {sin_colision}/>',
        ]
    if tipo == "valla":
        return [
            f'<geom name="{nombre}_poste" type="box" '
            f'size="0.045 0.045 0.50" pos="{x} {y} 0.52" '
            f'rgba="0.28 0.31 0.32 1" {solido}/>',
            f'<geom name="{nombre}_baranda" type="box" '
            f'size="{medio * 0.92} 0.035 0.045" pos="{x} {y} 0.78" '
            f'rgba="0.28 0.31 0.32 1" {solido}/>',
        ]
    if tipo == "roca":
        return [
            f'<geom name="{nombre}_base" type="ellipsoid" size="0.23 0.18 0.15" '
            f'pos="{x} {y} 0.15" rgba="0.39 0.41 0.38 1" {solido}/>',
            f'<geom name="{nombre}_luz" type="ellipsoid" size="0.15 0.13 0.11" '
            f'pos="{x - 0.05} {y - 0.04} 0.27" '
            f'rgba="0.52 0.54 0.49 1" {sin_colision}/>',
        ]
    if tipo == "macetero":
        return [
            f'<geom name="{nombre}_base" type="box" size="0.43 0.43 0.30" '
            f'pos="{x} {y} 0.30" rgba="0.80 0.79 0.74 1" {solido}/>'] + [
            f'<geom name="{nombre}_planta_{i}" type="sphere" size="0.28" '
            f'pos="{x + dx:.3f} {y + dy:.3f} {z:.3f}" '
            f'rgba="{color} 1" {sin_colision}/>'
            for i, (dx, dy, z, color) in enumerate((
                (-0.18, 0.0, 0.85, "0.16 0.39 0.19"),
                (0.12, 0.08, 0.92, "0.22 0.48 0.22"),
                (0.0, -0.15, 0.98, "0.13 0.33 0.17"))) ]
    if tipo == "asiento_moderno":
        return [
            f'<geom name="{nombre}_base" type="ellipsoid" size="0.42 0.40 0.31" '
            f'pos="{x} {y} 0.31" rgba="0.96 0.96 0.94 1" {solido}/>']
    if tipo == "mesa":
        return [
            f'<geom name="{nombre}_tapa" type="cylinder" size="0.38 0.045" '
            f'pos="{x} {y} 0.63" rgba="0.86 0.85 0.80 1" {solido}/>'] + [
            f'<geom name="{nombre}_pie" type="cylinder" size="0.06 0.29" '
            f'pos="{x} {y} 0.31" rgba="0.35 0.37 0.36 1" {solido}/>']
    if tipo == "bandera":
        # Tela con proporción 14:9 (1,26 x 0,81 m) pegada al mástil, sin
        # colisión y a 3,3 m del piso: no interfiere con el robot ni su cámara.
        textura = objeto.get("textura")
        if textura in banderas_disponibles:
            aspecto = f'material="mat_bandera_{textura}" rgba="1 1 1 1"'
        else:
            aspecto = 'rgba="0.30 0.55 0.82 1"'
        return [
            f'<geom name="{nombre}_mastil" type="cylinder" size="0.035 2.1" '
            f'pos="{x} {y} 2.12" rgba="0.76 0.77 0.75 1" {solido}/> ',
            f'<geom name="{nombre}_tela" type="box" size="0.63 0.012 0.405" '
            f'pos="{x + 0.665:.3f} {y} 3.72" {aspecto} {sin_colision}/>']
    raise ValueError(f"Tipo de obstáculo desconocido en escenario: {tipo!r}")


def _superficies_escenario(mapa, escenario: dict) -> list[str]:
    """Parches visuales simples para distinguir césped y sendero del parque."""
    if not escenario.get("parque"):
        return []
    ancho = mapa.columnas * mapa.tamano_celda
    alto = mapa.filas * mapa.tamano_celda
    cx, cy = ancho / 2 - mapa.tamano_celda / 2, -alto / 2 + mapa.tamano_celda / 2
    if escenario.get("patio_uade"):
        return _superficies_patio(mapa, ancho, alto, cx, cy)
    camino_mitad = min(1.65, ancho * 0.12)
    cesped_mitad = max(0.35, (ancho / 2 - camino_mitad - 0.12) / 2)
    cesped_x = camino_mitad + cesped_mitad + 0.12
    return [
        f'<geom name="park_grass_west" type="box" size="{cesped_mitad:.3f} '
        f'{alto * 0.48:.3f} 0.006" pos="{cx - cesped_x:.3f} {cy:.3f} 0.006" '
        'rgba="0.24 0.42 0.20 1" contype="0" conaffinity="0"/>',
        f'<geom name="park_grass_east" type="box" size="{cesped_mitad:.3f} '
        f'{alto * 0.48:.3f} 0.006" pos="{cx + cesped_x:.3f} {cy:.3f} 0.006" '
        'rgba="0.24 0.42 0.20 1" contype="0" conaffinity="0"/>',
        f'<geom name="park_lawn_north" type="box" size="{ancho * 0.48:.3f} '
        f'1.20 0.006" pos="{cx:.3f} {cy + alto * 0.30:.3f} 0.006" '
        'rgba="0.31 0.49 0.24 1" contype="0" conaffinity="0"/>',
        f'<geom name="park_path" type="box" size="{camino_mitad:.3f} '
        f'{alto * 0.48:.3f} 0.008" pos="{cx:.3f} {cy:.3f} 0.008" '
        'rgba="0.58 0.56 0.48 1" contype="0" conaffinity="0"/>',
        f'<geom name="park_cross_path" type="box" size="{ancho * 0.48:.3f} '
        f'1.05 0.007" pos="{cx:.3f} {cy * 0.45:.3f} 0.007" '
        'rgba="0.50 0.49 0.43 1" contype="0" conaffinity="0"/>',
    ]


def _superficies_patio(mapa, ancho, alto, cx, cy) -> list[str]:
    """Explanada universitaria: baldosas, canteros verdes y senderos claros."""
    piezas = [
        f'<geom name="uade_plaza" type="box" size="{ancho * .50:.3f} '
        f'{alto * .50:.3f} 0.009" pos="{cx:.3f} {cy:.3f} 0.009" '
        'rgba="0.68 0.67 0.62 1" contype="0" conaffinity="0"/>']
    # Juntas finas de baldosas visibles en toda la explanada.
    for c in range(1, mapa.columnas):
        x = (c - .5) * mapa.tamano_celda
        piezas.append(f'<geom name="uade_junta_v_{c}" type="box" size="0.012 '
                      f'{alto * .49:.3f} 0.002" pos="{x:.3f} {cy:.3f} 0.020" '
                      'rgba="0.48 0.49 0.47 1" contype="0" conaffinity="0"/>')
    for f in range(1, mapa.filas):
        y = -(f - .5) * mapa.tamano_celda
        piezas.append(f'<geom name="uade_junta_h_{f}" type="box" size="{ancho * .49:.3f} '
                      f'0.012 0.002" pos="{cx:.3f} {y:.3f} 0.020" '
                      'rgba="0.48 0.49 0.47 1" contype="0" conaffinity="0"/>')
    piezas.extend([
        f'<geom name="uade_jardin_oeste" type="box" size="2.15 {alto * .44:.3f} 0.012" '
        f'pos="{cx - ancho * .35:.3f} {cy:.3f} 0.020" rgba="0.26 0.43 0.22 1" '
        'contype="0" conaffinity="0"/>',
        f'<geom name="uade_jardin_este" type="box" size="1.65 {alto * .36:.3f} 0.012" '
        f'pos="{cx + ancho * .36:.3f} {cy + alto * .02:.3f} 0.020" '
        'rgba="0.28 0.45 0.23 1" contype="0" conaffinity="0"/>',
        f'<geom name="uade_cesped_central" type="box" size="3.3 2.2 0.012" '
        f'pos="{cx:.3f} {cy + alto * .23:.3f} 0.020" rgba="0.32 0.50 0.25 1" '
        'contype="0" conaffinity="0"/>',
    ])
    piezas.extend(_arquitectura_patio(ancho, alto, cx, cy))
    return piezas


def _arquitectura_patio(ancho: float, alto: float, cx: float, cy: float) -> list[str]:
    """Fachadas UADE con paños de vidrio, balcones, pérgola y luminarias."""
    g = []
    # Las fachadas rodean la explanada sin invadir el área de circulación.
    for lado, x, y, sx, sy in (
        ("norte", cx, cy + alto / 2 + 1.2, ancho / 2 + 2.5, 0.55),
        ("este", cx + ancho / 2 + 1.2, cy, 0.55, alto / 2 + 2.5),
        ("oeste", cx - ancho / 2 - 1.2, cy, 0.55, alto / 2 + 2.5),
    ):
        g.append(f'<geom name="uade_edificio_{lado}" type="box" size="{sx:.2f} {sy:.2f} 4.2" '
                 f'pos="{x:.2f} {y:.2f} 4.2" rgba="0.72 0.74 0.72 1" '
                 'contype="0" conaffinity="0"/>')
        # Ventanales, balcones y franjas de color en las tres fachadas (antes
        # solo la norte los tenía y las otras dos eran bloques lisos).
        franjas = {1: "0.55 0.45 0.70", 3: "0.35 0.62 0.58"}   # lila / verde agua
        for piso, z in enumerate((1.35, 2.75, 4.15, 5.55, 6.95)):
            if lado == "norte":
                largo, centro, cara = ancho + 5.0, x, y - sy - .04
            else:
                largo, centro = alto + 5.0, y
                cara = x - sx - .04 if lado == "este" else x + sx + .04
            sentido = 1 if lado != "oeste" else -1
            horizontal = lado == "norte"
            tam_faja = (f"{largo / 2 * .98:.2f} 0.035 0.055" if horizontal
                        else f"0.035 {largo / 2 * .98:.2f} 0.055")
            pos_faja = (f"{centro:.2f} {cara:.2f} {z:.2f}" if horizontal
                        else f"{cara:.2f} {centro:.2f} {z:.2f}")
            color_faja = franjas.get(piso, "0.25 0.29 0.31")
            g.append(f'<geom name="uade_{lado}_faja_{piso}" type="box" size="{tam_faja}" '
                     f'pos="{pos_faja}" rgba="{color_faja} 1" contype="0" conaffinity="0"/>')
            if horizontal:
                tb, pb = f"{largo / 2 * .98:.2f} 0.22 0.045", (centro, cara - .18, z - .08)
            else:
                tb, pb = f"0.22 {largo / 2 * .98:.2f} 0.045", (cara - sentido * .18, centro, z - .08)
            g.append(f'<geom name="uade_{lado}_balcon_{piso}" type="box" size="{tb}" '
                     f'pos="{pb[0]:.2f} {pb[1]:.2f} {pb[2]:.2f}" rgba="0.37 0.40 0.39 1" '
                     'contype="0" conaffinity="0"/>')
            paso_v, n = 2.0, int(largo // 2.0)
            for k in range(1, n):
                off = -largo / 2 + k * paso_v
                if horizontal:
                    tv, pv = "0.72 0.025 0.43", (centro + off, cara - .03, z + .48)
                else:
                    tv, pv = "0.025 0.72 0.43", (cara - sentido * .03, centro + off, z + .48)
                g.append(f'<geom name="uade_{lado}_ventana_{piso}_{k}" type="box" size="{tv}" '
                         f'pos="{pv[0]:.2f} {pv[1]:.2f} {pv[2]:.2f}" rgba="0.12 0.24 0.34 1" '
                         'contype="0" conaffinity="0"/>')

    # Pérgola peatonal, como la galería de las fotos, sobre un lateral.
    px, py = cx + ancho * .156, cy + alto * .31   # entre las columnas 18 y 23
    g.append(f'<geom name="uade_pergola_techo" type="box" size="4.6 1.65 0.10" '
             f'pos="{px:.2f} {py:.2f} 3.0" rgba="0.83 0.82 0.74 1" '
             'contype="0" conaffinity="0"/>')
    for i in range(9):
        xx = px - 4.4 + i * 1.1
        g.append(f'<geom name="uade_pergola_viga_{i}" type="box" size="0.055 1.75 0.065" '
                 f'pos="{xx:.2f} {py:.2f} 3.18" rgba="0.84 0.83 0.75 1" '
                 'contype="0" conaffinity="0"/>')
    # Los pilares son objetos "columna" del mapa (colisión real); ver _geometria_decorado.
    return g


def _personas_mjcf(mapa, escenario: dict) -> list[str]:
    resultado = []
    for indice, persona in enumerate(escenario.get("personas", [])):
        fila, columna = persona["celda"]
        x, y = mapa.a_mundo(fila, columna)
        nombre = f"nav_person_{indice}"
        resultado.append(
            f'<body name="{nombre}" mocap="true" pos="{x} {y} 0">'
            f'<geom name="{nombre}_piernas" type="capsule" '
            'fromto="0 0 0.12 0 0 0.63" size="0.11" '
            'rgba="0.16 0.25 0.72 1" contype="1" conaffinity="1"/>'
            f'<geom name="{nombre}_torso" type="capsule" '
            'fromto="0 0 0.60 0 0 1.12" size="0.18" '
            'rgba="0.76 0.28 0.16 1" contype="1" conaffinity="1"/>'
            f'<geom name="{nombre}_cabeza" type="sphere" size="0.14" '
            'pos="0 0 1.29" rgba="0.80 0.63 0.48 1" '
            'contype="1" conaffinity="1"/></body>')
    return resultado


# (camisa, pantalón, piel, pelo) de cada variante de persona sentada.
_VARIANTES_PERSONA = (
    ("0.15 0.15 0.17", "0.12 0.14 0.20", "0.85 0.68 0.55", "0.16 0.10 0.07"),
    ("0.84 0.84 0.80", "0.20 0.28 0.45", "0.62 0.45 0.34", "0.07 0.06 0.06"),
    ("0.30 0.52 0.34", "0.30 0.28 0.25", "0.92 0.76 0.64", "0.45 0.30 0.14"),
    ("0.70 0.30 0.28", "0.10 0.10 0.12", "0.75 0.58 0.45", "0.10 0.08 0.06"),
)


def _personas_sentadas_mjcf(mapa, escenario: dict) -> list[str]:
    """Personas estáticas sentadas sobre bancos del mapa (solo ambientación).

    Cada entrada de ``personas_sentadas`` apunta a la celda de un ``banco``.
    La geometría no tiene colisión y su nombre (``deco_persona_*``) no entra en
    la segmentación como "persona": no detiene al Go2 ni cambia la
    planificación. El banco (y su celda ocupada en la grilla) ya bloquea el
    paso; las piernas llegan a ~0,5 m del centro del banco, dentro del margen
    de seguridad de A* (ver test_personas_sentadas_no_invaden_la_ruta).
    Medidas: asiento a 0,48 m, torso y cabeza hasta ~1,30 m (adulto sentado).
    El banco mira hacia -y (el respaldo está en +y).
    """
    bancos = {tuple(o["celda"]) for o in escenario.get("objetos", [])
              if o.get("tipo") == "banco"}
    geoms = []
    for i, persona in enumerate(escenario.get("personas_sentadas", [])):
        celda = tuple(persona["celda"])
        if celda not in bancos:
            raise ValueError(f"personas_sentadas[{i}]: la celda {list(celda)} "
                             "no tiene un banco.")
        camisa, pantalon, piel, pelo = _VARIANTES_PERSONA[
            int(persona.get("variante", i)) % len(_VARIANTES_PERSONA)]
        x, y = mapa.a_mundo(*celda)
        n = f"deco_persona_{i}"
        nc = 'contype="0" conaffinity="0"'
        for lado, dx in (("izq", -0.09), ("der", 0.09)):
            geoms += [
                f'<geom name="{n}_muslo_{lado}" type="capsule" '
                f'fromto="{x + dx:.3f} {y - 0.08:.3f} 0.56 {x + dx:.3f} {y - 0.40:.3f} 0.56" '
                f'size="0.075" rgba="{pantalon} 1" {nc}/>',
                f'<geom name="{n}_pierna_{lado}" type="capsule" '
                f'fromto="{x + dx:.3f} {y - 0.42:.3f} 0.52 {x + dx:.3f} {y - 0.43:.3f} 0.10" '
                f'size="0.06" rgba="{pantalon} 1" {nc}/>',
                f'<geom name="{n}_pie_{lado}" type="box" size="0.05 0.11 0.04" '
                f'pos="{x + dx:.3f} {y - 0.50:.3f} 0.04" rgba="0.10 0.10 0.11 1" {nc}/>',
                f'<geom name="{n}_brazo_{lado}" type="capsule" '
                f'fromto="{x + 2.3 * dx:.3f} {y - 0.08:.3f} 1.02 {x + 2.3 * dx:.3f} {y - 0.10:.3f} 0.74" '
                f'size="0.045" rgba="{camisa} 1" {nc}/>',
                f'<geom name="{n}_antebrazo_{lado}" type="capsule" '
                f'fromto="{x + 2.3 * dx:.3f} {y - 0.10:.3f} 0.74 {x + 1.3 * dx:.3f} {y - 0.30:.3f} 0.64" '
                f'size="0.04" rgba="{piel} 1" {nc}/>',
            ]
        geoms += [
            f'<geom name="{n}_torso" type="capsule" '
            f'fromto="{x:.3f} {y - 0.08:.3f} 0.62 {x:.3f} {y - 0.06:.3f} 1.00" '
            f'size="0.155" rgba="{camisa} 1" {nc}/>',
            f'<geom name="{n}_cuello" type="cylinder" size="0.05 0.05" '
            f'pos="{x:.3f} {y - 0.07:.3f} 1.20" rgba="{piel} 1" {nc}/>',
            f'<geom name="{n}_cabeza" type="sphere" size="0.105" '
            f'pos="{x:.3f} {y - 0.07:.3f} 1.31" rgba="{piel} 1" {nc}/>',
            f'<geom name="{n}_pelo" type="sphere" size="0.108" '
            f'pos="{x:.3f} {y - 0.045:.3f} 1.34" rgba="{pelo} 1" {nc}/>',
        ]
    return geoms


def limpiar_escena(temporales: list[str]) -> None:
    for ruta in temporales:
        try:
            os.unlink(ruta)
        except FileNotFoundError:
            pass


