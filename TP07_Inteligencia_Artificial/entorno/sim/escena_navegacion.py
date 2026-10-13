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
    scene_xml = scene_xml.replace(
        "<visual>", '<visual>\n    <quality shadowsize="2048"/>', 1)
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
        return [
            f'<geom name="{nombre}_tronco" type="cylinder" size="0.12 0.48" '
            f'pos="{x} {y} 0.50" rgba="0.34 0.18 0.08 1" {solido}/>',
            f'<geom name="{nombre}_copa" type="sphere" size="0.46" '
            f'pos="{x} {y} 1.16" rgba="0.10 0.34 0.13 1" {sin_colision}/>',
            f'<geom name="{nombre}_copa_alta" type="sphere" size="0.31" '
            f'pos="{x - 0.19} {y + 0.08} 1.44" '
            f'rgba="0.16 0.42 0.17 1" {sin_colision}/>',
            f'<geom name="{nombre}_copa_lateral" type="sphere" size="0.30" '
            f'pos="{x + 0.22} {y - 0.08} 1.34" '
            f'rgba="0.13 0.38 0.15 1" {sin_colision}/>',
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
        return [
            f'<geom name="{nombre}_base" type="cylinder" size="0.07 0.75" '
            f'pos="{x} {y} 0.78" rgba="0.18 0.20 0.22 1" {solido}/>',
            f'<geom name="{nombre}_luz" type="sphere" size="0.14" '
            f'pos="{x} {y} 1.58" rgba="0.96 0.83 0.40 1" {sin_colision}/>',
            f'<geom name="{nombre}_brazo" type="capsule" '
            f'fromto="{x} {y} 1.42 {x + 0.22} {y} 1.57" size="0.035" '
            f'rgba="0.18 0.20 0.22 1" {solido}/>',
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
            f'<geom name="{nombre}_base" type="box" size="0.43 0.43 0.22" '
            f'pos="{x} {y} 0.23" rgba="0.70 0.67 0.59 1" {solido}/>'] + [
            f'<geom name="{nombre}_planta_{i}" type="sphere" size="0.28" '
            f'pos="{x + dx:.3f} {y + dy:.3f} {z:.3f}" '
            f'rgba="{color} 1" {sin_colision}/>'
            for i, (dx, dy, z, color) in enumerate((
                (-0.18, 0.0, 0.70, "0.16 0.39 0.19"),
                (0.12, 0.08, 0.76, "0.22 0.48 0.22"),
                (0.0, -0.15, 0.82, "0.13 0.33 0.17"))) ]
    if tipo == "asiento_moderno":
        return [
            f'<geom name="{nombre}_base" type="ellipsoid" size="0.46 0.39 0.22" '
            f'pos="{x} {y} 0.24" rgba="0.91 0.91 0.87 1" {solido}/>']
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
        f'<geom name="uade_jardin_oeste" type="box" size="2.15 {alto * .44:.3f} 0.008" '
        f'pos="{cx - ancho * .35:.3f} {cy:.3f} 0.012" rgba="0.26 0.43 0.22 1" '
        'contype="0" conaffinity="0"/>',
        f'<geom name="uade_jardin_este" type="box" size="1.65 {alto * .36:.3f} 0.008" '
        f'pos="{cx + ancho * .36:.3f} {cy + alto * .02:.3f} 0.012" '
        'rgba="0.28 0.45 0.23 1" contype="0" conaffinity="0"/>',
        f'<geom name="uade_cesped_central" type="box" size="3.3 2.2 0.008" '
        f'pos="{cx:.3f} {cy + alto * .23:.3f} 0.012" rgba="0.32 0.50 0.25 1" '
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
        if lado == "norte":
            for piso, z in enumerate((1.35, 2.75, 4.15, 5.55, 6.95)):
                g.append(f'<geom name="uade_frente_faja_{piso}" type="box" '
                         f'size="{sx * .98:.2f} 0.035 0.055" pos="{x:.2f} {y - sy - .04:.2f} '
                         f'{z:.2f}" rgba="0.25 0.29 0.31 1" contype="0" conaffinity="0"/>')
                for columna in range(1, max(2, int(ancho // 2))):
                    wx = x - ancho / 2 + columna * 2.0
                    g.append(f'<geom name="uade_ventana_{piso}_{columna}" type="box" '
                             f'size="0.72 0.025 0.43" pos="{wx:.2f} {y - sy - .07:.2f} '
                             f'{z + .48:.2f}" rgba="0.12 0.24 0.34 1" contype="0" conaffinity="0"/>')
                    g.append(f'<geom name="uade_balcon_{piso}_{columna}" type="box" '
                             f'size="0.78 0.22 0.045" pos="{wx:.2f} {y - sy - .22:.2f} '
                             f'{z - .08:.2f}" rgba="0.37 0.40 0.39 1" contype="0" conaffinity="0"/>')
    # Pérgola peatonal, como la galería de las fotos, sobre un lateral.
    px, py = cx + ancho * .22, cy + alto * .31
    g.append(f'<geom name="uade_pergola_techo" type="box" size="5.0 1.65 0.10" '
             f'pos="{px:.2f} {py:.2f} 3.0" rgba="0.83 0.82 0.74 1" '
             'contype="0" conaffinity="0"/>')
    for i in range(9):
        xx = px - 4.5 + i * 1.125
        g.append(f'<geom name="uade_pergola_viga_{i}" type="box" size="0.055 1.75 0.065" '
                 f'pos="{xx:.2f} {py:.2f} 3.18" rgba="0.84 0.83 0.75 1" '
                 'contype="0" conaffinity="0"/>')
    for i, xx in enumerate((px - 4.5, px + 4.5)):
        for j, yy in enumerate((py - 1.35, py + 1.35)):
            g.append(f'<geom name="uade_pergola_pilar_{i}_{j}" type="cylinder" '
                     f'size="0.10 1.45" pos="{xx:.2f} {yy:.2f} 1.48" '
                     'rgba="0.76 0.75 0.68 1" contype="0" conaffinity="0"/>')
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


def limpiar_escena(temporales: list[str]) -> None:
    for ruta in temporales:
        try:
            os.unlink(ruta)
        except FileNotFoundError:
            pass


