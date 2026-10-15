"""Letreros en las fachadas del patio (p. ej. FACO, FACE, UBS, FASA, FAIN).

MuJoCo no dibuja texto 3D dentro de la escena (las etiquetas del visor son
planas y no escalan con la distancia), y una textura con texto exigiría
generarla con una librería de imágenes que el proyecto no tiene. Por eso las
letras se arman con trazos: cada letra es un conjunto de polilíneas en un
rectángulo unitario y cada tramo se convierte en una cápsula decorativa (sin
colisión). Se parece a las letras corpóreas de la fachada real y no agrega
dependencias.

Los letreros salen del JSON del mapa (``escenario["letreros"]``), por ejemplo::

    {"lado": "norte", "texto": "FACO", "x_m": 3.5, "piso": 4, "altura_m": 0.8}

* ``lado``: fachada ("norte", "este" u "oeste"); el texto se lee de izquierda a
  derecha para quien mira la fachada desde el patio.
* ``x_m``: coordenada MUNDIAL donde EMPIEZA el texto (su borde izquierdo visto
  desde la plaza) a lo largo de la fachada: x en la norte (el texto avanza hacia
  +x); y en la este (avanza hacia -y) y en la oeste (avanza hacia +y).
* ``piso``: banda de ventanas (0 = la más baja).
* ``altura_m``: alto de las letras (opcional, 0,8 m por defecto).
"""
from __future__ import annotations

import math

# Alto de cada banda de ventanas y altura de su centro (coinciden con la
# fachada de escena_navegacion._arquitectura_patio).
CENTROS_PISO_M = (1.35, 2.75, 4.15, 5.55, 6.95)
DESPLAZAMIENTO_VENTANA_M = 0.48     # el centro de la ventana está 0,48 m sobre la faja
SEPARACION_M = 0.14                 # entre letras, como fracción del alto (ver abajo)
SALIENTE_M = 0.13                   # cuánto sobresalen las letras de la pared
COLOR = "0.92 0.94 0.96 1"
COLOR_FONDO = "0.09 0.16 0.23 1"     # vidrio oscuro detrás de las letras

# Cada letra: (ancho relativo al alto, [polilíneas en un cuadro 0..1 x 0..1]).
_ARCO_C = [(1, .85), (.8, 1), (.25, 1), (0, .8), (0, .2), (.25, 0), (.8, 0), (1, .15)]
FUENTE: dict[str, tuple[float, list[list[tuple[float, float]]]]] = {
    "A": (.72, [[(0, 0), (.5, 1), (1, 0)], [(.2, .38), (.8, .38)]]),
    "B": (.62, [[(0, 0), (0, 1), (.7, 1), (.95, .86), (.95, .66), (.7, .53), (0, .53)],
                [(.7, .53), (1, .38), (1, .16), (.76, 0), (0, 0)]]),
    "C": (.64, [_ARCO_C]),
    "D": (.64, [[(0, 0), (0, 1), (.55, 1), (.95, .78), (.95, .22), (.55, 0), (0, 0)]]),
    "E": (.56, [[(1, 1), (0, 1), (0, 0), (1, 0)], [(0, .5), (.8, .5)]]),
    "F": (.54, [[(0, 0), (0, 1), (1, 1)], [(0, .55), (.78, .55)]]),
    "I": (.12, [[(.5, 0), (.5, 1)]]),
    "L": (.52, [[(0, 1), (0, 0), (1, 0)]]),
    "M": (.8, [[(0, 0), (0, 1), (.5, .35), (1, 1), (1, 0)]]),
    "N": (.64, [[(0, 0), (0, 1), (1, 0), (1, 1)]]),
    "O": (.68, [[(.25, 1), (.75, 1), (1, .8), (1, .2), (.75, 0), (.25, 0),
                 (0, .2), (0, .8), (.25, 1)]]),
    "P": (.58, [[(0, 0), (0, 1), (.75, 1), (1, .85), (1, .65), (.75, .5), (0, .5)]]),
    "R": (.62, [[(0, 0), (0, 1), (.75, 1), (1, .85), (1, .65), (.75, .5), (0, .5)],
                [(.55, .5), (1, 0)]]),
    "S": (.62, [[(1, .85), (.8, 1), (.2, 1), (0, .82), (0, .66), (.2, .53), (.8, .47),
                 (1, .34), (1, .18), (.8, 0), (.2, 0), (0, .15)]]),
    "T": (.6, [[(0, 1), (1, 1)], [(.5, 1), (.5, 0)]]),
    "U": (.64, [[(0, 1), (0, .2), (.25, 0), (.75, 0), (1, .2), (1, 1)]]),
    "Y": (.64, [[(0, 1), (.5, .5), (1, 1)], [(.5, .5), (.5, 0)]]),
    " ": (.4, []),
}


def ancho_texto_m(texto: str, altura_m: float) -> float:
    """Ancho total del texto (m), con la separación entre letras."""
    for c in texto.upper():
        if c not in FUENTE:
            raise ValueError(f"Letreros: la letra '{c}' no está en la fuente "
                             f"({''.join(sorted(FUENTE))}).")
    anchos = [FUENTE[c][0] * altura_m for c in texto.upper()]
    return sum(anchos) + SEPARACION_M * altura_m * max(0, len(texto) - 1)


def geoms_letrero(nombre: str, texto: str, lado: str, x_m: float, piso: int,
                  cara: float, altura_m: float = 0.8) -> list[str]:
    """Geometrías MJCF (cápsulas sin colisión) de un texto sobre una fachada.

    ``cara`` es la coordenada de la pared hacia el patio (y en la norte, x en
    la este/oeste). Devuelve la lista de ``<geom .../>``.
    """
    if lado not in ("norte", "este", "oeste"):
        raise ValueError(f"Letreros: lado desconocido '{lado}'.")
    if not 0 <= piso < len(CENTROS_PISO_M):
        raise ValueError(f"Letreros: el piso {piso} no existe (0..{len(CENTROS_PISO_M) - 1}).")
    ancho_texto_m(texto, altura_m)                       # valida las letras
    z_base = CENTROS_PISO_M[piso] + DESPLAZAMIENTO_VENTANA_M - altura_m / 2
    radio = max(0.03, 0.08 * altura_m)
    geoms = []
    # Franja de vidrio oscuro detrás del texto (como en la fachada real): sin
    # ella las letras claras se pierden sobre los pilares de la pared.
    largo = ancho_texto_m(texto, altura_m)
    semi_largo, semi_alto = largo / 2 + 0.35, altura_m / 2 + 0.14
    z_fondo = z_base + altura_m / 2
    if lado == "norte":
        geoms.append(f'<geom name="{nombre}_fondo" type="box" size="{semi_largo:.3f} 0.015 {semi_alto:.3f}" '
                     f'pos="{x_m + largo / 2:.3f} {cara - 0.07:.3f} {z_fondo:.3f}" '
                     f'rgba="{COLOR_FONDO}" contype="0" conaffinity="0"/>')
    else:
        sentido = -1.0 if lado == "este" else 1.0
        cx_fondo = cara - 0.07 if lado == "este" else cara + 0.07
        geoms.append(f'<geom name="{nombre}_fondo" type="box" size="0.015 {semi_largo:.3f} {semi_alto:.3f}" '
                     f'pos="{cx_fondo:.3f} {x_m + sentido * largo / 2:.3f} {z_fondo:.3f}" '
                     f'rgba="{COLOR_FONDO}" contype="0" conaffinity="0"/>')
    cursor = 0.0                                          # a lo largo del texto
    n = 0
    for indice, caracter in enumerate(texto.upper()):
        ancho_rel, trazos = FUENTE[caracter]
        ancho = ancho_rel * altura_m
        for trazo in trazos:
            for (ua, va), (ub, vb) in zip(trazo, trazo[1:]):
                p = []
                for u, v in ((ua, va), (ub, vb)):
                    a = cursor + u * ancho                # posición a lo largo
                    z = z_base + v * altura_m
                    if lado == "norte":                   # se lee hacia +x, pared en +y
                        p.append((x_m + a, cara - SALIENTE_M, z))
                    elif lado == "este":                  # se mira hacia +x: derecha = -y
                        p.append((cara - SALIENTE_M, x_m - a, z))
                    else:                                 # oeste: se mira hacia -x: derecha = +y
                        p.append((cara + SALIENTE_M, x_m + a, z))
                geoms.append(
                    f'<geom name="{nombre}_{n}" type="capsule" '
                    f'fromto="{p[0][0]:.3f} {p[0][1]:.3f} {p[0][2]:.3f} '
                    f'{p[1][0]:.3f} {p[1][1]:.3f} {p[1][2]:.3f}" size="{radio:.3f}" '
                    f'rgba="{COLOR}" contype="0" conaffinity="0"/>')
                n += 1
        cursor += ancho + (SEPARACION_M * altura_m if indice < len(texto) - 1 else 0.0)
    return geoms


def geoms_letreros(letreros: list[dict], ancho: float, alto: float,
                   cx: float, cy: float) -> list[str]:
    """Todos los letreros del mapa. Misma geometría de edificios que el patio."""
    caras = {   # coordenada de la pared hacia la explanada (ver _arquitectura_patio)
        "norte": cy + alto / 2 + 1.2 - 0.55 - 0.04,
        "este": cx + ancho / 2 + 1.2 - 0.55 - 0.04,
        "oeste": cx - ancho / 2 - 1.2 + 0.55 + 0.04,
    }
    geoms = []
    for i, d in enumerate(letreros):
        try:
            geoms.extend(geoms_letrero(
                f"uade_letrero_{i}", str(d["texto"]), d.get("lado", "norte"),
                float(d["x_m"]), int(d.get("piso", 3)), caras[d.get("lado", "norte")],
                float(d.get("altura_m", 0.8))))
        except KeyError as exc:
            raise ValueError(f"letreros[{i}]: falta el campo {exc}.") from exc
    return geoms
