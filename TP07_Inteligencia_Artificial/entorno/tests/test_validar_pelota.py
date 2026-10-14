"""Validación cruzada segmentación-RGB de la pelota (falsos positivos de ID-color)."""
import math
import unittest

import numpy as np

from sim.camara_go2 import ObjetoPercibido, validar_pelota_con_rgb

FOCAL = 480 / (2 * math.tan(math.radians(30)))


def pelota(distancia, x=300, y=240, radio_px=None, pixeles=None):
    radio_px = radio_px or max(1.0, 0.10 * FOCAL / distancia)
    r = int(round(radio_px))
    return ObjetoPercibido("nav_target_ball", "pelota", x, y, 0.0, distancia,
                           pixeles or int(math.pi * radio_px ** 2),
                           x - r, x + r, y - r, y + r)


def imagen_con_disco(x, y, radio, color=(220, 30, 30)):
    img = np.full((480, 640, 3), (120, 160, 90), dtype=np.uint8)
    yy, xx = np.ogrid[:480, :640]
    img[(xx - x) ** 2 + (yy - y) ** 2 <= radio ** 2] = color
    return img


class ValidarPelotaTests(unittest.TestCase):
    def test_pelota_real_roja_y_coherente_se_conserva(self):
        det = pelota(4.0)
        radio = 0.10 * FOCAL / 4.0
        validas, desc = validar_pelota_con_rgb([det], imagen_con_disco(300, 240, radio))
        self.assertEqual((len(validas), desc), (1, 0))

    def test_deteccion_sin_rojo_en_la_imagen_se_descarta(self):
        det = pelota(0.4, radio_px=12)             # "pelota" pegada a la cámara
        gris = np.full((480, 640, 3), 90, dtype=np.uint8)
        validas, desc = validar_pelota_con_rgb([det], gris)
        self.assertEqual((validas, desc), ([], 1))

    def test_escala_incoherente_con_la_profundidad_se_descarta(self):
        # Hay rojo, pero un objeto de 0,2 m a 0,4 m ocuparía ~100 px, no 4.
        det = pelota(0.4, radio_px=2)
        validas, desc = validar_pelota_con_rgb([det], imagen_con_disco(300, 240, 2))
        self.assertEqual(desc, 1)

    def test_otras_categorias_no_se_tocan(self):
        arbol = ObjetoPercibido("nav_obstacle_arbol_1_1_tronco", "arbol", 1, 1,
                                0.0, 2.0, 30, 0, 5, 0, 5)
        validas, desc = validar_pelota_con_rgb(
            [arbol], np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual((validas, desc), ([arbol], 0))


class VistaPreviaTests(unittest.TestCase):
    """La vista previa se achica; la captura y la detección no cambian."""

    def png_publicado(self, factor):
        import base64
        import struct
        from sim.camara_go2 import VistaCamaraTk
        vista = VistaCamaraTk(factor=factor)
        vista._activa = True            # sin abrir la ventana de Tk
        det = pelota(4.0)
        vista.publicar(np.zeros((480, 640, 3), dtype=np.uint8), [det], estado="X")
        datos, texto = vista._queue.get_nowait()
        png = base64.b64decode(datos)
        return struct.unpack(">II", png[16:24]), texto

    def test_por_defecto_la_vista_es_la_mitad(self):
        (ancho, alto), texto = self.png_publicado(None)
        self.assertEqual((ancho, alto), (320, 240))
        self.assertIn("pelota", texto)

    def test_factor_1_conserva_la_resolucion_completa(self):
        (ancho, alto), _ = self.png_publicado(1)
        self.assertEqual((ancho, alto), (640, 480))

    def test_la_barra_muestra_solo_las_detecciones_mas_cercanas(self):
        from sim.camara_go2 import VistaCamaraTk
        vista = VistaCamaraTk()
        vista._activa = True
        dets = [pelota(d) for d in (9, 8, 7, 6, 5, 4, 3)]
        vista.publicar(np.zeros((480, 640, 3), dtype=np.uint8), dets, estado="")
        _, texto = vista._queue.get_nowait()
        self.assertEqual(texto.count("pelota"), VistaCamaraTk.MAX_DETALLES)
        self.assertIn("3.0 m", texto)


if __name__ == "__main__":
    unittest.main()
