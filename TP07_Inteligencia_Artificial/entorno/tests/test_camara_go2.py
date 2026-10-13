import unittest
from types import SimpleNamespace
import base64
import struct
import numpy as np

from sim.camara_go2 import (ObjetoPercibido, SeguidorPelota,
                            VistaCamaraTk, detectar_objetos_segmentados,
                            detectar_pelota)
from sim.safety import perfil
from sim.mundo import Mundo


class CamaraGo2Tests(unittest.TestCase):
    def test_segmentacion_y_depth_detectan_persona_solo_en_pixeles_visibles(self):
        nombres = {0: "nav_person_0_torso", 1: "nav_target_ball",
                   2: "floor"}
        mujoco = SimpleNamespace(
            mjtObj=SimpleNamespace(mjOBJ_GEOM=5),
            mj_id2name=lambda model, tipo, indice: nombres.get(indice))
        model = SimpleNamespace(ngeom=3)
        segmentacion = np.zeros((40, 60, 2), dtype=np.int32)
        segmentacion[:, :, 0] = -1
        segmentacion[10:25, 10:25, 0] = 0
        segmentacion[10:25, 10:25, 1] = 5
        segmentacion[15:20, 40:45, 0] = 1
        segmentacion[15:20, 40:45, 1] = 5
        profundidad = np.full((40, 60), 3.0, dtype=np.float32)
        detecciones = detectar_objetos_segmentados(
            segmentacion, profundidad, model, mujoco, pixeles_minimos=5)
        self.assertEqual({d.categoria for d in detecciones}, {"persona", "pelota"})
        persona = next(d for d in detecciones if d.categoria == "persona")
        self.assertAlmostEqual(persona.distancia_m, 3.0)
        self.assertLess(persona.bearing_rad, 0)

    def test_preview_publica_png_y_anota_cajas(self):
        vista = VistaCamaraTk()
        vista._activa = True
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        objeto = ObjetoPercibido("nav_target_ball", "pelota", 10, 10,
                                 0.0, 1.0, 25, 4, 15, 5, 15)
        vista.publicar(frame, [objeto], estado="MISION: NAVEGANDO")
        contenido, texto = vista._queue.get_nowait()
        png = base64.b64decode(contenido)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(struct.unpack(">II", png[16:24]), (30, 20))
        self.assertIn("MISION", texto)

    def test_detecta_marcador_rojo_y_estima_error_angular(self):
        imagen = np.zeros((120, 160, 3), dtype=np.uint8)
        imagen[45:65, 100:120] = (240, 20, 15)
        deteccion = detectar_pelota(imagen)
        self.assertIsNotNone(deteccion)
        self.assertGreater(deteccion.bearing_rad, 0)
        self.assertGreater(deteccion.distancia_m, 0)

    def test_seguidor_se_detiene_si_pierde_el_objetivo(self):
        mundo = Mundo(perfil("tp07"))
        seguidor = SeguidorPelota(mundo)
        self.assertEqual(seguidor.paso(None), "objetivo_perdido")
        self.assertFalse(mundo.leer()["moviendose"])

    def test_seguidor_orienta_hacia_la_pelota(self):
        imagen = np.zeros((120, 160, 3), dtype=np.uint8)
        imagen[45:65, 100:120] = (240, 20, 15)
        mundo = Mundo(perfil("tp07"))
        seguidor = SeguidorPelota(mundo)
        seguidor.paso(detectar_pelota(imagen))
        self.assertLess(mundo.vyaw, 0)


if __name__ == "__main__":
    unittest.main()
