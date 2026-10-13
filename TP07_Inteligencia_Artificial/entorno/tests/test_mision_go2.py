import os
import time
import unittest

from sim.camara_go2 import ObjetoPercibido
from sim.grilla import cargar
from sim.mision_go2 import MisionParque
from sim.mundo import Mundo
from sim.safety import perfil


class MisionGo2Tests(unittest.TestCase):
    def setUp(self):
        ruta = os.path.join(os.path.dirname(__file__), "..", "..", "mi_desarrollo",
                            "mapas", "parque_go2_mision.json")
        self.mapa = cargar(ruta)
        self.mundo = Mundo(perfil("tp07"))
        x, y = self.mapa.a_mundo(*self.mapa.inicio)
        self.mundo.situar(x, y, self.mapa.rumbo_inicial())
        self.mision = MisionParque(self.mundo, self.mapa, margen_m=0.38)

    def objeto(self, categoria, distancia, bearing=0.0):
        return ObjetoPercibido(categoria, categoria, 160, 120, bearing,
                               distancia, 25)

    def test_explora_waypoints_sin_usar_celda_de_la_pelota(self):
        pelota = tuple(self.mapa.escenario["pelota_oculta"])
        self.assertNotIn(pelota, self.mision.waypoints)
        self.assertEqual(self.mision.ruta[-1], self.mision.waypoints[0])

    def test_ignora_etiqueta_de_pelota_antes_de_las_zonas_de_busqueda(self):
        estado = self.mision.paso([self.objeto("pelota", 1.3, -0.6)])
        self.assertNotIn(estado, ("DETECTANDO_PELOTA", "ACERCANDOSE"))
        self.assertEqual(self.mision.indice_waypoint, 0)

    def test_persona_cercana_detiene_incluso_con_pelota_visible(self):
        estado = self.mision.paso([
            self.objeto("persona", 0.8), self.objeto("pelota", 0.7)])
        self.assertEqual(estado, "ESPERANDO_PERSONA")
        self.assertFalse(self.mundo.leer()["moviendose"])

    def test_persona_detectada_ocupa_celda_temporal_desde_su_rango(self):
        x, y = self.mapa.a_mundo(*self.mapa.inicio)
        self.mundo.situar(x, y, 0.0)
        pose = self.mundo.leer()
        temporal = self.mision._mapa_temporal(
            [self.objeto("persona", 1.0)], pose)
        self.assertEqual(temporal.celda(11, 5), 1)
        self.assertEqual(self.mapa.celda(11, 5), 0)

    def test_escanea_rotando_y_luego_planifica_el_siguiente_sector(self):
        self.mision.indice_waypoint = 3
        self.mision.estado = "ESCANEANDO"
        self.mision._escaneo_desde = time.monotonic()
        estado = self.mision.paso([])
        self.assertEqual(estado, "ESCANEANDO")
        self.assertEqual(self.mundo.vyaw, 0.45)
        self.mision._escaneo_desde -= 8.0
        estado = self.mision.paso([])
        self.assertEqual(estado, "EXPLORANDO")
        self.assertEqual(self.mision.indice_waypoint, 4)

    def test_backflip_visual_se_animara_sin_orden_fisica(self):
        self.mision.indice_waypoint = self.mapa.escenario[
            "detectar_pelota_desde_waypoint"]
        deteccion = [self.objeto("pelota", 0.7)]
        self.assertEqual(self.mision.paso(deteccion), "DETECTANDO_PELOTA")
        estado = self.mision.paso(deteccion)
        self.assertEqual(estado, "VERIFICANDO_MANIOBRA")
        estado = self.mision.paso(deteccion)
        self.assertEqual(estado, "BACKFLIP_SIMULADO")
        self.assertIn("no envía una orden", self.mision.motivo)
        self.assertFalse(self.mundo.leer()["moviendose"])
        self.assertEqual(self.mundo.leer()["accion"], "backflip_visual")
        self.assertGreater(self.mundo.leer()["gesto_progreso"], 0.0)

    def test_backflip_se_omite_con_objetos_cercanos_sin_marcar_error(self):
        self.mision.estado = "VERIFICANDO_MANIOBRA"
        self.mision._verificar_backflip([self.objeto("arbol", 1.4)])
        self.assertEqual(self.mision.estado, "BACKFLIP_OMITIDO")
        self.assertIn("omitido", self.mision.motivo)
        self.assertFalse(self.mundo.leer()["moviendose"])
        self.assertEqual(self.mision.paso([]), "BACKFLIP_OMITIDO")

    def test_busqueda_tiene_limite_de_tiempo(self):
        mision = MisionParque(self.mundo, self.mapa, margen_m=0.38,
                              timeout_s=0.0)
        self.assertEqual(mision.paso([]), "ERROR")
        self.assertIn("tiempo límite", mision.motivo)

    def test_ruta_temporal_bloqueada_espera_y_termina_con_limite(self):
        self.mision._esperar_ruta(10.0, "persona en el paso")
        self.assertEqual(self.mision.estado, "ESPERANDO_OBSTACULO")
        self.mision._esperar_ruta(20.1, "persona en el paso")
        self.assertEqual(self.mision.estado, "ERROR")
        self.assertIn("10 s", self.mision.motivo)


if __name__ == "__main__":
    unittest.main()
