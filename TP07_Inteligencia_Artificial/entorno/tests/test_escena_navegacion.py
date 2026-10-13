import os
import tempfile
import unittest

from sim.grilla import Mapa, cargar
from sim.escena_navegacion import crear_escena, limpiar_escena


class EscenaNavegacionTests(unittest.TestCase):
    def test_crea_geometrias_con_colision_y_cam_go2(self):
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "go2.xml"), "w", encoding="utf-8") as f:
                f.write('<mujoco><worldbody><body name="base_link"/></worldbody></mujoco>')
            base = os.path.join(root, "scene.xml")
            with open(base, "w", encoding="utf-8") as f:
                f.write('<mujoco><include file="go2.xml"/><worldbody></worldbody></mujoco>')
            mapa = Mapa("x", [[0, 1], [0, 0]], (0, 0), (1, 1), 0.5, "ESTE", 4)
            escena, temporales = crear_escena(base, mapa, incluir_bola=True)
            try:
                with open(escena, encoding="utf-8") as f:
                    xml = f.read()
                with open(temporales[0], encoding="utf-8") as f:
                    robot = f.read()
                self.assertIn('name="nav_obstacle_0_1"', xml)
                self.assertIn('contype="1" conaffinity="1"', xml)
                self.assertIn('name="nav_target_ball"', xml)
                self.assertIn('name="go2_front_camera"', robot)
                self.assertIn('quat="0.5 0.5 -0.5 -0.5"', robot)
                self.assertNotIn('file="go2.xml"', xml)
            finally:
                limpiar_escena(temporales)
            self.assertFalse(any(os.path.exists(p) for p in temporales))

    def test_parque_tiene_mobiliario_colisionable_persona_y_pelota_oculta(self):
        mapa = cargar(os.path.join(
            os.path.dirname(__file__), "..", "..", "mi_desarrollo", "mapas",
            "parque_go2_mision.json"))
        self.assertEqual(mapa.escenario["pelota_oculta"], [3, 11])
        self.assertNotEqual(tuple(mapa.escenario["pelota_oculta"]), mapa.destino)
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "go2.xml"), "w", encoding="utf-8") as f:
                f.write('<mujoco><worldbody><body name="base_link"/></worldbody></mujoco>')
            base = os.path.join(root, "scene.xml")
            with open(base, "w", encoding="utf-8") as f:
                f.write('<mujoco><include file="go2.xml"/><worldbody></worldbody></mujoco>')
            escena, temporales = crear_escena(base, mapa)
            try:
                with open(escena, encoding="utf-8") as f:
                    xml = f.read()
                self.assertIn('name="nav_obstacle_arbol_3_7_tronco"', xml)
                self.assertIn('name="nav_obstacle_banco_6_7_asiento"', xml)
                self.assertIn('mocap="true"', xml)
                self.assertIn('name="nav_target_ball"', xml)
                self.assertIn('park_grass_west', xml)
                self.assertIn('park_path', xml)
            finally:
                limpiar_escena(temporales)

    def test_mapa_ampliado_mide_12_por_14_y_tiene_ruta_de_busqueda(self):
        mapa = cargar(os.path.join(
            os.path.dirname(__file__), "..", "..", "mi_desarrollo", "mapas",
            "parque_go2_ampliado.json"))
        self.assertEqual((mapa.filas, mapa.columnas), (24, 28))
        self.assertEqual(mapa.tamano_celda, 0.5)
        self.assertEqual(len(mapa.escenario["objetos"]), 62)
        self.assertEqual(mapa.escenario["pelota_oculta"], [3, 23])
        self.assertEqual(len(mapa.escenario["busqueda"]), 5)


if __name__ == "__main__":
    unittest.main()

