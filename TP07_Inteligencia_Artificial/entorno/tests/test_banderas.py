import os
import tempfile
import unittest
import warnings

from sim import escena_navegacion as esc
from sim.grilla import Mapa


def mapa_con_banderas(textura_a, textura_b):
    mapa = Mapa("x", [[0, 1, 1, 0], [0, 0, 0, 0]], (1, 0), (1, 3), 1.0, "ESTE", 4)
    mapa.escenario = {"objetos": [
        {"tipo": "bandera", "celda": [0, 1], "textura": textura_a},
        {"tipo": "bandera", "celda": [0, 2], "textura": textura_b}]}
    return mapa


class BanderasTests(unittest.TestCase):
    def crear(self, mapa):
        raiz = tempfile.mkdtemp()
        with open(os.path.join(raiz, "go2.xml"), "w", encoding="utf-8") as f:
            f.write('<mujoco><worldbody><body name="base_link"/></worldbody></mujoco>')
        base = os.path.join(raiz, "scene.xml")
        with open(base, "w", encoding="utf-8") as f:
            f.write('<mujoco><include file="go2.xml"/><worldbody></worldbody></mujoco>')
        escena, temporales = esc.crear_escena(base, mapa)
        try:
            with open(escena, encoding="utf-8") as f:
                return f.read()
        finally:
            esc.limpiar_escena(temporales)

    def test_textura_argentina_existe_y_es_14_a_9(self):
        png = os.path.join(esc.DIR_BANDERAS, "bandera_argentina.png")
        self.assertTrue(os.path.isfile(png))
        with open(png, "rb") as f:
            cab = f.read(24)
        ancho = int.from_bytes(cab[16:20], "big")
        alto = int.from_bytes(cab[20:24], "big")
        self.assertEqual(ancho * 9, alto * 14)

    def test_bandera_con_textura_usa_material_sin_colision(self):
        xml = self.crear(mapa_con_banderas("argentina", "argentina"))
        self.assertIn('material="mat_bandera_argentina"', xml)
        self.assertIn('file="', xml)
        tela = [l for l in xml.splitlines() if "_tela" in l][0]
        self.assertIn('contype="0" conaffinity="0"', tela)
        self.assertIn('size="0.63 0.012 0.405"', tela)   # 14:9
        self.assertEqual(xml.count('<texture name="tex_bandera_argentina"'), 1)

    def test_textura_faltante_avisa_y_usa_tela_lisa(self):
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always")
            xml = self.crear(mapa_con_banderas("argentina", "no_existe"))
        self.assertTrue(any("no_existe" in str(a.message) for a in avisos))
        self.assertNotIn("mat_bandera_no_existe", xml)

    def test_rechaza_nombre_de_textura_peligroso(self):
        with self.assertRaises(ValueError):
            self.crear(mapa_con_banderas('x" y="', "argentina"))


if __name__ == "__main__":
    unittest.main()


class IluminacionTests(unittest.TestCase):
    PLANTILLA = ('<headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>'
                 '<light pos="0 0 1.5" dir="0 0 -1" directional="true"/>'
                 '<material reflectance="0.2"/>')

    def test_reemplaza_luz_cenital_por_sol_con_sombras_y_quita_reflejo(self):
        xml = esc._mejorar_iluminacion(self.PLANTILLA)
        self.assertIn('name="sol"', xml)
        self.assertIn('castshadow="true"', xml)
        self.assertNotIn('dir="0 0 -1"', xml)
        self.assertIn('reflectance="0"', xml)

    def test_plantilla_desconocida_no_falla(self):
        self.assertEqual(esc._mejorar_iluminacion("<mujoco/>"), "<mujoco/>")
