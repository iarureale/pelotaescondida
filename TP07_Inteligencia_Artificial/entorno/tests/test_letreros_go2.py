"""Letreros de fachada: FACO, FACE, UBS, FASA, FAIN sobre el edificio norte."""
import os
import re
import unittest

from sim import letreros_go2 as L
from sim.escena_navegacion import crear_escena, limpiar_escena
from sim.grilla import cargar

RAIZ = os.path.join(os.path.dirname(__file__), "..", "..")
MAPA = os.path.join(RAIZ, "mi_desarrollo", "mapas", "patio_uade_go2.json")
ESCENA = os.path.join(RAIZ, "entorno", "sim", "unitree_mujoco", "unitree_robots", "go2", "scene_uade.xml")


def pos(geom):
    m = re.search(r'fromto="([-\d. ]+)"', geom)
    v = [float(x) for x in m.group(1).split()]
    return v[:3], v[3:]


class LetrerosTests(unittest.TestCase):
    def test_texto_se_lee_de_izquierda_a_derecha_desde_la_plaza(self):
        g = [x for x in L.geoms_letrero("t", "FACO", "norte", 3.0, 4, cara=1.1, altura_m=0.8)
             if "fondo" not in x]
        xs_por_letra = []
        # Las letras salen en orden; el x mínimo de cada una crece.
        cursor = 3.0
        for c in "FACO":
            ancho = L.FUENTE[c][0] * 0.8
            xs_por_letra.append((cursor, cursor + ancho))
            cursor += ancho + L.SEPARACION_M * 0.8
        todos = [p for geom in g for p in pos(geom)]
        self.assertAlmostEqual(min(p[0] for p in todos), 3.0, places=2)
        self.assertAlmostEqual(max(p[0] for p in todos), xs_por_letra[-1][1], places=2)

    def test_las_letras_van_delante_de_la_pared_y_dentro_de_su_banda(self):
        cara = 1.1
        g = L.geoms_letrero("t", "UBS", "norte", 3.4, 2, cara=cara, altura_m=0.8)
        z_centro = L.CENTROS_PISO_M[2] + L.DESPLAZAMIENTO_VENTANA_M
        for geom in g:
            if "fondo" in geom:
                continue
            for x, y, z in pos(geom):
                self.assertLess(y, cara)                            # hacia la plaza (-y)
                self.assertGreater(y, cara - 0.3)                   # pegadas a la pared
                self.assertTrue(z_centro - 0.45 < z < z_centro + 0.45)

    def test_sin_colision_y_nombres_unicos(self):
        g = L.geoms_letreros([{"texto": "FACO", "x_m": 3, "piso": 4},
                              {"texto": "FACE", "x_m": 6, "piso": 3}], 32, 24, 15.5, -11.5)
        nombres = re.findall(r'name="([^"]+)"', "".join(g))
        self.assertEqual(len(nombres), len(set(nombres)))
        self.assertTrue(all('contype="0" conaffinity="0"' in x for x in g))

    def test_letra_desconocida_o_piso_invalido_dan_error_claro(self):
        with self.assertRaisesRegex(ValueError, "Ñ"):
            L.geoms_letrero("t", "ÑU", "norte", 0, 1, 1.0)
        with self.assertRaisesRegex(ValueError, "piso"):
            L.geoms_letrero("t", "UBS", "norte", 0, 9, 1.0)
        with self.assertRaisesRegex(ValueError, "lado"):
            L.geoms_letrero("t", "UBS", "sur", 0, 1, 1.0)
        with self.assertRaisesRegex(ValueError, "x_m"):
            L.geoms_letreros([{"texto": "UBS"}], 32, 24, 15.5, -11.5)

    def test_laterales_se_leen_hacia_el_patio(self):
        este = L.geoms_letrero("e", "FA", "este", 10.0, 1, cara=33.0, altura_m=0.8)
        oeste = L.geoms_letrero("o", "FA", "oeste", 10.0, 1, cara=-2.0, altura_m=0.8)
        ys_e = [p[1] for g in este if "fondo" not in g for p in pos(g)]
        ys_o = [p[1] for g in oeste if "fondo" not in g for p in pos(g)]
        # Visto desde la plaza: en la este (se mira hacia +x) la derecha es -y;
        # en la oeste (se mira hacia -x) la derecha es +y.
        self.assertLess(min(ys_e), 10.0)
        self.assertAlmostEqual(max(ys_e), 10.0, places=2)
        self.assertGreater(max(ys_o), 10.0)
        self.assertAlmostEqual(min(ys_o), 10.0, places=2)
        self.assertTrue(all(p[0] < 33.0 for g in este if "fondo" not in g for p in pos(g)))
        self.assertTrue(all(p[0] > -2.0 for g in oeste if "fondo" not in g for p in pos(g)))

    def test_el_patio_uade_tiene_los_cinco_letreros_de_la_foto(self):
        mapa = cargar(MAPA)
        textos = [d["texto"] for d in mapa.escenario["letreros"]]
        self.assertEqual(sorted(textos), sorted(["FACO", "FACE", "UBS", "FASA", "FAIN"]))
        ruta, temporales = crear_escena(ESCENA, mapa)
        try:
            with open(ruta, encoding="utf-8") as f:
                xml = f.read()
        finally:
            limpiar_escena(temporales)
        for i in range(5):
            self.assertIn(f'name="uade_letrero_{i}_fondo"', xml)
        # Disposición escalonada de la foto: FACO arriba, FAIN debajo.
        por_texto = {d["texto"]: d for d in mapa.escenario["letreros"]}
        self.assertGreater(por_texto["FACO"]["piso"], por_texto["FACE"]["piso"])
        self.assertGreater(por_texto["FACE"]["piso"], por_texto["FASA"]["piso"])
        self.assertGreater(por_texto["FASA"]["piso"], por_texto["FAIN"]["piso"])
        self.assertEqual(por_texto["UBS"]["piso"], por_texto["FASA"]["piso"])
        # UBS queda a la izquierda de FASA (visto desde la plaza: mayor y).
        self.assertGreater(por_texto["UBS"]["x_m"], por_texto["FASA"]["x_m"])

    def test_los_letreros_estan_en_la_pared_derecha_dentro_de_la_fachada(self):
        mapa = cargar(MAPA)
        # Pared derecha = fachada este. Se extiende de y = cy-alto/2-2,5 a cy+alto/2+2,5.
        y_min, y_max = -11.5 - 12 - 2.5, -11.5 + 12 + 2.5
        for d in mapa.escenario["letreros"]:
            self.assertEqual(d["lado"], "este")
            ancho = L.ancho_texto_m(d["texto"], d.get("altura_m", 0.8))
            # En la fachada este el texto avanza hacia -y desde x_m.
            self.assertLess(d["x_m"] + 0.35, y_max, d["texto"])
            self.assertGreater(d["x_m"] - ancho - 0.35, y_min, d["texto"])

    def test_los_letreros_no_chocan_entre_si(self):
        mapa = cargar(MAPA)
        por_piso = {}
        for d in mapa.escenario["letreros"]:
            fin = d["x_m"]                                   # borde izquierdo (+y)
            ini = fin - L.ancho_texto_m(d["texto"], d.get("altura_m", 0.8))
            por_piso.setdefault(d["piso"], []).append((ini, fin, d["texto"]))
        for tramos in por_piso.values():
            tramos.sort()
            for (_, f1, t1), (i2, _, t2) in zip(tramos, tramos[1:]):
                self.assertLess(f1 + 0.7, i2, f"{t1} y {t2} se pisan")


if __name__ == "__main__":
    unittest.main()
