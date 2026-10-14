"""Escena del patio UADE: persona móvil retirada, personas sentadas, columnas."""
import math
import os
import re
import tempfile
import unittest

from sim.escena_navegacion import crear_escena, limpiar_escena
from sim.grilla import cargar
from sim.planificacion_go2 import astar
from sim.robots import HUELLA

RAIZ = os.path.join(os.path.dirname(__file__), "..", "..")
MAPA = os.path.join(RAIZ, "mi_desarrollo", "mapas", "patio_uade_go2.json")
ESCENA = os.path.join(RAIZ, "entorno", "sim", "unitree_mujoco",
                      "unitree_robots", "go2", "scene_uade.xml")


class PatioUadeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapa = cargar(MAPA)
        cls.escena, cls.temporales = crear_escena(ESCENA, cls.mapa)
        with open(cls.escena, encoding="utf-8") as f:
            cls.xml = f.read()

    @classmethod
    def tearDownClass(cls):
        limpiar_escena(cls.temporales)

    def test_no_hay_persona_movil(self):
        self.assertNotIn("personas", self.mapa.escenario)
        self.assertNotIn("nav_person_", self.xml)
        self.assertNotIn('mocap="true"', self.xml)

    def test_personas_sentadas_estan_sobre_bancos_y_sin_colision(self):
        sentadas = self.mapa.escenario["personas_sentadas"]
        self.assertGreaterEqual(len(sentadas), 3)
        self.assertLess(len(sentadas), len(
            [o for o in self.mapa.escenario["objetos"] if o["tipo"] == "banco"]) + 1)
        bancos = {tuple(o["celda"]) for o in self.mapa.escenario["objetos"]
                  if o["tipo"] == "banco"}
        for p in sentadas:
            self.assertIn(tuple(p["celda"]), bancos)
        geoms = re.findall(r'<geom name="deco_persona_[^>]*>', self.xml)
        self.assertGreater(len(geoms), 10 * len(sentadas))
        self.assertTrue(all('contype="0" conaffinity="0"' in g for g in geoms))

    def test_persona_en_celda_sin_banco_es_error(self):
        from sim.escena_navegacion import _personas_sentadas_mjcf
        esc = {"objetos": [], "personas_sentadas": [{"celda": [1, 1]}]}
        with self.assertRaises(ValueError):
            _personas_sentadas_mjcf(self.mapa, esc)

    def test_proporciones_de_persona_sentada(self):
        # Cabeza entre 1,1 y 1,5 m, asiento a 0,48 m: adulto sentado real.
        z = [float(m.group(1)) for m in re.finditer(
            r'name="deco_persona_0_cabeza" type="sphere" size="[\d.]+" '
            r'pos="[-\d.]+ [-\d.]+ ([\d.]+)"', self.xml)]
        self.assertEqual(len(z), 1)
        self.assertTrue(1.1 < z[0] < 1.5)

    def test_columnas_de_la_pergola_son_obstaculos_solidos_en_la_grilla(self):
        columnas = [o["celda"] for o in self.mapa.escenario["objetos"]
                    if o["tipo"] == "columna"]
        self.assertEqual(len(columnas), 4)
        for f, c in columnas:
            self.assertEqual(self.mapa.celda(f, c), 1)
        self.assertEqual(len(re.findall(
            r'name="nav_obstacle_columna_\d+_\d+_fuste"[^>]*contype="1"', self.xml)), 4)
        self.assertNotIn("uade_pergola_pilar", self.xml)

    def test_modelo_compila_con_mujoco(self):
        import mujoco
        # crear_escena genera archivos temporales; se vuelve a crear para compilar.
        ruta, temporales = crear_escena(ESCENA, self.mapa)
        try:
            modelo = mujoco.MjModel.from_xml_path(ruta)
        finally:
            limpiar_escena(temporales)
        self.assertGreater(modelo.ngeom, 300)

    def test_ruta_de_busqueda_completa_existe_y_evita_las_columnas(self):
        largo, ancho = HUELLA["go2"]
        margen = math.hypot(largo / 2, ancho / 2) + 0.05
        pts = [tuple(self.mapa.inicio)] + [tuple(p) for p in
                                          self.mapa.escenario["busqueda"]]
        columnas = {tuple(o["celda"]) for o in self.mapa.escenario["objetos"]
                    if o["tipo"] == "columna"}
        for a, b in zip(pts, pts[1:]):
            ruta = astar(self.mapa, a, b, margen)
            self.assertFalse(columnas & set(ruta))

    def test_piernas_de_personas_no_invaden_el_margen_de_la_ruta(self):
        largo, ancho = HUELLA["go2"]
        margen = math.hypot(largo / 2, ancho / 2) + 0.05
        pts = [tuple(self.mapa.inicio)] + [tuple(p) for p in
                                          self.mapa.escenario["busqueda"]]
        rutas = [c for a, b in zip(pts, pts[1:])
                 for c in astar(self.mapa, a, b, margen)]
        extremos = []        # pies/rodillas: lo más adelantado de cada persona
        for m in re.finditer(r'name="deco_persona_\d+_pie_(?:izq|der)"[^>]*'
                             r'pos="([-\d.]+) ([-\d.]+) ', self.xml):
            extremos.append((float(m.group(1)), float(m.group(2))))
        self.assertTrue(extremos)
        for celda in rutas:
            cx, cy = self.mapa.a_mundo(*celda)
            for ex, ey in extremos:
                # El robot (medio largo 0,31 m) no toca un pie de la persona.
                self.assertGreater(math.hypot(cx - ex, cy - ey), largo / 2 + 0.1)


class SuavizadoDeRutaTests(unittest.TestCase):
    """Los atajos acortan la ruta sin acercarse a los obstáculos más que A*."""

    @classmethod
    def setUpClass(cls):
        cls.mapa = cargar(MAPA)
        largo, ancho = HUELLA["go2"]
        cls.margen = math.hypot(largo / 2, ancho / 2) + 0.05

    @staticmethod
    def largo(v):
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(v, v[1:]))

    def test_acorta_y_reduce_giros_sin_cambiar_extremos(self):
        from sim.planificacion_go2 import suavizar_ruta
        ruta = astar(self.mapa, (21, 2), (4, 26), self.margen)
        v = suavizar_ruta(self.mapa, ruta, self.margen)
        self.assertEqual((v[0], v[-1]), (ruta[0], ruta[-1]))
        self.assertLess(self.largo(v), 0.9 * self.largo(ruta))
        self.assertLess(len(v), len(ruta) // 3)

    def test_todo_el_trayecto_conserva_el_despeje_pedido(self):
        from sim.planificacion_go2 import _despeje_m, suavizar_ruta
        margen = self.margen + 0.10
        for destino in ((4, 26), (3, 3), (20, 3)):
            ruta = astar(self.mapa, (21, 15), destino, self.margen)
            v = suavizar_ruta(self.mapa, ruta, margen)
            for a, b in zip(v, v[1:]):
                n = max(1, int(self.largo([a, b]) / 0.05))
                for i in range(n + 1):
                    f = a[0] + (b[0] - a[0]) * i / n
                    c = a[1] + (b[1] - a[1]) * i / n
                    self.assertGreaterEqual(_despeje_m(self.mapa, f, c), margen - 1e-6)

    def test_no_hay_atajo_a_traves_de_un_obstaculo(self):
        from sim.planificacion_go2 import _segmento_libre
        banco = next(o["celda"] for o in self.mapa.escenario["objetos"]
                     if o["tipo"] == "banco")
        izq = (banco[0], banco[1] - 3)
        der = (banco[0], banco[1] + 3)
        self.assertFalse(_segmento_libre(self.mapa, izq, der, self.margen))

    def test_ruta_densa_pasa_por_los_vertices(self):
        from sim.planificacion_go2 import ruta_densa
        celdas = ruta_densa([(0, 0), (0, 4), (3, 4)])
        self.assertEqual((celdas[0], celdas[-1]), ((0, 0), (3, 4)))
        self.assertIn((0, 4), celdas)


if __name__ == "__main__":
    unittest.main()
