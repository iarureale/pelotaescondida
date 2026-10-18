"""Planificador de ángulo libre (Theta*): ruta corta, segura y siempre que exista."""
import math
import os
import random
import time
import unittest
from unittest import mock

from sim import planificacion_go2 as P
from sim.grilla import Mapa, cargar
from sim.mundo import Mundo
from sim.planificacion_go2 import SinRuta, astar, planificar, recortar_final, suavizar_ruta
from sim.safety import perfil

MAPA = os.path.join(os.path.dirname(__file__), "..", "..", "mi_desarrollo", "mapas", "patio_uade_go2.json")
MARGEN, HOLGURA = 0.39, 0.10


def largo(mapa, v):
    return P.largo_ruta_m(mapa, v)


def viejo(mapa, a, b):
    """El pipeline anterior: A* de 4 vecinos + atajos."""
    return suavizar_ruta(mapa, astar(mapa, a, b, MARGEN), MARGEN + HOLGURA)


class PlanificadorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapa = cargar(MAPA)
        P.campo_de(cls.mapa)
        cls.libres = [(f, c) for f in range(cls.mapa.filas) for c in range(cls.mapa.columnas)
                      if cls.mapa.es_transitable(f, c)]

    def test_si_se_ven_en_zona_ancha_la_ruta_es_la_recta(self):
        campo = P.campo_de(self.mapa)
        pares = [(a, b) for a in self.libres[::17] for b in self.libres[::23]
                 if a != b and campo.en(*a) > 1.0 and campo.en(*b) > 1.0
                 and campo.evaluar(a, b, MARGEN, MARGEN + HOLGURA)[1] <= 1.01 * math.hypot(
                     a[0] - b[0], a[1] - b[1]) * self.mapa.tamano_celda]
        self.assertGreater(len(pares), 5)
        for a, b in pares[:10]:
            v = planificar(self.mapa, a, b, MARGEN, HOLGURA)
            self.assertEqual(len(v), 2, f"{a}->{b}")
            self.assertAlmostEqual(largo(self.mapa, v), math.hypot(a[0] - b[0], a[1] - b[1]), places=3)

    def test_nunca_es_mas_larga_que_el_pipeline_anterior(self):
        bola = tuple(self.mapa.escenario["pelota_oculta"])
        casos = [((21, 2), bola), ((21, 15), bola), ((12, 8), bola), ((3, 3), (20, 3)),
                 ((21, 29), (3, 29)), ((20, 3), (4, 26)), ((8, 28), (18, 5))]
        total_n = total_v = 0.0
        for a, b in casos:
            n = largo(self.mapa, planificar(self.mapa, a, b, MARGEN, HOLGURA))
            v = largo(self.mapa, viejo(self.mapa, a, b))
            self.assertLessEqual(n, v * 1.03 + 1e-6, f"{a}->{b}: {n:.2f} vs {v:.2f}")
            total_n, total_v = total_n + n, total_v + v
        self.assertLess(total_n, 0.98 * total_v)           # en conjunto, claramente mejor

    def test_conserva_el_despeje_en_todo_el_trayecto(self):
        bola = tuple(self.mapa.escenario["pelota_oculta"])
        for a, b in (((21, 2), bola), ((3, 3), (20, 3)), ((20, 3), (4, 26))):
            v = planificar(self.mapa, a, b, MARGEN, HOLGURA)
            for p, q in zip(v, v[1:]):
                n = max(2, int(largo(self.mapa, [p, q]) / 0.05))
                for i in range(n + 1):
                    t = i / n
                    d = P._despeje_m(self.mapa, p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)
                    self.assertGreaterEqual(d, MARGEN - 0.03)      # margen duro

    def test_siempre_encuentra_ruta_entre_celdas_libres(self):
        azar = random.Random(7)
        razones = []
        for _ in range(60):
            a, b = azar.sample(self.libres, 2)
            v = planificar(self.mapa, a, b, MARGEN, HOLGURA)
            self.assertEqual((tuple(map(round, v[0])), tuple(map(round, v[-1]))), (a, b))
            razones.append(largo(self.mapa, v) / max(1e-9, largo(self.mapa, viejo(self.mapa, a, b))))
            # Prefiere zonas anchas: a lo sumo un poco más larga que el pipeline anterior.
            self.assertLessEqual(razones[-1], 1.10, f"{a}->{b}")
        self.assertLess(sum(razones) / len(razones), 0.97)         # en promedio, más corta

    def test_parte_de_la_posicion_real_y_llega_al_punto_exacto(self):
        campo = P.campo_de(self.mapa)
        anchas = [c for c in self.libres if campo.en(*c) > 1.0]
        a, b = anchas[0], anchas[-1]
        ini, fin = (a[0] + 0.21, a[1] - 0.17), (b[0] - 0.13, b[1] + 0.28)
        v = planificar(self.mapa, ini, fin, MARGEN, HOLGURA)
        self.assertEqual(v[0], ini)
        self.assertEqual(v[-1], fin)

    def test_puede_salir_si_el_robot_ya_esta_dentro_del_margen(self):
        banco = next(o["celda"] for o in self.mapa.escenario["objetos"] if o["tipo"] == "banco")
        pegado = (banco[0] + 0.5 + 0.30, float(banco[1]))     # a 0,30 m del borde del banco
        despeje = P.campo_de(self.mapa).en(*pegado)
        self.assertTrue(0.2 < despeje < MARGEN, despeje)      # libre pero dentro del margen
        v = planificar(self.mapa, pegado, (3, 3), MARGEN, HOLGURA)
        self.assertEqual(v[-1], (3.0, 3.0))

    def test_rapido_para_replanificar(self):
        bola = tuple(self.mapa.escenario["pelota_oculta"])
        t = time.perf_counter()
        for _ in range(5):
            planificar(self.mapa, (21, 2), bola, MARGEN, HOLGURA)
        self.assertLess((time.perf_counter() - t) / 5, 0.25)    # s por plan

    def test_destino_encerrado_levanta_sin_ruta(self):
        g = [[0] * 7 for _ in range(7)]
        for f in (2, 3, 4):
            for c in (2, 3, 4):
                if (f, c) != (3, 3):
                    g[f][c] = 1                                  # (3,3) rodeado de obstáculos
        m = Mapa("cerrado", g, (0, 0), (6, 6), 1.0, "ESTE", 100)
        with self.assertRaises(SinRuta):
            planificar(m, (0, 0), (3, 3), MARGEN, HOLGURA)

    def test_pasillo_estrecho_usa_solo_el_margen_si_no_hay_otra_forma(self):
        # Pasillo de 1 celda: despeje 0,5 m en el centro. Con margen+holgura = 0,55 no
        # cabe; con solo el margen (0,45) sí. Debe devolver ruta igual.
        g = [[1] * 9 for _ in range(3)]
        for c in range(9):
            g[1][c] = 0
        m = Mapa("pasillo", g, (1, 0), (1, 8), 1.0, "ESTE", 100)
        v = planificar(m, (1, 0), (1, 8), 0.45, 0.10)
        self.assertAlmostEqual(largo(m, v), 8.0, places=3)
        with self.assertRaises(SinRuta):                         # con margen 0,6 no cabe
            planificar(m, (1, 0), (1, 8), 0.60, 0.10)

    def test_penalizacion_opcional_decide_entre_el_paso_estrecho_y_el_rodeo_ancho(self):
        # Pared en la columna 7 con un hueco estrecho de 1 celda en la fila 7
        # (despeje 0,5 m) y un paso ancho de 3 celdas en las filas 11-13.
        g = [[0] * 15 for _ in range(15)]
        for f in range(15):
            if f != 7 and f not in (11, 12, 13):
                g[f][7] = 1
        m = Mapa("hueco", g, (7, 1), (7, 13), 1.0, "ESTE", 100)
        ini, fin = (7.0, 1.0), (7.0, 13.0)
        directa = planificar(m, ini, fin, MARGEN, HOLGURA, 0.0)
        ancha = planificar(m, ini, fin, MARGEN, HOLGURA, 20.0)
        self.assertAlmostEqual(largo(m, directa), 12.0, places=3)   # recta por el hueco
        self.assertGreater(largo(m, ancha), 14.0)                   # rodea por el paso ancho
        campo = P.campo_de(m)

        def despeje_min(v):
            return min(campo.en(p[0] + (q[0] - p[0]) * k / 40, p[1] + (q[1] - p[1]) * k / 40)
                       for p, q in zip(v, v[1:]) for k in range(41))
        self.assertLess(despeje_min(directa), 0.55)                 # pasa raspando el hueco
        # El rodeo evita la zona penalizada: nunca baja de margen + holgura (0,49 m).
        self.assertGreaterEqual(despeje_min(ancha), MARGEN + HOLGURA)

    def test_la_penalizacion_por_defecto_sale_del_mapa(self):
        a, b = (21, 2), tuple(self.mapa.escenario["pelota_oculta"])
        sin = largo(self.mapa, planificar(self.mapa, a, b, MARGEN, HOLGURA))
        self.mapa.escenario["penalizacion_estrecho"] = 5.0
        try:
            con = largo(self.mapa, planificar(self.mapa, a, b, MARGEN, HOLGURA))
        finally:
            del self.mapa.escenario["penalizacion_estrecho"]
        self.assertGreaterEqual(con, sin - 1e-6)
        self.assertEqual(P.PENALIZACION_ESTRECHO, 0.0)

    def test_el_recorrido_de_busqueda_cruza_el_campo_de_obstaculos(self):
        ws = [tuple(w) for w in self.mapa.escenario["busqueda"]]
        for w in ws:
            self.assertTrue(self.mapa.es_transitable(*w), w)
        objetos = [tuple(o["celda"]) for o in self.mapa.escenario["objetos"] if o["tipo"] != "bandera"]
        pts, total, cerca = [tuple(self.mapa.inicio)] + ws, 0.0, set()
        for a, b in zip(pts, pts[1:]):
            v = planificar(self.mapa, a, b, MARGEN, HOLGURA)
            total += largo(self.mapa, v)
            for p, q in zip(v, v[1:]):
                for i in range(41):
                    f, c = p[0] + (q[0] - p[0]) * i / 40, p[1] + (q[1] - p[1]) * i / 40
                    cerca.update(k for k, (of, oc) in enumerate(objetos) if math.hypot(f - of, c - oc) < 1.2)
        self.assertGreaterEqual(len(cerca), 8)                     # pasa junto a muchos obstáculos
        self.assertLess(total, 90.0)                               # sin alargar la misión

    def test_recortar_final(self):
        v = [(0.0, 0.0), (0.0, 10.0), (4.0, 10.0)]
        r = recortar_final(self.mapa, v, 2.0)
        self.assertAlmostEqual(largo(self.mapa, r), 12.0, places=6)
        r = recortar_final(self.mapa, v, 5.0)                    # pasa la esquina
        self.assertAlmostEqual(largo(self.mapa, r), 9.0, places=6)
        self.assertEqual(len(recortar_final(self.mapa, v, 99.0)), 1)

    def test_el_campo_de_despeje_se_cachea_por_mapa(self):
        a = P.campo_de(self.mapa)
        self.assertIs(a, P.campo_de(self.mapa))
        self.assertGreater(a.campo.shape[0], 200)


class SeguidorContinuoTests(unittest.TestCase):
    def correr(self, v0, distancia, bearing=0.0, camara_hz=10.0):
        from sim.camara_go2 import DeteccionPelota, SeguidorPelota
        reloj = [500.0]
        with mock.patch("time.monotonic", lambda: reloj[0]):
            mundo = Mundo(perfil("tp07"))
            mundo.habilitar_carrera_simulada(3.0, 2.0)
            mundo.situar(0.0, 0.0, 0.0)
            mundo.mover_autonomo(v0, 0.0, 0.0, duracion=0.5)
            seg = SeguidorPelota(mundo)
            x, velocidades, estado = 0.0, [], "siguiendo"
            for _ in range(int(20 * camara_hz)):
                for _ in range(5):
                    reloj[0] += 1.0 / camara_hz / 5
                    mundo.avanzar()
                p = mundo.leer()
                d = distancia - (p["x"] - 0.0)
                estado = seg.paso(DeteccionPelota(320, 240, bearing, d, 40))
                velocidades.append(mundo.vx)
                if estado == "distancia_objetivo":
                    break
            return velocidades, estado, mundo.leer()["x"]

    def test_no_se_detiene_hasta_llegar_y_frena_de_forma_continua(self):
        for hz in (10.0, 4.0):
            v, estado, x = self.correr(v0=2.0, distancia=2.2, camara_hz=hz)
            self.assertEqual(estado, "distancia_objetivo", f"{hz} Hz")
            self.assertGreater(min(v[:-1]), 0.05)                 # nunca se queda quieto antes
            # Frenada sin repuntes: la velocidad no vuelve a subir más de 0,3 m/s.
            for a, b in zip(v, v[1:]):
                self.assertLess(b - a, 0.35)
            self.assertAlmostEqual(2.2 - x, 0.75, delta=0.16)      # queda a ~0,75 m

    def test_avanza_mientras_corrige_el_rumbo(self):
        v, _, _ = self.correr(v0=0.0, distancia=3.0, bearing=0.3)
        self.assertGreater(max(v[:10]), 0.2)                      # antes giraba sin avanzar


class TransicionTests(unittest.TestCase):
    def test_del_tramo_por_ruta_al_visual_no_se_frena(self):
        from sim.camara_go2 import ObjetoPercibido
        from sim.mision_go2 import MisionParque
        mapa = cargar(MAPA)
        mundo = Mundo(perfil("tp07"))
        mundo.habilitar_carrera_simulada(3.0, 2.0)
        x, y = mapa.a_mundo(20, 20)
        mundo.situar(x, y, 0.0)
        mision = MisionParque(mundo, mapa, margen_m=0.39)
        mision.estado = "APROXIMANDO_PELOTA"
        mundo.mover_autonomo(2.0, 0.0, 0.0, duracion=0.5)
        bola = ObjetoPercibido("nav_target_ball", "pelota", 320, 240, 0.0, 2.0, 40)
        estados = [mision.paso([bola]) for _ in range(3)]
        self.assertEqual(estados[-1], "ACERCANDOSE")
        self.assertNotIn("DETECTANDO_PELOTA", estados)
        self.assertGreater(mundo.vx, 0.3)                          # sigue en marcha


if __name__ == "__main__":
    unittest.main()
