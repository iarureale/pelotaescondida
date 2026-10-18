"""Ruta dinámica: solo se dibuja lo que falta recorrer y lo recorrido desaparece."""
import math
import os
import time
import unittest
from unittest import mock

from sim import dibujo
from sim.grilla import Mapa
from sim.mundo import Mundo
from sim.navegador_astar import NavegadorAstar
from sim.safety import perfil

L = [(0.0, 0.0), (10.0, 0.0), (10.0, 6.0)]        # 16 m con una esquina


def longitud(puntos):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(puntos, puntos[1:]))


class TrazosTests(unittest.TestCase):
    def test_sin_progreso_se_ve_toda_la_ruta_con_fundido_cerca_del_robot(self):
        trazos = dibujo.trazos_ruta(L, 0.0)
        self.assertGreater(len(trazos), 25)
        self.assertLessEqual(len(trazos), 16 / dibujo.ESTILO_TRAZOS.paso_m + 2)   # no se acumulan
        # El más lejano es opaco; el más cercano al robot casi transparente.
        self.assertGreater(trazos[-1][1], 0.99)
        self.assertLess(trazos[0][1], 0.5)

    def test_nada_queda_dibujado_detras_del_robot(self):
        for progreso in (1.0, 4.3, 9.9, 10.0, 12.5):
            for puntos, _ in dibujo.trazos_ruta(L, progreso):
                # Cada punto debe estar a >= progreso medido sobre la ruta.
                for x, y in puntos:
                    s = x if y <= 1e-9 and x <= 10.0 + 1e-9 else 10.0 + y
                    self.assertGreaterEqual(s, progreso - 1e-6)

    def test_al_llegar_la_ruta_desaparece(self):
        self.assertEqual(dibujo.trazos_ruta(L, 16.0), [])
        self.assertEqual(dibujo.trazos_ruta(L, float("inf")), [])
        self.assertEqual(dibujo.trazos_ruta([], 0.0), [])

    def test_desaparicion_progresiva_sin_cortes_bruscos(self):
        # "Tinta visible" = largo x opacidad. Al avanzar 2 cm no cambia más de ~6 cm.
        previo = None
        p = 0.0
        while p < 16.0:
            tinta = sum(longitud(pts) * op for pts, op in dibujo.trazos_ruta(L, p))
            if previo is not None:
                self.assertLess(abs(tinta - previo), 0.07, f"salto en progreso={p:.2f}")
            previo = tinta
            p += 0.02

    def test_trazos_que_cruzan_la_esquina_la_respetan(self):
        esquina = [(0.0, 0.0), (10.1, 0.0), (10.1, 6.0)]      # 10,1 cae dentro de un trazo
        trazos = dibujo.trazos_ruta(esquina, 0.0)
        con_esquina = [pts for pts, _ in trazos if len(pts) == 3]
        self.assertEqual(len(con_esquina), 1)
        self.assertEqual(con_esquina[0][1], (10.1, 0.0))


class EscenaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mujoco
        cls.mj = mujoco
        cls.model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")

    def test_estilo_trazos_escribe_capsulas_sobre_el_piso_y_reutiliza_los_slots(self):
        scn = self.mj.MjvScene(self.model, maxgeom=500)
        n1 = dibujo.actualizar_ruta_pendiente(scn, 20, L, 0.0, dibujo.ESTILO_TRAZOS)
        self.assertGreater(n1, 20)
        for i in range(20, n1):
            g = scn.geoms[i]
            self.assertEqual(g.type, self.mj.mjtGeom.mjGEOM_CAPSULE)
            self.assertAlmostEqual(g.pos[2], dibujo.ESTILO_TRAZOS.z, places=6)
            self.assertGreater(g.pos[2] - dibujo.ESTILO_TRAZOS.ancho_m / 2, 0.032)    # sobre el césped
        # Muchos cuadros seguidos no acumulan geometrías.
        for p in [i * 0.1 for i in range(200)]:
            n = dibujo.actualizar_ruta_pendiente(scn, 20, L, p, dibujo.ESTILO_TRAZOS)
            self.assertLessEqual(n, n1 + 2)
        self.assertEqual(scn.ngeom, n)

    def test_ruta_vacia_borra_todo_lo_dinamico(self):
        scn = self.mj.MjvScene(self.model, maxgeom=500)
        dibujo.actualizar_ruta_pendiente(scn, 20, L, 0.0)
        self.assertEqual(dibujo.actualizar_ruta_pendiente(scn, 20, [], 0.0), 20)
        self.assertEqual(scn.ngeom, 20)

    def test_respeta_maxgeom(self):
        scn = self.mj.MjvScene(self.model, maxgeom=30)
        self.assertLessEqual(dibujo.actualizar_ruta_pendiente(scn, 25, L, 0.0), 30)

    def test_patio_sin_baldosas_ahorra_cientos_de_geoms(self):
        ruta = os.path.join(os.path.dirname(__file__), "..", "..", "mi_desarrollo",
                            "mapas", "patio_uade_go2.json")
        from sim.grilla import cargar
        mapa = cargar(ruta)
        scn = self.mj.MjvScene(self.model, maxgeom=2000)
        con = dibujo.dibujar(scn, mapa, mostrar_ruta=False, mostrar_obstaculos=False)
        sin = dibujo.dibujar(scn, mapa, mostrar_ruta=False, mostrar_obstaculos=False,
                             mostrar_baldosas=False)
        self.assertGreater(con, 700)
        self.assertLess(sin, 10)


class FlechasTests(unittest.TestCase):
    """Estilo por defecto: flechas planas orientadas con la ruta."""

    def test_flechas_siguen_la_direccion_de_cada_tramo(self):
        fl = dibujo.flechas_ruta(L, 0.0)
        self.assertGreater(len(fl), 10)
        self.assertLessEqual(len(fl), 16 / dibujo.ESTILO_FLECHAS.paso_m + 1)   # no se acumulan
        for x, y, rumbo, _ in fl:
            if y < 1e-6 and x < 9.5:                       # primer tramo: hacia +x
                self.assertAlmostEqual(rumbo, 0.0, places=3)
            if x > 10.5 - 1e-6 and y > 0.5:                # último tramo: hacia +y
                self.assertAlmostEqual(rumbo, math.pi / 2, places=3)

    def test_las_flechas_de_atras_desaparecen_y_las_cercanas_se_desvanecen(self):
        for progreso in (0.0, 3.3, 9.0, 12.0):
            for x, y, _, op in dibujo.flechas_ruta(L, progreso):
                s = x if y < 1e-6 else 10.0 + y
                self.assertGreater(s, progreso)             # nada detrás del robot
        fl = dibujo.flechas_ruta(L, 5.0)
        self.assertLess(fl[0][3], fl[-1][3])                # la más cercana, más tenue
        self.assertGreater(fl[-1][3], 0.99)

    def test_al_llegar_no_queda_ninguna_flecha(self):
        for p in (15.96, 16.0, float("inf")):
            self.assertEqual(dibujo.flechas_ruta(L, p), [])

    def test_aparicion_y_desaparicion_continuas(self):
        # Suma de opacidad: no salta al avanzar 2 cm.
        previo, p = None, 0.0
        while p < 16.0:
            total = sum(op for *_, op in dibujo.flechas_ruta(L, p))
            if previo is not None:
                self.assertLess(abs(total - previo), 0.6, f"salto en {p:.2f}")
            previo, p = total, p + 0.02

    def test_geometria_de_la_flecha_es_un_chevron_plano_sobre_el_piso(self):
        import mujoco
        modelo = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        scn = mujoco.MjvScene(modelo, maxgeom=800)
        n = dibujo.actualizar_ruta_pendiente(scn, 10, L, 0.0)
        fl = dibujo.flechas_ruta(L, 0.0)
        self.assertEqual(n - 10, 2 * len(fl))                # 2 brazos por flecha
        est = dibujo.ESTILO_FLECHAS
        for i in range(10, n):
            g = scn.geoms[i]
            self.assertEqual(g.type, mujoco.mjtGeom.mjGEOM_BOX)
            self.assertAlmostEqual(g.pos[2], est.z, places=6)
            self.assertLess(g.size[2], 0.005)                # plana
            self.assertGreater(g.pos[2] - g.size[2], 0.032)  # sobre el césped
        # Primera flecha (a lo largo de +x): los brazos quedan a ambos lados y detrás de la punta.
        a, b = scn.geoms[10], scn.geoms[11]
        self.assertAlmostEqual(a.pos[1] + b.pos[1], 2 * fl[0][1], places=3)   # simétricos en y
        self.assertGreater(abs(a.pos[1] - b.pos[1]), 0.1)
        for _ in range(100):                                 # sin acumulación
            n2 = dibujo.actualizar_ruta_pendiente(scn, 10, L, _ * 0.1)
            self.assertLessEqual(n2, n)

    def test_estilo_configurable_y_ruta_vacia(self):
        import mujoco
        modelo = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        scn = mujoco.MjvScene(modelo, maxgeom=800)
        ancho = dibujo.EstiloRuta(paso_m=1.6, largo_m=0.5)
        grande = dibujo.actualizar_ruta_pendiente(scn, 0, L, 0.0, dibujo.ESTILO_FLECHAS)
        raro = dibujo.actualizar_ruta_pendiente(scn, 0, L, 0.0, ancho)
        self.assertLess(raro, grande)
        self.assertEqual(dibujo.actualizar_ruta_pendiente(scn, 7, [], 0.0), 7)


class PolilineaTests(unittest.TestCase):
    def test_suavizado_redondea_esquinas_y_conserva_extremos(self):
        s = dibujo.suavizar_polilinea(L, 0.9)
        self.assertEqual((s[0], s[-1]), (L[0], L[-1]))
        self.assertLess(longitud(s), longitud(L))            # recorta la esquina
        self.assertGreater(longitud(s), 0.97 * longitud(L))  # pero muy poco
        # Solo se desvía cerca de la esquina (10, 0) y nunca más de ~0,4 m.
        def dist(p):
            return min(abs(p[1]) if 0 <= p[0] <= 10 else 99, abs(p[0] - 10) if 0 <= p[1] <= 6 else 99)
        self.assertLess(max(dist(p) for p in s), 0.4)
        self.assertTrue(all(dist(p) < 1e-9 for p in s if p[0] < 9.0 or p[1] > 1.0))
        self.assertEqual(dibujo.suavizar_polilinea(L, 0.0), L)
        self.assertEqual(dibujo.suavizar_polilinea([(0, 0), (1, 1)], 0.9), [(0, 0), (1, 1)])

    def test_el_redondeo_no_deforma_tramos_cortos(self):
        zigzag = [(0, 0), (0.6, 0), (0.6, 0.6), (1.2, 0.6)]    # tramos de 0,6 m
        s = dibujo.suavizar_polilinea(zigzag, 0.9)
        self.assertEqual((s[0], s[-1]), (zigzag[0], zigzag[-1]))
        # el recorte es como mucho la mitad de cada tramo (0,3 m)
        self.assertTrue(all(0 <= p[0] <= 1.2 and 0 <= p[1] <= 0.6 for p in s))

    def test_proyectar_avanza_y_nunca_retrocede(self):
        pol = dibujo.suavizar_polilinea(L, 2)
        prog = 0.0
        for x in [i * 0.1 for i in range(100)]:               # robot avanza por el primer tramo
            prog = max(prog, dibujo.proyectar(pol, x, 0.05, prog))
        self.assertGreater(prog, 8.5)
        # Aunque la pose "vuelva" cerca del inicio, el mínimo manda.
        self.assertGreaterEqual(dibujo.proyectar(pol, 0.5, 0.0, prog), prog - 0.5 - 1e-9)


class ProgresoNavegadorTests(unittest.TestCase):
    def test_progreso_crece_con_el_robot_y_se_completa(self):
        reloj = [100.0]
        with mock.patch.object(time, "monotonic", lambda: reloj[0]):
            mapa = Mapa("libre", [[0] * 12 for _ in range(12)], (0, 0), (0, 8), 1.0, "ESTE", 100)
            mundo = Mundo(perfil("tp07"))
            mundo.habilitar_carrera_simulada(3.0, 2.0)
            x, y = mapa.a_mundo(0, 0)
            mundo.situar(x, y, 0.0)
            nav = NavegadorAstar(mundo, mapa, [(0, c) for c in range(9)] + [(r, 8) for r in range(1, 5)])
            self.assertAlmostEqual(longitud(nav.polilinea()), 12.0, places=3)
            previo, estado = 0.0, "navegando"
            while estado == "navegando" and reloj[0] < 200:
                reloj[0] += 0.02
                mundo.avanzar()
                estado = nav.paso()
                p = nav.progreso_m()
                if estado == "navegando":
                    self.assertGreaterEqual(p, previo - 1e-6)
                    previo = p
            self.assertEqual(estado, "completado")
            self.assertEqual(nav.progreso_m(), float("inf"))
            self.assertEqual(dibujo.trazos_ruta(nav.polilinea(), nav.progreso_m()), [])


if __name__ == "__main__":
    unittest.main()
