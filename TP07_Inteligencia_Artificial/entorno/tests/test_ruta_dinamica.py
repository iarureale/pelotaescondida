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
        self.assertLessEqual(len(trazos), 16 / dibujo.PASO_TRAZO_M + 2)   # no se acumulan
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

    def test_escribe_capsulas_sobre_el_piso_y_reutiliza_los_slots(self):
        scn = self.mj.MjvScene(self.model, maxgeom=500)
        n1 = dibujo.actualizar_ruta_pendiente(scn, 20, L, 0.0)
        self.assertGreater(n1, 20)
        for i in range(20, n1):
            g = scn.geoms[i]
            self.assertEqual(g.type, self.mj.mjtGeom.mjGEOM_CAPSULE)
            self.assertAlmostEqual(g.pos[2], dibujo.Z_TRAZO, places=6)
            self.assertGreater(g.pos[2] - dibujo.RADIO_TRAZO, 0.032)    # sobre el césped
        # Muchos cuadros seguidos no acumulan geometrías.
        for p in [i * 0.1 for i in range(200)]:
            n = dibujo.actualizar_ruta_pendiente(scn, 20, L, p)
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
