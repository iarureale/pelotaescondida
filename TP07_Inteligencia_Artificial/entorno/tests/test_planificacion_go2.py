import unittest

from sim.grilla import Mapa, validar_ruta
from sim.planificacion_go2 import SinRuta, astar
from sim.mundo import Mundo
from sim.safety import perfil
from sim.navegador_astar import NavegadorAstar


def mapa_demo():
    grid = [[0] * 15 for _ in range(14)]
    for fila in range(1, 12):
        grid[fila][7] = 1
    return Mapa("demo", grid, (12, 2), (2, 12), 0.5, "ESTE", 80)


class AstarGo2Tests(unittest.TestCase):
    def test_encuentra_ruta_y_no_pisa_obstaculos(self):
        mapa = mapa_demo()
        ruta = astar(mapa)
        self.assertEqual(ruta[0], mapa.inicio)
        self.assertEqual(ruta[-1], mapa.destino)
        self.assertEqual(validar_ruta(mapa, ruta), [])
        self.assertGreater(len(ruta), 20)  # rodea la pared

    def test_margen_inflado_obliga_a_rodear_extremo(self):
        mapa = mapa_demo()
        ruta = astar(mapa, margen_m=0.35)
        self.assertEqual(ruta[0], mapa.inicio)
        self.assertEqual(ruta[-1], mapa.destino)
        self.assertEqual(validar_ruta(mapa, ruta), [])
        # No pasa pegada a la pared: evita las celdas vecinas a su cara.
        self.assertNotIn((10, 7), ruta)

    def test_objetivo_bloqueado_o_inalcanzable_falla(self):
        grid = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        mapa = Mapa("bloqueo", grid, (0, 0), (0, 2), 0.5, "ESTE", 20)
        with self.assertRaises(SinRuta):
            astar(mapa)

    def test_navegador_alcanza_waypoint_con_estado_del_simulador(self):
        mapa = Mapa("recta", [[0, 0, 0]], (0, 0), (0, 2), 0.5, "ESTE", 10)
        mundo = Mundo(perfil("tp07"))
        mundo.situar(*mapa.a_mundo(*mapa.inicio), 0.0)
        ruta = astar(mapa)
        navegador = NavegadorAstar(mundo, mapa, ruta)
        for _ in range(1500):
            if navegador.paso() == "completado":
                break
            mundo.avanzar(dt=0.02)
        self.assertEqual(navegador.estado, "completado")
        self.assertAlmostEqual(mundo.leer()["x"], 1.0, delta=0.13)
        self.assertEqual(mundo.leer()["accion"], "quieto")

    def test_rechaza_margen_negativo(self):
        with self.assertRaises(ValueError):
            astar(mapa_demo(), margen_m=-0.1)


if __name__ == "__main__":
    unittest.main()





