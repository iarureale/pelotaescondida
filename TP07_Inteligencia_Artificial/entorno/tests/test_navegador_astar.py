"""Seguimiento de ruta: llegada, giros progresivos y detección de atasco."""
import math
import unittest
from unittest import mock

from sim import navegador_astar
from sim.grilla import Mapa
from sim.mundo import Mundo
from sim.navegador_astar import NavegadorAstar
from sim.safety import perfil


class RelojFalso:
    def __init__(self):
        self.t = 100.0

    def monotonic(self):
        return self.t


def mapa_libre(filas=12, columnas=12):
    return Mapa("libre", [[0] * columnas for _ in range(filas)], (0, 0),
                (filas - 1, columnas - 1), 1.0, "ESTE", 100)


class NavegadorTests(unittest.TestCase):
    def correr(self, ruta, yaw0=0.0, limite_s=120.0, hasta=None):
        reloj = RelojFalso()
        with mock.patch("time.monotonic", reloj.monotonic):
            mundo = Mundo(perfil("tp07"))
            mundo.habilitar_carrera_simulada(3.0, 2.0)
            mapa = mapa_libre()
            x, y = mapa.a_mundo(*ruta[0])
            mundo.situar(x, y, yaw0)
            nav = NavegadorAstar(mundo, mapa, ruta)
            traza, estado = [], "navegando"
            while reloj.t - 100.0 < limite_s and estado == "navegando":
                for _ in range(2):          # 50 Hz
                    reloj.t += 0.02
                    mundo.avanzar()
                    estado = nav.paso()
                p = mundo.leer()
                traza.append((reloj.t, p["x"], p["y"], p["yaw"], mundo.vx, mundo.vyaw))
            return nav, mundo, traza, estado

    def test_llega_al_destino_y_completa(self):
        ruta = [(0, c) for c in range(0, 9)] + [(r, 8) for r in range(1, 7)]
        nav, mundo, traza, estado = self.correr(ruta)
        self.assertEqual(estado, "completado")
        destino = mapa_libre().a_mundo(*ruta[-1])
        self.assertLess(math.hypot(traza[-1][1] - destino[0],
                                   traza[-1][2] - destino[1]), 0.15)
        self.assertFalse(mundo.leer()["moviendose"])

    def test_no_se_aparta_de_la_ruta_en_las_esquinas(self):
        # Ruta en L: el robot no debe cortar la esquina más de ~0,7 m.
        ruta = [(0, c) for c in range(0, 9)] + [(r, 8) for r in range(1, 7)]
        _, _, traza, _ = self.correr(ruta)
        mapa = mapa_libre()
        puntos = [mapa.a_mundo(*p) for p in ruta]
        peor = max(min(math.hypot(x - px, y - py) for px, py in puntos)
                   for _, x, y, _, _, _ in traza)
        self.assertLess(peor, 0.7)

    def test_giros_y_velocidad_son_progresivos(self):
        ruta = [(0, c) for c in range(0, 9)] + [(r, 8) for r in range(1, 7)]
        _, _, traza, _ = self.correr(ruta, yaw0=math.pi)   # arranca de espaldas
        dt = 0.04
        salto_w = max(abs(b[5] - a[5]) for a, b in zip(traza, traza[1:]))
        salto_v = max(abs(b[4] - a[4]) for a, b in zip(traza, traza[1:]))
        # Aceleraciones acotadas (rampas): sin golpes de velocidad.
        self.assertLessEqual(salto_w, NavegadorAstar.ACEL_ANGULAR * dt * 1.05)
        self.assertLessEqual(salto_v, 3.0 * dt * 1.05)
        # Sin oscilación: el yaw no invierte el sentido de giro en el tramo recto.
        tramo_recto = [t for t in traza if t[3] is not None and 2.0 < t[1] < 5.0]
        signos = {1 if t[5] > 0.05 else -1 for t in tramo_recto if abs(t[5]) > 0.05}
        self.assertLessEqual(len(signos), 1)

    def test_detecta_atasco_si_no_progresa(self):
        reloj = RelojFalso()
        with mock.patch("time.monotonic", reloj.monotonic):
            mundo = Mundo(perfil("tp07"))
            mundo.habilitar_carrera_simulada(3.0, 2.0)
            mapa = mapa_libre()
            mundo.situar(0.0, 0.0, 0.0)
            nav = NavegadorAstar(mundo, mapa, [(0, 0), (0, 8)])
            estado = "navegando"
            for _ in range(600):
                reloj.t += 0.02
                # El mundo NO avanza (robot bloqueado): nunca llama a avanzar().
                estado = nav.paso()
                if estado != "navegando":
                    break
            self.assertEqual(estado, "atascado")
            self.assertFalse(mundo.leer()["moviendose"])


if __name__ == "__main__":
    unittest.main()
