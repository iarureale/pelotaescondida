"""Reinicio de la misión (tecla R): estado limpio y sin cuadros de cámara viejos."""
import os
import threading
import time
import unittest
from unittest import mock

from sim import robots, safety
from sim.arrancar import SimuladorOficial
from sim.grilla import cargar
from sim.mundo import Mundo

RAIZ = os.path.join(os.path.dirname(__file__), "..", "..")
MAPA = os.path.join(RAIZ, "mi_desarrollo", "mapas", "patio_uade_go2.json")
REPO = os.path.join(RAIZ, "entorno", "sim", "unitree_mujoco")


def simulador():
    mapa = cargar(MAPA)
    mundo = Mundo(safety.perfil("tp07"))
    x, y = mapa.a_mundo(*mapa.inicio)
    mundo.situar(x, y, mapa.rumbo_inicial())
    mundo.mision_parque = True
    mundo.habilitar_carrera_simulada(3.0, 2.0)
    return SimuladorOficial(mundo, robots.obtener("go2"), REPO, mapa=mapa, verboso=False), mundo


class ReinicioTests(unittest.TestCase):
    def test_reiniciar_deja_la_ruta_dibujada_y_la_marcha_en_cero(self):
        sim, mundo = simulador()
        sim._ruta_vertices = ((1.0, 2.0),)
        sim._ruta_dibujo = [(1.0, 2.0), (3.0, 4.0)]
        sim._ruta_progreso = 9.5                       # progreso de la corrida anterior
        sim._trote.calcular(1.0, 2.0, 0.0, 0.0, 0.33, t=5.0)
        mundo.mover_autonomo(2.0, 0.0, 0.0, duracion=0.5)
        viejo = sim.mision
        sim.reiniciar_mision()
        self.assertIsNot(sim.mision, viejo)
        self.assertEqual((sim._ruta_vertices, sim._ruta_dibujo, sim._ruta_progreso), ((), [], 0.0))
        self.assertFalse(sim._trote.activo())
        self.assertEqual(mundo.leer()["velocidad_m_s"], 0.0)
        x0, y0 = sim.mapa.a_mundo(*sim.mapa.inicio)
        self.assertAlmostEqual(mundo.leer()["x"], x0, places=3)

    def test_se_descartan_cuadros_de_camara_capturados_antes_del_reinicio(self):
        sim, _ = simulador()
        sim._pc_lock, sim._pc_error, sim._pc_ultimo = threading.Lock(), None, 0
        sim._hilo_percepcion = object()                # modo ventana (hilo activo)
        pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        sim._pc_resultado = (1, None, [], 0, pose, time.monotonic())
        sim.reiniciar_mision()                         # fija _t_reinicio = ahora
        sim._pc_resultado = (2, None, [], 0, pose, time.monotonic() - 5.0)   # capturado antes
        with mock.patch.object(sim, "_procesar_percepcion") as procesar:
            sim._paso_seguimiento_visual()
            procesar.assert_not_called()
            sim._pc_resultado = (3, None, [], 0, pose, time.monotonic() + 0.01)   # posterior
            sim._paso_seguimiento_visual()
            procesar.assert_called_once()


if __name__ == "__main__":
    unittest.main()
