"""Perfil del bucle del visor (50 Hz): cuánto cuesta cada etapa por iteración.

Uso (desde entorno/):  python herramientas/perfil_bucle.py [segundos_virtuales]

Mide en CPU (sin ventana, reloj virtual): integración del mundo, escritura de
pose + marcha, seguimiento de ruta, ruta dinámica (flechas) y el procesamiento
de un cuadro de cámara. El render de la cámara (GPU/driver) no se mide acá; va en
un hilo aparte (ver arrancar._bucle_percepcion). Presupuesto: 20 ms por iteración.
"""
import os, sys, time
os.environ.setdefault("MUJOCO_GL", "osmesa")
RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, RAIZ + "/entorno")
import warnings; warnings.simplefilter("ignore")

class Reloj:
    t = 1000.0
_perf = time.perf_counter
time.monotonic = lambda: Reloj.t

import mujoco
from sim import safety, robots, dibujo
from sim.mundo import Mundo
from sim.grilla import cargar
from sim.arrancar import SimuladorOficial
from sim.camara_go2 import CamaraGo2, detectar_objetos_segmentados, validar_pelota_con_rgb


class VisorFalso:
    """Mismo contrato que usa arrancar._actualizar_ruta_dinamica."""
    def __init__(self, model):
        self.user_scn = mujoco.MjvScene(model, maxgeom=2000)
    def lock(self):
        import contextlib
        return contextlib.nullcontext()


def construir():
    mapa = cargar(RAIZ + "/mi_desarrollo/mapas/patio_uade_go2.json")
    mundo = Mundo(safety.perfil("tp07"))
    x0, y0 = mapa.a_mundo(*mapa.inicio)
    mundo.situar(x0, y0, mapa.rumbo_inicial())
    mundo.mision_parque = True
    mundo.habilitar_carrera_simulada(3.0, 2.0)
    sim = SimuladorOficial(mundo, robots.obtener("go2"), RAIZ + "/entorno/sim/unitree_mujoco",
                           mapa=mapa, verboso=False)

    class Nula:
        def publicar(self, *a, **k): pass
        def iniciar(self): pass
    sim.vista_camara = Nula()
    return sim, mundo, mapa


if __name__ == "__main__":
    segundos = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    sim, mundo, mapa = construir()
    visor = VisorFalso(sim.model)
    sim._n_dibujo_base = dibujo.dibujar(visor.user_scn, mapa, mostrar_ruta=False, mostrar_obstaculos=False,
                                        mostrar_baldosas=False)
    etapas = {"mundo.avanzar": 0.0, "_escribir_pose (marcha+mj_forward)": 0.0, "mision.control": 0.0,
              "ruta dinámica (flechas)": 0.0}
    n = 0
    dt = 0.02
    sim.mision.estado  # arranca NAVEGANDO
    pr = cProfile.Profile()
    t_fin = Reloj.t + segundos
    while Reloj.t < t_fin and not sim.mision.estado.startswith("BACKFLIP"):
        Reloj.t += dt
        t0 = _perf(); mundo.avanzar(); t1 = _perf()
        sim._escribir_pose(); t2 = _perf()
        sim.mision.control(); t3 = _perf()
        sim._actualizar_ruta_dinamica(visor); t4 = _perf()
        for k, d in zip(etapas, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
            etapas[k] += d
        n += 1
        if n % 5 == 0:                         # un cuadro de "cámara" cada 0,1 s
            sim.mision.paso([])
    print(f"{n} iteraciones de {dt*1000:.0f} ms virtuales (presupuesto: 20,0 ms cada una)")
    total = 0.0
    for k, v in etapas.items():
        print(f"  {k:38s} {v / n * 1000:7.3f} ms/iter")
        total += v / n * 1000
    print(f"  {'TOTAL CPU por iteración':38s} {total:7.3f} ms  ({100 * total / 20:.1f} % del presupuesto)")
    # Procesamiento de un cuadro de cámara (numpy), sin el render.
    cam = CamaraGo2(sim.model)
    mujoco.mj_forward(sim.model, sim.data)
    frame, prof, seg = cam.capturar_sensores(sim.data)
    t = _perf()
    for _ in range(20):
        percibidas = detectar_objetos_segmentados(seg, prof, sim.model, mujoco)
        validar_pelota_con_rgb(percibidas, frame)
    print(f"  detección + validación de un cuadro:   {(_perf() - t) / 20 * 1000:7.2f} ms (hilo de percepción)")
    t = _perf()
    for _ in range(5):
        cam.capturar_sensores(sim.data)
    print(f"  render de la cámara (software, 3 pasadas): {(_perf() - t) / 5 * 1000:5.0f} ms (con GPU real es mucho menor)")
