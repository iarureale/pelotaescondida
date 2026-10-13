"""Benchmark reproducible de la percepción (cámara + detección).

Uso (desde entorno/):  python herramientas/bench_percepcion.py [N]
RAIZ_PROYECTO=<ruta> permite medir otra copia del proyecto (p. ej. la original).
Sin GPU use MUJOCO_GL=osmesa; los tiempos de render dependen del equipo,
los de detección (CPU) son comparables.
"""
import os, sys, time, statistics, json
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
RAIZ = os.environ.get("RAIZ_PROYECTO", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, RAIZ + "/entorno")
import mujoco
from sim.grilla import cargar
from sim.escena_navegacion import crear_escena, limpiar_escena
from sim import camara_go2

mapa = cargar(RAIZ + "/mi_desarrollo/mapas/patio_uade_go2.json")
base = RAIZ + "/entorno/sim/unitree_mujoco/unitree_robots/go2/scene_uade.xml"
escena, tmp = crear_escena(base, mapa)
try:
    model = mujoco.MjModel.from_xml_path(escena)
finally:
    limpiar_escena(tmp)
data = mujoco.MjData(model)
x, y = mapa.a_mundo(*mapa.inicio)
data.qpos[0], data.qpos[1], data.qpos[2] = x, y, 0.27
data.qpos[3:7] = [1, 0, 0, 0]
mujoco.mj_forward(model, data)
print("ngeom", model.ngeom, "nbody", model.nbody)

cam = camara_go2.CamaraGo2(model)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
t_cap, t_det = [], []
for i in range(N):
    t0 = time.perf_counter(); frame, prof, seg = cam.capturar_sensores(data); t1 = time.perf_counter()
    det = camara_go2.detectar_objetos_segmentados(seg, prof, model, mujoco); t2 = time.perf_counter()
    if i >= 2: t_cap.append(t1 - t0); t_det.append(t2 - t1)
f = lambda v: f"{statistics.median(v)*1000:.1f} ms"
print("captura(3 renders) mediana:", f(t_cap), "| deteccion mediana:", f(t_det))
print("detecciones:", sorted((d.categoria, d.nombre) for d in det)[:6], "n=", len(det))
