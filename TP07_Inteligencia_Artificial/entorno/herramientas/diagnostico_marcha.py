"""Mide contacto y deslizamiento de las patas del Go2 con el modelo real de MuJoCo.

Uso (desde entorno/):  python herramientas/diagnostico_marcha.py

La locomoción del paquete es cinemática (no hay dinámica ni controlador de
fuerzas): la pose de las patas se escribe en qpos y se evalúa con mj_forward.
Por eso el "contacto" se mide geométricamente sobre las esferas de pie (r=0,022):

  contacto   : base de la esfera a < 4 mm del piso
  arrastre   : cuánto se mueve (en el mundo) el pie durante un apoyo; un pie
               apoyado no debería moverse (~0 cm por paso)
  flotación  : altura media de la base de la esfera mientras el pie "debería"
               estar apoyado, y fracción del tiempo con algún pie en el suelo
Usa reloj virtual: el resultado no depende de la velocidad de la CPU.
"""
import os, sys, time
os.environ.setdefault("MUJOCO_GL", "osmesa")
RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, RAIZ + "/entorno")
import warnings; warnings.simplefilter("ignore")
import numpy as np

class Reloj:
    t = 1000.0
time.monotonic = lambda: Reloj.t
time.perf_counter = lambda: Reloj.t

import mujoco
from sim import safety, robots
from sim.mundo import Mundo
from sim.grilla import cargar
from sim.arrancar import SimuladorOficial

DT = 0.01
PIES = ("FL", "FR", "RL", "RR")


def construir():
    mapa = cargar(RAIZ + "/mi_desarrollo/mapas/patio_uade_go2.json")
    mundo = Mundo(safety.perfil("tp07"))
    mundo.situar(10.0, -10.0, 0.0)
    mundo.habilitar_carrera_simulada(3.0, 2.0)
    sim = SimuladorOficial(mundo, robots.obtener("go2"), RAIZ + "/entorno/sim/unitree_mujoco",
                           mapa=mapa, verboso=False)
    geoms = []
    for p in PIES:
        b = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, p + "_calf")
        g = next(g for g in range(sim.model.ngeom)
                 if sim.model.geom_bodyid[g] == b and sim.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE)
        geoms.append(g)
    return sim, mundo, geoms


def medir(vx, vy=0.0, vyaw=0.0, duracion=6.0, rampa=1.0):
    sim, mundo, geoms = construir()
    radio = sim.model.geom_size[geoms[0]][0]
    n = int(duracion / DT)
    pos = np.zeros((n, 4, 3)); cuerpo = np.zeros((n, 3))
    for i in range(n):
        Reloj.t += DT
        k = min(1.0, (i * DT) / rampa)
        mundo.mover_autonomo(vx * k, vy * k, vyaw * k, duracion=0.5)
        mundo.avanzar(); sim._escribir_pose()
        for j, g in enumerate(geoms):
            pos[i, j] = sim.data.geom_xpos[g]
        cuerpo[i] = sim.data.qpos[:3]
    sel = slice(int(n * 0.4), n)                        # régimen estable
    z = pos[sel, :, 2] - radio                           # base de la esfera
    contacto = z[1:] < 0.004
    v_cuerpo = float(np.linalg.norm(np.diff(cuerpo[sel, :2], axis=0), axis=1).mean() / DT)
    # Arrastre por apoyo: para cada tramo continuo en contacto, cuánto se desplaza
    # el pie (en el mundo) entre el 20 % y el 80 % del tramo. Un pie apoyado no
    # debería moverse. Se excluyen los extremos (despegue/aterrizaje).
    arrastres = []
    for j in range(4):
        c = contacto[:, j]
        i = 0
        while i < len(c):
            if not c[i]:
                i += 1; continue
            k = i
            while k < len(c) and c[k]:
                k += 1
            if k - i >= 8:
                a, b = i + int(0.2 * (k - i)), i + int(0.8 * (k - i))
                arrastres.append(float(np.linalg.norm(pos[sel][a + 1, j, :2] - pos[sel][b + 1, j, :2])))
            i = k
    desliz = float(np.mean(arrastres)) if arrastres else float("nan")
    return dict(v_cuerpo=v_cuerpo, z_min=float(z.min()), z_max=float(z.max()),
                frac_contacto=float(contacto.mean()),
                pies_en_suelo=float(contacto.sum(axis=1).mean()),
                deslizamiento=desliz,
                altura_cuerpo_min=float(cuerpo[sel, 2].min()), altura_cuerpo_max=float(cuerpo[sel, 2].max()))


if __name__ == "__main__":
    print(f"{'orden':>14} | {'v real':>6} | {'pies en suelo':>13} | {'contacto':>8} | {'arrastre/paso':>13} | {'z_pie min/max (cm)':>18} | {'z cuerpo (cm)':>13}")
    casos = [(0.3, 0, 0), (1.0, 0, 0), (2.0, 0, 0), (3.0, 0, 0), (0, 0, 0.45), (1.0, 0, 0.4)]
    for vx, vy, w in casos:
        r = medir(vx, vy, w)
        print(f"{f'v={vx} w={w}':>14} | {r['v_cuerpo']:6.2f} | {r['pies_en_suelo']:13.2f} | {r['frac_contacto']*100:7.0f}% | "
              f"{r['deslizamiento']*100:9.1f} cm | {r['z_min']*100:8.1f} / {r['z_max']*100:5.1f} | "
              f"{r['altura_cuerpo_min']*100:5.1f}-{r['altura_cuerpo_max']*100:4.1f}")
