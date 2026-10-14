"""Misión del parque, sin ventana y con RELOJ VIRTUAL (determinista).

Uso (desde entorno/):  python -u herramientas/e2e_reloj_virtual.py [limite_s] [repeticiones]

El simulador usa time.monotonic() para integrar el movimiento y para los
temporizadores de la misión. Acá se reemplaza por un reloj que avanza un paso
fijo por ciclo, así el resultado NO depende de la velocidad de la CPU ni del
render por software: la misma ruta da siempre el mismo recorrido. Con
repeticiones > 1 se verifica además que el reinicio repite la trayectoria.
Cada ciclo corresponde a un frame de cámara (DT_CAMARA) con CICLOS_CONTROL
ejecuciones del lazo rápido de seguimiento, como en el visor real.
"""
import os, sys, time, math, json
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
RAIZ = os.environ.get("RAIZ_PROYECTO", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, RAIZ + "/entorno")
import warnings; warnings.simplefilter("ignore")

class Reloj:
    t = 1000.0
    @classmethod
    def monotonic(cls): return cls.t
    @classmethod
    def perf_counter(cls): return cls.t
time.monotonic = Reloj.monotonic
time.perf_counter = Reloj.perf_counter

from sim import safety, robots
from sim.mundo import Mundo
from sim.grilla import cargar
from sim.arrancar import SimuladorOficial

DT_CAMARA = float(os.environ.get("DT_CAMARA", "0.10"))
CICLOS_CONTROL = int(os.environ.get("CICLOS_CONTROL", "5"))
MAPA = os.environ.get("MAPA", "patio_uade_go2.json")
FINALES = ("ERROR", "FINALIZADA", "BACKFLIP_SIMULADO", "BACKFLIP_OMITIDO", "BACKFLIP_NO_DISPONIBLE")

if os.environ.get("SIN_PELOTA"):
    # Prueba de la ruta de búsqueda completa: la cámara "no ve" la pelota.
    from sim import camara_go2, arrancar as _arr
    _orig = camara_go2.detectar_objetos_segmentados
    def _sin_pelota(*a, **k):
        return [o for o in _orig(*a, **k) if o.categoria != "pelota"]
    camara_go2.detectar_objetos_segmentados = _sin_pelota

class VistaNula:
    def publicar(self, *a, **k): pass
    def iniciar(self): pass

def correr(sim, mundo, limite_s):
    t0 = Reloj.t; estados = []; traza = []; max_salto = 0.0; prev = None
    giro_prev = 0.0; saltos_giro = 0.0; vel_prev = 0.0; saltos_v = 0.0
    while Reloj.t - t0 < limite_s:
        for _ in range(CICLOS_CONTROL):
            Reloj.t += DT_CAMARA / CICLOS_CONTROL
            mundo.avanzar(); sim._escribir_pose()
            if sim.mision is not None: sim.mision.control()
        sim._paso_seguimiento_visual()
        a = mundo.leer()
        est = sim.mision.estado
        if not estados or estados[-1][1] != est: estados.append((round(Reloj.t - t0, 1), est))
        traza.append((round(Reloj.t - t0, 1), a["x"], a["y"], a["yaw"], est))
        if prev is not None:
            max_salto = max(max_salto, math.hypot(a["x"] - prev[0], a["y"] - prev[1]))
            dw = abs((a["yaw"] - prev[2] + math.pi) % (2 * math.pi) - math.pi) / DT_CAMARA
            saltos_giro = max(saltos_giro, abs(dw - giro_prev)); giro_prev = dw
        prev = (a["x"], a["y"], a["yaw"])
        if est in FINALES:
            if not hasattr(sim, "_fin"): sim._fin = Reloj.t
            if Reloj.t - sim._fin > 3.0: break
    return dict(estados=estados, traza=traza, max_salto_m=max_salto, duracion=Reloj.t - t0)

if __name__ == "__main__":
    lim = float(sys.argv[1]) if len(sys.argv) > 1 else 300
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    mapa = cargar(RAIZ + "/mi_desarrollo/mapas/" + MAPA)
    robot = robots.obtener("go2"); mundo = Mundo(safety.perfil("tp07"))
    x0, y0 = mapa.a_mundo(*mapa.inicio); mundo.situar(x0, y0, mapa.rumbo_inicial())
    mundo.mision_parque = True; mundo.habilitar_carrera_simulada(float(os.environ.get("VEL", "3.0")), 2.0)
    sim = SimuladorOficial(mundo, robot, RAIZ + "/entorno/sim/unitree_mujoco", mapa=mapa, verboso=False)
    sim.vista_camara = VistaNula()
    resultados = []
    for i in range(reps):
        if i: 
            sim.reiniciar_mision(); sim.vista_camara = VistaNula()
            if hasattr(sim, "_fin"): del sim._fin
        t = time.time(); r = correr(sim, mundo, lim); r["pared_s"] = time.time() - t
        a = mundo.leer(); bx, by = mapa.a_mundo(*mapa.escenario["pelota_oculta"])
        r.update(final=sim.mision.estado, motivo=sim.mision.motivo, metricas=dict(sim.mision.metricas),
                 dist_pelota=math.hypot(a["x"] - bx, a["y"] - by))
        # Despeje mínimo entre el centro del robot y las celdas ocupadas del mapa.
        c = mapa.tamano_celda / 2
        despeje = min(
            math.hypot(max(0.0, abs(px - mapa.a_mundo(f, col)[0]) - c),
                       max(0.0, abs(py - mapa.a_mundo(f, col)[1]) - c))
            for _, px, py, _, _ in r["traza"][::3]
            for f in range(mapa.filas) for col in range(mapa.columnas)
            if mapa.celda(f, col) == 1
            and abs(px - mapa.a_mundo(f, col)[0]) < 3 and abs(py - mapa.a_mundo(f, col)[1]) < 3)
        r["despeje_min_m"] = despeje
        print("despeje mínimo a celdas ocupadas: %.2f m (el robot mide 0,62 x 0,28 m)" % despeje)
        resultados.append(r)
        print(f"\n=== CORRIDA {i+1} ===")
        print("estados:", r["estados"])
        print("final:", r["final"], "|", r["motivo"])
        print("métricas:", json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r["metricas"].items()}))
        print("duración virtual %.1f s | pared %.0f s | salto máx/ciclo %.3f m | dist. final a pelota %.2f m" %
              (r["duracion"], r["pared_s"], r["max_salto_m"], r["dist_pelota"]))
    if reps > 1:
        a, b = resultados[0]["traza"], resultados[1]["traza"]
        n = min(len(a), len(b)); dmax = max(math.hypot(a[k][1]-b[k][1], a[k][2]-b[k][2]) for k in range(n))
        print(f"\nREPETIBILIDAD: diferencia máxima de posición entre corridas = {dmax:.4f} m ({n} muestras)")
    json.dump([{k: v for k, v in r.items()} for r in resultados], open(os.environ.get("SALIDA", "/tmp/e2e_virtual.json"), "w"))
