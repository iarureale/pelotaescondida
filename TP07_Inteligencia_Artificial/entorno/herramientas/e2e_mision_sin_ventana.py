"""Misión del parque de punta a punta, sin ventana, con el SimuladorOficial real.

Uso (desde entorno/):  VEL=0.75 python -u herramientas/e2e_mision_sin_ventana.py 235
VEL = velocidad de carrera simulada (m/s). Con render por software (~4x más
lento que una GPU) conviene VEL=0.75 para conservar el avance por cuadro.
Es una medición de una sola corrida: la persona se mueve en tiempo real.
"""
import os, sys, time, math, json
os.environ.setdefault("MUJOCO_GL","osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM","osmesa")
RAIZ = os.environ.get("RAIZ_PROYECTO", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))); sys.path.insert(0, RAIZ + "/entorno")
import warnings; warnings.simplefilter("ignore")
from sim import safety, robots
from sim.mundo import Mundo
from sim.grilla import cargar
from sim.arrancar import SimuladorOficial

mapa = cargar(RAIZ+"/mi_desarrollo/mapas/patio_uade_go2.json")
robot = robots.obtener("go2")
mundo = Mundo(safety.perfil("tp07"))
x0,y0 = mapa.a_mundo(*mapa.inicio); mundo.situar(x0,y0,mapa.rumbo_inicial())
mundo.mision_parque = True; mundo.habilitar_carrera_simulada(float(os.environ.get("VEL","3.0")), 2.0)
sim = SimuladorOficial(mundo, robot, RAIZ+"/entorno/sim/unitree_mujoco", mapa=mapa, verboso=False)
class VistaNula:
    def publicar(self,*a,**k): pass
    def iniciar(self): pass
sim.vista_camara = VistaNula()

LIM = float(sys.argv[1]) if len(sys.argv)>1 else 240
t0 = time.monotonic(); estados=[]; z_max=0; pitch_max=0; fin=None; frames=0
STAND = None
while time.monotonic()-t0 < LIM:
    mundo.avanzar(); sim._escribir_pose()
    t = time.monotonic()
    sim._paso_seguimiento_visual()
    frames += 1
    est = sim.mision.estado
    if not estados or estados[-1][1] != est: estados.append((round(t-t0,1), est))
    q = sim.data.qpos
    if mundo.leer()["accion"] == "backflip_visual":
        z_max = max(z_max, q[2]); pitch_max = max(pitch_max, abs(2*math.asin(max(-1,min(1,q[5]))) ))
    if est in ("ERROR","FINALIZADA","BACKFLIP_SIMULADO","BACKFLIP_OMITIDO","BACKFLIP_NO_DISPONIBLE"):
        if fin is None: fin = time.monotonic()
        if time.monotonic()-fin > 3.0: break
    time.sleep(0.01)
a = mundo.leer(); q = sim.data.qpos
print("\n=== RESULTADO E2E ===")
print("duración real: %.1f s" % (time.monotonic()-t0), "| ciclos:", frames)
print("transiciones:", estados)
print("estado final:", sim.mision.estado, "|", sim.mision.motivo)
print("métricas:", json.dumps({k:(round(v,2) if isinstance(v,float) else v) for k,v in sim.mision.metricas.items()}))
print("acción final:", a["accion"], "| z base final: %.3f" % q[2], "| z de pie esperado: %.3f" % robot.altura)
print("quat final (w,x,y,z):", [round(float(v),3) for v in q[3:7]], "| yaw mundo: %.3f" % a["yaw"])
print("durante festejo -> z máx %.2f, |pitch| máx %.2f rad" % (z_max, pitch_max))
bx,by = mapa.a_mundo(*mapa.escenario["pelota_oculta"]); print("distancia final a la pelota: %.2f m" % math.hypot(a["x"]-bx,a["y"]-by))
