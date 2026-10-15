"""Trote cinemático del Unitree Go2: pies apoyados en el piso, sin deslizar.

POR QUÉ EXISTE. La locomoción del simulador es cinemática (no hay dinámica ni
controlador de fuerzas): cada ciclo se escribe la pose en ``qpos``. La marcha
anterior era una suma de senoides en las articulaciones con fase arbitraria.
Medida sobre el modelo real de MuJoCo (``herramientas/diagnostico_marcha.py``)
daba: 1,26 pies de 4 apoyados, deslizamiento de 1,5 a 4,3 m/s en los pies que
tocaban el suelo, pies hundidos 2,9 cm y elevados 13,5 cm, cuerpo que solo
subía, y saltos de fase de hasta 494 rad/s al acelerar. Nada ataba el paso a la
velocidad del cuerpo, así que el cuerpo "flotaba" o "patinaba".

QUÉ HACE. Para cada pata define la TRAYECTORIA DEL PIE y obtiene los ángulos
de las articulaciones por cinemática inversa analítica:

* Apoyo (50 % del ciclo, diagonales juntas: FL+RR y FR+RL): el pie queda fijo
  en el MUNDO; visto desde el cuerpo se mueve hacia atrás a la velocidad del
  punto de la cadera (v + ω×r). Ahí está la sincronización marcha-velocidad.
* Balanceo: arco con despeje, vuelve adelante.
* El cuerpo sube un poco a mitad de apoyo (péndulo invertido); la altura del
  pie en apoyo se calcula con esa altura, de modo que el pie toca z = 0.
* La frecuencia de paso crece con la velocidad y la fase se integra de forma
  continua (``frecuencia_hz``): no hay saltos de fase.

Geometría y límites tomados del MJCF oficial del Go2 (unitree_mujoco):
cadera en (±0,1934, ±0,0465), desplazamiento lateral del muslo 0,0955, muslo y
pantorrilla de 0,213 m, pie esférico de 0,022 m, rangos articulares del XML.
No usa SportClient ni el controlador oficial: es una animación cinemática, no
una marcha dinámica del robot físico.
"""
from __future__ import annotations

import math

# --- Geometría oficial del Go2 (go2.xml) ---------------------------------
L1 = 0.213                      # muslo
L2 = 0.213                      # pantorrilla
D_CADERA = 0.0955               # desplazamiento lateral del muslo (signo según lado)
# Esfera de contacto del pie: r = 0,022 m, centrada 2 mm hacia atrás del eje de
# la pantorrilla (go2.xml). Ese desplazamiento rota con la pierna y deja el pie
# 1-2 mm bajo el piso si se usa el radio completo; se compensa con el radio efectivo.
RADIO_PIE = 0.0207
CADERAS = {                     # posición de la cadera en el marco del cuerpo
    "FL": (0.1934, 0.0465), "FR": (0.1934, -0.0465),
    "RL": (-0.1934, 0.0465), "RR": (-0.1934, -0.0465),
}
LADO = {"FL": 1.0, "FR": -1.0, "RL": 1.0, "RR": -1.0}
# Índices dentro de qpos[7:] (hip, thigh, calf) por pata, orden del MJCF.
INDICE = {"FL": 0, "FR": 3, "RL": 6, "RR": 9}
# Rangos articulares del XML: (cadera, muslo, pantorrilla).
RANGOS = {
    "FL": ((-1.047, 1.047), (-1.571, 3.491), (-2.723, -0.838)),
    "FR": ((-1.047, 1.047), (-1.571, 3.491), (-2.723, -0.838)),
    "RL": ((-1.047, 1.047), (-0.524, 4.538), (-2.723, -0.838)),
    "RR": ((-1.047, 1.047), (-0.524, 4.538), (-2.723, -0.838)),
}

# --- Parámetros de la marcha ---------------------------------------------
DUTY = 0.5                      # fracción del ciclo con el pie apoyado (trote)
FASE_PATA = {"FL": 0.0, "RR": 0.0, "FR": math.pi, "RL": math.pi}   # diagonales
DESPEJE_M = 0.055               # altura máxima del pie en balanceo
REBOTE_M = 0.008                # el cuerpo sube a mitad de apoyo
ZANCADA_MAX_M = 0.20            # desplazamiento máx. del pie respecto de la posición neutra
V_REF = 0.06                    # m/s: por debajo, la marcha se desvanece a "de pie"
F_MIN_HZ, F_PENDIENTE, F_MAX_HZ = 1.6, 0.9, 3.8
LONGITUD_CADERA = 0.21          # |r| de la cadera, para el aporte del giro a la frecuencia


def frecuencia_hz(vx: float, vy: float, vyaw: float) -> float:
    """Frecuencia de paso: crece con la velocidad (más rápido, pasos más seguidos)."""
    v = math.hypot(vx, vy) + abs(vyaw) * LONGITUD_CADERA
    return min(F_MAX_HZ, F_MIN_HZ + F_PENDIENTE * v)


def _suave(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def cinematica_inversa(pata: str, px: float, py: float, pz: float):
    """Ángulos (cadera, muslo, pantorrilla) para que el pie esté en (px,py,pz).

    Coordenadas relativas al origen de la cadera, en el marco del cuerpo
    (x adelante, y a la izquierda, z arriba). Devuelve los ángulos y un flag que
    indica si alguno debió recortarse al rango articular o al alcance de la pata.
    """
    d = D_CADERA * LADO[pata]
    recortado = False
    r_yz = math.hypot(py, pz)
    if r_yz < abs(d) + 1e-6:
        r_yz, recortado = abs(d) + 1e-6, True
    h = math.sqrt(max(r_yz * r_yz - d * d, 1e-12))          # distancia vertical en el plano de la pata
    q0 = math.atan2(pz, py) - math.atan2(-h, d)
    D = math.hypot(px, h)
    maximo, minimo = L1 + L2 - 1e-4, abs(L1 - L2) + 1e-3
    if D > maximo or D < minimo:
        recortado = True
        D = max(minimo, min(maximo, D))
    c = (D * D - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    q2 = -math.acos(max(-1.0, min(1.0, c)))                   # rodilla hacia atrás
    theta_pie = math.atan2(-px, h)
    q1 = theta_pie - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
    rangos = RANGOS[pata]
    q = []
    for valor, (lo, hi) in zip((q0, q1, q2), rangos):
        if valor < lo or valor > hi:
            recortado = True
        q.append(max(lo, min(hi, valor)))
    return q[0], q[1], q[2], recortado


def cinematica_directa(pata: str, q0: float, q1: float, q2: float):
    """Posición del centro del pie respecto de la cadera (inversa de la anterior)."""
    d = D_CADERA * LADO[pata]
    x = -L1 * math.sin(q1) - L2 * math.sin(q1 + q2)
    h = L1 * math.cos(q1) + L2 * math.cos(q1 + q2)
    y = d * math.cos(q0) + h * math.sin(q0)
    z = d * math.sin(q0) - h * math.cos(q0)
    return x, y, z


class Trote:
    """Genera los ángulos de las 12 articulaciones a partir de fase y velocidad."""

    TAU_VELOCIDAD_S = 0.06      # suaviza vx,vy,vyaw: sin tirones al arrancar/frenar

    def __init__(self, altura_base: float):
        self.altura_base = altura_base
        self._v = (0.0, 0.0, 0.0)
        self._t_previo = None
        self.recortes = 0       # cuántas poses debieron recortarse (diagnóstico)

    def reiniciar(self):
        self._v = (0.0, 0.0, 0.0)
        self._t_previo = None

    def _suavizar(self, vx, vy, vyaw, t):
        if self._t_previo is None:
            self._v = (vx, vy, vyaw)
        else:
            k = 1.0 - math.exp(-max(0.0, t - self._t_previo) / self.TAU_VELOCIDAD_S)
            self._v = tuple(a + (b - a) * k for a, b in zip(self._v, (vx, vy, vyaw)))
        self._t_previo = t
        return self._v

    def activo(self) -> bool:
        return math.hypot(self._v[0], self._v[1]) + abs(self._v[2]) * LONGITUD_CADERA > 0.004

    def calcular(self, fase: float, vx: float, vy: float, vyaw: float,
                 altura_base: float | None = None, t: float | None = None):
        """Devuelve ({pata: (q0,q1,q2)}, dz_cuerpo).

        ``altura_base`` es la altura del cuerpo de pie; ``dz`` se suma a ella.
        ``vx, vy, vyaw`` en el marco del cuerpo (m/s, rad/s).
        """
        z0 = self.altura_base if altura_base is None else altura_base
        if t is not None:
            vx, vy, vyaw = self._suavizar(vx, vy, vyaw, t)
        f = frecuencia_hz(vx, vy, vyaw)
        t_apoyo = DUTY / f
        # Rebote: máximo a mitad de apoyo de cada diagonal (fase = π/2 y 3π/2).
        g_cuerpo = _suave((math.hypot(vx, vy) + abs(vyaw) * LONGITUD_CADERA) / V_REF)
        dz = REBOTE_M * g_cuerpo * math.sin(fase) ** 2
        z_cadera = z0 + dz
        angulos = {}
        for pata, (hx, hy) in CADERAS.items():
            # Velocidad del punto de la cadera: v + ω × r.
            ux, uy = vx - vyaw * hy, vy + vyaw * hx
            mod = math.hypot(ux, uy)
            g = _suave(mod / V_REF)
            if mod * t_apoyo / 2.0 > ZANCADA_MAX_M:                # límite de alcance
                escala = ZANCADA_MAX_M / (mod * t_apoyo / 2.0)
                ux, uy = ux * escala, uy * escala
            psi = ((fase + FASE_PATA[pata]) % (2 * math.pi)) / (2 * math.pi)
            if psi < DUTY:                                          # APOYO: pie fijo en el mundo
                u = psi / DUTY
                avance = 0.5 - u
                alzada = 0.0
            else:                                                   # BALANCEO
                w = (psi - DUTY) / (1.0 - DUTY)
                avance = -0.5 * math.cos(math.pi * w)
                alzada = DESPEJE_M * g * math.sin(math.pi * w)
            px = avance * ux * t_apoyo
            py = D_CADERA * LADO[pata] + avance * uy * t_apoyo
            pz = -(z_cadera - RADIO_PIE) + alzada
            q0, q1, q2, recortado = cinematica_inversa(pata, px, py, pz)
            self.recortes += int(recortado)
            angulos[pata] = (q0, q1, q2)
        return angulos, dz
