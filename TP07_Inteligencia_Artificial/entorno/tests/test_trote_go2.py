"""Trote del Go2: cinemática verificada contra el MJCF, pies plantados, fase continua."""
import math
import os
import time
import unittest
from unittest import mock

from sim import trote_go2 as T
from sim.mundo import Mundo
from sim.safety import perfil

XML = os.path.join(os.path.dirname(__file__), "..", "sim", "unitree_mujoco",
                   "unitree_robots", "go2", "go2.xml")
ALTURA = 0.33


class CinematicaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mujoco
        cls.mj = mujoco
        cls.model = mujoco.MjModel.from_xml_path(XML)

    def test_rangos_articulares_coinciden_con_el_xml(self):
        mj = self.mj
        for pata in T.INDICE:
            for k, nombre in enumerate(("hip", "thigh", "calf")):
                jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, f"{pata}_{nombre}_joint")
                lo, hi = self.model.jnt_range[jid]
                self.assertAlmostEqual(lo, T.RANGOS[pata][k][0], places=2)
                self.assertAlmostEqual(hi, T.RANGOS[pata][k][1], places=2)

    def test_cinematica_directa_coincide_con_mujoco(self):
        mj, m = self.mj, self.model
        d = mj.MjData(m)
        for pata, q in (("FL", (0.1, 0.9, -1.6)), ("FR", (-0.2, 0.6, -1.2)),
                        ("RL", (0.0, 1.1, -1.9)), ("RR", (0.15, 0.7, -1.4))):
            i = T.INDICE[pata]
            d.qpos[:] = 0
            d.qpos[3] = 1
            d.qpos[7 + i:7 + i + 3] = q
            mj.mj_forward(m, d)
            pie = d.xpos[mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, pata + "_foot")]
            cadera = d.xpos[mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, pata + "_hip")]
            esperado = T.cinematica_directa(pata, *q)
            for a, b in zip(pie - cadera, esperado):
                self.assertAlmostEqual(a, b, places=6)

    def test_inversa_es_la_inversa_de_la_directa(self):
        for pata in T.INDICE:
            for q in ((0.05, 0.8, -1.55), (-0.2, 1.0, -1.9), (0.3, 0.5, -1.1)):
                objetivo = T.cinematica_directa(pata, *q)
                q0, q1, q2, recortado = T.cinematica_inversa(pata, *objetivo)
                self.assertFalse(recortado)
                for a, b in zip((q0, q1, q2), q):
                    self.assertAlmostEqual(a, b, places=6)


class MarchaTests(unittest.TestCase):
    def simular(self, vx, vy=0.0, vyaw=0.0, segundos=4.0, dt=0.005):
        """Devuelve, por pata, los pies en el MUNDO y su flag de apoyo (por fase)."""
        trote = T.Trote(ALTURA)
        x = y = yaw = 0.0
        fase = 0.0
        pies = {p: [] for p in T.CADERAS}
        t = 0.0
        for _ in range(int(segundos / dt)):
            t += dt
            fase = (fase + 2 * math.pi * dt * T.frecuencia_hz(vx, vy, vyaw)) % (2 * math.pi)
            ang, dz = trote.calcular(fase, vx, vy, vyaw, ALTURA, t=t)
            c, s = math.cos(yaw), math.sin(yaw)
            for pata, (hx, hy) in T.CADERAS.items():
                px, py, pz = T.cinematica_directa(pata, *ang[pata])
                bx, by = hx + px, hy + py
                pies.setdefault(pata, []).append((
                    x + bx * c - by * s, y + bx * s + by * c,
                    ALTURA + dz + pz - T.RADIO_PIE,
                    ((fase + T.FASE_PATA[pata]) % (2 * math.pi)) / (2 * math.pi) < T.DUTY))
            x += (vx * c - vy * s) * dt
            y += (vx * s + vy * c) * dt
            yaw += vyaw * dt
        return pies, trote

    def test_pie_en_apoyo_no_se_mueve_en_el_mundo(self):
        for vx, vy, w in ((0.5, 0, 0), (1.0, 0, 0), (2.0, 0, 0), (3.0, 0, 0),
                          (0, 0.5, 0), (1.0, 0, 0.4), (0, 0, 0.45)):
            pies, _ = self.simular(vx, vy, w)
            for pata, muestras in pies.items():
                tramo, peor = [], 0.0
                for (px, py, pz, apoyo) in muestras[400:]:      # régimen estable
                    if apoyo:
                        tramo.append((px, py))
                    elif tramo:
                        n = len(tramo)
                        a, b = tramo[int(0.1 * n)], tramo[int(0.9 * n)]
                        peor = max(peor, math.hypot(a[0] - b[0], a[1] - b[1]))
                        tramo = []
                self.assertLess(peor, 0.015, f"{pata} desliza {peor*100:.1f} cm a v=({vx},{vy},{w})")

    def test_pie_apoyado_toca_el_piso_y_el_balanceo_despeja(self):
        pies, _ = self.simular(1.0)
        for muestras in pies.values():
            apoyo = [z for (_, _, z, a) in muestras[400:] if a]
            balanceo = [z for (_, _, z, a) in muestras[400:] if not a]
            self.assertLess(max(abs(z) for z in apoyo), 0.004)      # sobre el piso
            self.assertGreater(max(balanceo), 0.04)                 # levanta el pie
            self.assertLess(max(balanceo), T.DESPEJE_M + 0.005)

    def test_siempre_hay_dos_pies_apoyados_en_el_trote(self):
        pies, _ = self.simular(1.5)
        for k in range(400, 800):
            self.assertEqual(sum(1 for m in pies.values() if m[k][3]), 2)

    def test_no_se_sale_del_alcance_de_las_patas_hasta_3_ms(self):
        for v in (0.3, 1.0, 2.0, 3.0):
            _, trote = self.simular(v, segundos=2.0)
            self.assertEqual(trote.recortes, 0, f"v={v}: poses recortadas")
        _, trote = self.simular(1.0, 0.0, 0.5, segundos=2.0)
        self.assertEqual(trote.recortes, 0)

    def test_en_reposo_las_patas_quedan_de_pie_bajo_la_cadera(self):
        trote = T.Trote(ALTURA)
        ang, dz = trote.calcular(1.0, 0.0, 0.0, 0.0, ALTURA, t=1.0)
        self.assertEqual(dz, 0.0)
        for pata, q in ang.items():
            x, y, z = T.cinematica_directa(pata, *q)
            self.assertAlmostEqual(x, 0.0, places=4)
            self.assertAlmostEqual(z, -(ALTURA - T.RADIO_PIE), places=4)
            self.assertAlmostEqual(abs(y), T.D_CADERA, places=4)

    def test_arranque_y_frenado_son_continuos(self):
        # Ningún ángulo salta entre cuadros consecutivos al acelerar y frenar.
        trote = T.Trote(ALTURA)
        fase, previo, salto = 0.0, None, 0.0
        t, dt = 0.0, 0.01
        for i in range(600):
            t += dt
            v = min(3.0, 1.5 * i * dt) if i < 300 else max(0.0, 3.0 - 1.5 * (i - 300) * dt)
            fase = (fase + 2 * math.pi * dt * T.frecuencia_hz(v, 0, 0)) % (2 * math.pi)
            ang, _ = trote.calcular(fase, v, 0.0, 0.0, ALTURA, t=t)
            plano = [q for p in sorted(ang) for q in ang[p]]
            if previo is not None:
                salto = max(salto, max(abs(a - b) for a, b in zip(plano, previo)))
            previo = plano
        self.assertLess(salto, 0.25)        # rad por cuadro de 10 ms (~25 rad/s)


class FaseTroteTests(unittest.TestCase):
    def test_la_fase_del_trote_es_continua_al_acelerar(self):
        reloj = [100.0]
        with mock.patch.object(time, "monotonic", lambda: reloj[0]):
            mundo = Mundo(perfil("tp07"))
            mundo.habilitar_carrera_simulada(3.0, 2.0)
            previo, peor = mundo.fase_trote, 0.0
            for i in range(600):
                reloj[0] += 0.01
                v = min(3.0, 1.5 * i * 0.01)
                mundo.mover_autonomo(v, 0.0, 0.0, duracion=0.5)
                mundo.avanzar()
                f = mundo.fase_trote
                d = (f - previo + math.pi) % (2 * math.pi) - math.pi
                peor = max(peor, abs(d) / 0.01)
                previo = f
        # Antes: hasta ~494 rad/s. Ahora nunca supera la frecuencia máxima de paso.
        self.assertLessEqual(peor, 2 * math.pi * T.F_MAX_HZ + 0.5)

    def test_leer_expone_velocidades_y_fase_del_trote(self):
        mundo = Mundo(perfil("tp07"))
        mundo.habilitar_carrera_simulada(3.0, 2.0)
        mundo.mover_autonomo(1.0, 0.0, 0.2, duracion=0.5)
        e = mundo.leer()
        self.assertEqual((e["vx"], e["vy"], e["vyaw"]), (1.0, 0.0, 0.2))
        self.assertIn("fase_trote", e)


if __name__ == "__main__":
    unittest.main()
