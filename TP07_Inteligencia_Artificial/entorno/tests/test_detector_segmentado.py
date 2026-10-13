"""La detección vectorizada debe dar lo mismo que la definición por máscaras."""
import math
import unittest

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None

from sim.camara_go2 import _categoria_geometria, detectar_objetos_segmentados

XML = """<mujoco><worldbody>
  <geom name="suelo" type="plane" size="5 5 .1"/>
  <geom name="nav_target_ball" type="sphere" size=".1" pos="0 0 1"/>
  <geom name="nav_person_0_torso" type="capsule" size=".1 .3" pos="1 0 1"/>
  <geom name="nav_obstacle_arbol_1_1_copa" type="sphere" size=".3" pos="2 0 1"/>
  <geom name="uade_ventana_0_1" type="box" size=".1 .1 .1" pos="3 0 1"/>
</worldbody></mujoco>"""


def referencia(seg, depth, model, fovy=60.0, minimo=8):
    """Implementación original: una máscara de imagen completa por geom."""
    alto, ancho = seg.shape[:2]
    focal = alto / (2.0 * math.tan(math.radians(fovy) / 2.0))
    salida = []
    for gid in range(model.ngeom):
        nombre = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        cat = _categoria_geometria(nombre)
        if cat is None:
            continue
        ys, xs = np.nonzero((seg[:, :, 0] == gid) & (seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))
        if len(xs) < minimo:
            continue
        d = depth[ys, xs]
        d = d[np.isfinite(d) & (d > 0)]
        if not len(d):
            continue
        cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
        salida.append((nombre, cat, float(cx), float(cy),
                       math.atan2(cx - (ancho - 1) / 2.0, focal),
                       float(np.median(d)), len(xs),
                       int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())))
    return salida


@unittest.skipIf(mujoco is None, "requiere mujoco")
class DetectorSegmentadoTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.ids = {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i): i
                    for i in range(self.model.ngeom)}

    def escena_sintetica(self, semilla):
        rng = np.random.default_rng(semilla)
        alto, ancho = 48, 64
        seg = np.full((alto, ancho, 2), -1, dtype=np.int32)
        depth = rng.uniform(0.5, 6.0, (alto, ancho)).astype(np.float32)
        for nombre in ("nav_target_ball", "nav_person_0_torso",
                       "nav_obstacle_arbol_1_1_copa", "uade_ventana_0_1", "suelo"):
            y, x = rng.integers(0, alto - 12), rng.integers(0, ancho - 12)
            h, w = rng.integers(1, 12), rng.integers(1, 12)
            seg[y:y + h, x:x + w, 0] = self.ids[nombre]
            seg[y:y + h, x:x + w, 1] = int(mujoco.mjtObj.mjOBJ_GEOM)
        depth[rng.random(depth.shape) < 0.05] = np.inf   # profundidad inválida
        return seg, depth

    def test_coincide_con_la_definicion_por_mascaras(self):
        for semilla in range(25):
            seg, depth = self.escena_sintetica(semilla)
            esperado = referencia(seg, depth, self.model)
            obtenido = [(o.nombre, o.categoria, o.centro_x, o.centro_y, o.bearing_rad,
                         o.distancia_m, o.pixeles, o.x_min, o.x_max, o.y_min, o.y_max)
                        for o in detectar_objetos_segmentados(seg, depth, self.model, mujoco)]
            self.assertEqual(obtenido, esperado, f"semilla {semilla}")

    def test_imagen_sin_objetos_no_detecta_nada(self):
        seg = np.full((10, 10, 2), -1, dtype=np.int32)
        self.assertEqual(
            detectar_objetos_segmentados(seg, np.ones((10, 10), np.float32), self.model, mujoco), [])

    def test_ignora_geoms_que_no_son_objetos_de_mision(self):
        seg, depth = self.escena_sintetica(3)
        nombres = {o.nombre for o in detectar_objetos_segmentados(seg, depth, self.model, mujoco)}
        self.assertTrue(nombres <= {"nav_target_ball", "nav_person_0_torso",
                                    "nav_obstacle_arbol_1_1_copa"})


if __name__ == "__main__":
    unittest.main()
