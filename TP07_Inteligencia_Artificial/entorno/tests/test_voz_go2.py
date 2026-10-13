import unittest

from sim.voz_go2 import normalizar_comando


class VozGo2Tests(unittest.TestCase):
    def test_acepta_pedido_explicito_de_pelota_roja(self):
        self.assertEqual(normalizar_comando("Busca la pelota roja"),
                         "buscar_pelota_roja")
        self.assertEqual(normalizar_comando("Encontra la pelota roja"),
                         "buscar_pelota_roja")

    def test_no_convierte_frases_ambiguas_en_orden(self):
        self.assertIsNone(normalizar_comando("busca a la persona"))
        self.assertIsNone(normalizar_comando("pelota roja"))


if __name__ == "__main__":
    unittest.main()
