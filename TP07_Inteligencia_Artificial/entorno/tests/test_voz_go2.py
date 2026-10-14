import unittest

from sim.voz_go2 import _SCRIPT_POWERSHELL, normalizar_comando


class VozGo2Tests(unittest.TestCase):
    def test_acepta_pedido_explicito_de_pelota_roja(self):
        self.assertEqual(normalizar_comando("Busca la pelota roja"),
                         "buscar_pelota_roja")
        self.assertEqual(normalizar_comando("Encontra la pelota roja"),
                         "buscar_pelota_roja")

    def test_no_convierte_frases_ambiguas_en_orden(self):
        self.assertIsNone(normalizar_comando("busca a la persona"))
        self.assertIsNone(normalizar_comando("pelota roja"))

    def test_el_script_busca_cualquier_reconocedor_en_espanol(self):
        # No se puede ejecutar PowerShell/System.Speech fuera de Windows: se
        # verifica que el script no dependa de un único es-ES.
        self.assertIn("InstalledRecognizers", _SCRIPT_POWERSHELL)
        self.assertIn("TwoLetterISOLanguageName -eq 'es'", _SCRIPT_POWERSHELL)
        self.assertIn("busca la pelota roja", _SCRIPT_POWERSHELL)


if __name__ == "__main__":
    unittest.main()
