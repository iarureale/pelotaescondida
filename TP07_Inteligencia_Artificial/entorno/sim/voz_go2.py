"""Entrada de voz local para la misión Go2 en Windows (System.Speech)."""
from __future__ import annotations

import os
import re
import subprocess


_SCRIPT_POWERSHELL = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$culture = [System.Globalization.CultureInfo]::new('es-ES')
$engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine -ArgumentList $culture
try {
  $choices = New-Object System.Speech.Recognition.Choices
  @(
    'busca la pelota roja', 'busca pelota roja', 'buscar la pelota roja',
    'quiero que busques la pelota roja', 'encuentra la pelota roja',
    'encontra la pelota roja', 'inicia la busqueda de la pelota roja'
  ) | ForEach-Object { $choices.Add($_) }
  $builder = New-Object System.Speech.Recognition.GrammarBuilder
  $builder.Culture = $culture
  $builder.Append($choices)
  $grammar = New-Object System.Speech.Recognition.Grammar -ArgumentList $builder
  $engine.LoadGrammar($grammar)
  $engine.SetInputToDefaultAudioDevice()
  Write-Output 'ESCUCHANDO'
  $result = $engine.Recognize([TimeSpan]::FromSeconds($env:UADE_GO2_VOICE_TIMEOUT))
  if ($null -eq $result) { Write-Output 'SIN_COMANDO' }
  else { Write-Output ('COMANDO:' + $result.Text) }
} finally {
  $engine.Dispose()
}
"""


def normalizar_comando(texto: str) -> str | None:
    """Devuelve una intención de misión solo para pedidos explícitos de pelota roja."""
    palabras = re.findall(r"[a-z]+", texto.lower())
    verbos = {"busca", "buscar", "busques", "encuentra", "encontra", "inicia"}
    if "pelota" in palabras and any(p in palabras for p in ("roja", "rojo")) \
            and any(p in verbos for p in palabras):
        return "buscar_pelota_roja"
    return None


def escuchar_busqueda_pelota_roja(timeout_s: int = 15) -> tuple[str, str]:
    """Escucha una frase de una gramática cerrada y devuelve intención y texto.

    Usa el reconocedor español local de Windows. No sube audio a servicios
    externos y no acciona motores reales: el llamador solo activa el simulador.
    """
    if os.name != "nt":
        raise RuntimeError("El modo por voz necesita Windows y System.Speech.")
    timeout_s = max(3, min(int(timeout_s), 60))
    entorno = os.environ.copy()
    entorno["UADE_GO2_VOICE_TIMEOUT"] = str(timeout_s)
    try:
        resultado = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
             _SCRIPT_POWERSHELL], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout_s + 10,
            env=entorno, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("No se encontró powershell.exe para usar el micrófono.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Se agotó el tiempo de escucha por voz.") from exc
    if resultado.returncode:
        detalle = resultado.stderr.strip().splitlines()
        mensaje = detalle[-1] if detalle else "No se pudo inicializar el micrófono."
        raise RuntimeError(mensaje)
    reconocido = next((line[len("COMANDO:"):].strip()
                       for line in resultado.stdout.splitlines()
                       if line.startswith("COMANDO:")), "")
    intencion = normalizar_comando(reconocido)
    if intencion is None:
        raise RuntimeError("No se reconoció ‘busca la pelota roja’. Intentá de nuevo.")
    return intencion, reconocido
