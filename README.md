# Go2: navegación autónoma y búsqueda de pelota

Proyecto basado en el laboratorio TP07 de UADE. Incluye el simulador MuJoCo completo, el robot Unitree Go2, mapas de ocupación, planificación A*, obstáculos, cámara simulada y una misión de búsqueda en un patio inspirado en UADE.

## Requisitos

- Python 3.10 o superior.
- MuJoCo (`python -m pip install mujoco`).
- En Windows, Visual C++ Redistributable.

La guía completa de instalación está en [`TP07_Inteligencia_Artificial/INSTALACION.md`](TP07_Inteligencia_Artificial/INSTALACION.md). La documentación específica de navegación está en [`TP07_Inteligencia_Artificial/tp iaru/NAVEGACION_GO2.md`](TP07_Inteligencia_Artificial/tp%20iaru/NAVEGACION_GO2.md).

## Ejecutar la misión del parque

Desde `TP07_Inteligencia_Artificial/entorno`:

```powershell
python -m sim --robot go2 --materia tp07 --grilla ..\mi_desarrollo\mapas\patio_uade_go2.json --mision-parque
```

Para iniciar con reconocimiento de voz local de Windows:

```powershell
python -m sim --voz
```

Luego decir: «busca la pelota roja».

## Alcance

La cámara usa renderizado y segmentación de MuJoCo; A* usa la pose real del simulador. La locomoción rápida (hasta 3 m/s) y el backflip son animaciones cinemáticas locales. No representan dinámica física, SLAM ni una acrobacia validada para el robot real.
