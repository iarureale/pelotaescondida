# Navegación autónoma del Go2 en MuJoCo

El prototipo agrega un circuito navegable a la escena limpia UADE, con mapa de
ocupación, A*, obstáculos con colisión, cámara simulada y una misión de búsqueda
de pelota. No modifica el XML oficial del Go2: genera MJCF temporal para cada
ejecución y lo elimina después de compilar el modelo.

## Misión del parque

Desde PowerShell, dentro de `entorno`:

    python -m sim --robot go2 --materia tp07 --grilla ..\mi_desarrollo\mapas\patio_uade_go2.json --mision-parque

Para iniciar la misma misión con reconocimiento local de voz de Windows:

    python -m sim --voz

Decir «busca la pelota roja». El modo usa System.Speech en español de España y
no envía audio a un servicio externo. Se puede combinar `--voz` con `--grilla`
para elegir otro mapa.

La misión dibuja un patio universitario inspirado en la explanada UADE de las
referencias: explanada embaldosada, jardines, maceteros, árboles, bancos,
asientos, farolas, banderas, pérgola y edificios con ventanales y balcones. El
mapa mide 32 x 24 m y contiene 43 piezas de mobiliario/obstáculos decoradas,
más una persona móvil. Estas construcciones se generan como geometría MuJoCo;
no reemplazan el XML oficial del Go2 ni necesitan texturas externas. El Go2
recorre sectores de búsqueda conocidos con A*, evitando los obstáculos que
figuran en el mapa. La celda de la pelota se usa para colocarla en la escena,
pero no se entrega al planificador. La cámara debe observarla para iniciar el
seguimiento. El escenario restringe la confirmación de candidatos a partir del
sector de búsqueda configurado; las etiquetas aisladas fuera de esa zona no
interrumpen la ruta A*. Hay además una persona animada: si aparece a menos de 1 m, el robot
se detiene; las detecciones más lejanas se incorporan como celdas ocupadas
temporales y pueden provocar una replanificación.

Los estados y sus motivos se imprimen en la terminal. Al terminar, también se
muestran distancia recorrida, replans, detecciones, obstáculos evitados,
colisiones y duración. La misión tiene un límite de cinco minutos y espera hasta
10 s si un obstáculo dinámico deja la ruta sin salida.

El mapa ampliado anterior sigue disponible como alternativa. El mapa nuevo usa
celdas de 1 m. Para una marcha rápida, el modo de navegación autónoma local
permite hasta 3.0 m/s, con giros de hasta 2.0 rad/s. A* conserva la ruta de
celdas para dibujarla, pero el controlador condensa los tramos rectos y solo
frena para las esquinas. La velocidad baja al aproximarse al objetivo.

## Modos por separado

Planificar y seguir la ruta A* de un mapa:

    python -m sim --robot go2 --materia tp07 --grilla ..\mi_desarrollo\mapas\circuito_go2_astar.json --autonavegar-astar

Seguir una pelota roja colocada en el destino, sin la misión de búsqueda:

    python -m sim --robot go2 --materia tp07 --grilla ..\mi_desarrollo\mapas\circuito_go2_pelota.json --seguir-pelota

El seguidor simple no replanifica A* ante obstáculos inesperados. Para búsqueda
con obstáculos dinámicos, usar `--mision-parque`.

## Alcance de la percepción y de la navegación

La cámara es un render de MuJoCo con RGB, depth y etiquetas de segmentación por
geometría. La pelota y los objetos del escenario se reconocen por la etiqueta de
su geometría, no por un modelo general de visión artificial. Distancia y
bearing se estiman desde píxeles visibles y depth. La posición del robot para
A* proviene de `Mundo` (ground truth del simulador), no de odometría ni SLAM.
La persona se mueve por una trayectoria definida en el JSON; no es una persona
detectada mediante una red neuronal.

La locomoción del paquete es cinemática: integra comandos de velocidad, escribe
la pose en MuJoCo y anima las patas. No ejecuta `mj_step` ni un controlador de
marcha con dinámica física. Los 3.0 m/s son una velocidad visual de carrera
para la simulación local, no una afirmación sobre la velocidad máxima física
del Go2. Solo se habilitan con `--autonavegar-astar`, `--seguir-pelota`,
`--mision-parque` o `--voz`, usando el transporte local. La API manual conserva
los límites pedagógicos/físicos y DDS no habilita el modo rápido. Los contactos sirven para detener al robot cuando
entra en un obstáculo; no simulan empujes o caídas. Esto permite validar A*,
estados de misión, cámara y coordinación del recorrido, pero no prueba
sim-to-real ni locomoción física autónoma.

El Go2 usa un trote diagonal más marcado, con cadencia ajustada a la velocidad,
oscilación vertical leve del cuerpo e inclinación sincronizada al desplazarse.

Al encontrar la pelota, la misión muestra un backflip visual cinemático de 360°
como festejo, si la cámara no detecta personas u objetos próximos. Es una
animación de pose: no es una acrobacia dinámica ni envía la orden SportClient
`Backflip` al Go2 físico. La locomoción sigue siendo cinemática, por lo que no
valida equilibrio ni sim-to-real.

## Pruebas

Desde la carpeta `entorno`:

    python -m unittest discover -s tests -v

Las pruebas unitarias cubren A*, huella, rutas bloqueadas, cámara, detección
segmentada, construcción del parque y estados de la misión. La apertura de la
ventana 3D depende del soporte OpenGL del equipo. El render offscreen de la
cámara se puede validar aunque el visor no abra.
