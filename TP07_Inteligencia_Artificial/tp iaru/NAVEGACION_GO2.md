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
asientos, farolas, banderas con textura, pérgola y edificios con ventanales y balcones. El
mapa mide 32 x 24 m y contiene 43 piezas de mobiliario/obstáculos decoradas,
más cuatro personas sentadas en bancos (solo ambientación). Estas construcciones se generan como geometría MuJoCo;
no reemplazan el XML oficial del Go2 ni necesitan texturas externas. El Go2
recorre sectores de búsqueda conocidos con A*, evitando los obstáculos que
figuran en el mapa. La celda de la pelota se usa para colocarla en la escena,
pero no se entrega al planificador. La cámara debe observarla para iniciar el
seguimiento. El escenario restringe la confirmación de candidatos a partir del
sector de búsqueda configurado; las etiquetas aisladas fuera de esa zona no
interrumpen la ruta A*. El patio UADE ya no tiene persona móvil. Los mapas
`parque_go2_*` conservan la persona animada: si aparece a menos de 1 m, el
robot se detiene; las detecciones más lejanas se incorporan como celdas
ocupadas temporales y pueden provocar una replanificación.

Los estados y sus motivos se imprimen en la terminal. Al terminar, también se
muestran distancia recorrida, replans, detecciones, obstáculos evitados,
colisiones y duración. La misión tiene un límite de cinco minutos y espera hasta
10 s si un obstáculo dinámico deja la ruta sin salida.

El mapa ampliado anterior sigue disponible como alternativa. El mapa nuevo usa
celdas de 1 m. Para una marcha rápida, el modo de navegación autónoma local
permite hasta 3.0 m/s, con giros de hasta 2.0 rad/s. A* conserva la ruta de
celdas para dibujarla, pero el controlador condensa los tramos rectos y solo
frena para las esquinas. La velocidad baja al aproximarse al objetivo.

## Seguimiento de ruta, pelota y reinicio

- **Seguidor (`navegador_astar.py`).** Sigue cada tramo con un punto de mira
  (0,8 m) sobre la línea de la grilla: no corta esquinas en diagonal. Avanza
  mientras corrige el rumbo y solo gira en el sitio con error > 0,9 rad.
  Aceleración lineal limitada a 1,5 m/s² y angular a 5 rad/s²; frena para las
  esquinas y para el destino. Si no progresa 0,2 m en 6 s, informa `atascado`
  y la misión replanifica. Se ejecuta en cada ciclo del visor (`control()`),
  separado de la captura de cámara.
- **Pelota.** Una pelota a más de 2,5 m se acepta tras 4 cuadros consecutivos y
  se alcanza con A* (estado `APROXIMANDO_PELOTA`); más cerca se usa la
  aproximación final reactiva. Si se tapa, el robot sigue hacia la última
  posición estimada y, tras 8 s sin verla, retoma los waypoints. Antes
  perseguía en línea recta y oscilaba entre buscar y perseguir.
- **Planificación de ruta (`planificacion_go2.planificar`).** Theta* de ángulo
  libre sobre un campo de despeje continuo (0,1 m, cacheado por mapa): parte de
  la posición REAL del robot, llega al punto exacto pedido y los vértices pueden
  tener cualquier ángulo (antes: A* de 4 vecinos + atajos, que dejaba escaleras y
  rodeos). Despeje mínimo duro = margen de seguridad (0,39 m); entre ese margen y
  +0,10 m cada metro cuesta hasta 1,6 m, así que prefiere zonas anchas salvo que
  el rodeo sea mayor. Si no hay ruta así, cae al A* de celdas: siempre devuelve
  ruta si existe una. Medido con `herramientas/benchmark_rutas.py` contra la ruta
  óptima continua: 161,7 m contra 162,7 m de referencia (el A* anterior: 170,1 m,
  +4,6 %); en 150 pares aleatorios, 7 % más corta en promedio y 18 ms por plan.
  Los replanes conservan la velocidad (no frenan a 0) y las esquinas suaves se
  toman sin reducir la velocidad.
- **Acercamiento a la pelota.** La ruta termina 1 m antes de la pelota estimada
  (no en el centro de su celda). A 2,2 m o menos, con la pelota a la vista, el
  seguidor visual la toma SIN detenerse: avanza mientras corrige el rumbo, frena
  de forma proporcional a la distancia (rampas de 1,8/3 m/s²) y para a 0,75 m.
  Medido de 3 m a la parada: 1,8-2,1 s (antes 4,0 s con cámara de 10 Hz y 5,3 s a
  4 Hz). Si la pelota se tapa, sigue el comportamiento de memoria y barrido de
  360°.
- **Voz.** `python -m sim --voz` escucha «busca la pelota roja» (hasta 15 s,
  antes de abrir la ventana) y luego inicia esta misma misión. Usa el
  reconocedor en español de Windows (es-ES, o el español que haya instalado).
  La orden se da una sola vez, al lanzar el programa; **R** reinicia sin volver a
  pedirla.
- **Replanificación.** Solo ante personas detectadas, atasco o cambio de objetivo
  (antes, cada 0,5 s).
- **Reinicio.** Tecla **R** en la ventana 3D: el Go2 vuelve al inicio y repite el
  recorrido. El mundo es determinista, así que la trayectoria es idéntica.
- **Percepción en hilo aparte.** Con ventana 3D, la captura de cámara (3 renders
  sincrónicos) corre en su propio hilo y trabaja sobre una copia de la pose; el
  visor y el seguimiento de ruta (50 Hz) no esperan al render. La misión usa la
  pose del instante del cuadro para proyectar la pelota. En una GPU lenta esto
  evita que la ventana baje a ~3-5 cuadros/s y el robot avance a tirones.
- **Trote (`trote_go2.py`).** La locomoción es cinemática (no hay dinámica):
  la pose de las patas se escribe en `qpos`. La marcha anterior eran senoides
  sin relación con la velocidad y daba, medido sobre el modelo oficial con
  `herramientas/diagnostico_marcha.py`: 0,87 pies apoyados (un trote tiene 2),
  arrastre de 3,6-12,9 cm por apoyo, pies 2,9 cm bajo el piso y 13,5 cm en el
  aire, cuerpo que solo subía, y saltos de fase de hasta 494 rad/s al acelerar
  (fase acumulada × cadencia variable). Ahora cada pie sigue una trayectoria:
  en apoyo queda fijo en el mundo (se mueve hacia atrás respecto del cuerpo a
  v + ω×r), en balanceo describe un arco de 5,5 cm; los ángulos salen de
  cinemática inversa analítica con la geometría y rangos del XML oficial
  (verificada contra MuJoCo, error 0,000 mm). La frecuencia (1,6-3,8 Hz) crece
  con la velocidad y la fase se integra de forma continua. Medido: 2,1 pies
  apoyados, arrastre 0,0-0,8 cm, pie hasta 3 mm bajo el piso, cuerpo 33,0-33,8 cm.
  Sigue siendo una animación cinemática; no usa el controlador oficial ni
  SportClient.
- **Camino que se borra.** Solo se dibuja la ruta que falta (`dibujo.py`):
  flechas planas (chevrones ">") orientadas con la ruta y ancladas a la
  polilínea que sigue el navegador (la misma, no una línea aparte), que
  desaparecen al pasar el robot, con fundido de 1,5 m frente a él. Las esquinas
  se redondean (radio 0,9 m) para parecerse al recorrido real. El aspecto es un
  `EstiloRuta` (forma, paso, tamaño, color, fundido, radio): `ESTILO_FLECHAS` por
  defecto, `ESTILO_TRAZOS` (rayitas) o uno propio asignado a `sim.estilo_ruta`.
  Sirve para cualquier polilínea. Al llegar o terminar, se borra. Reutiliza los mismos slots cada
  cuadro. En el patio UADE además se omiten ~720 baldosas del visor que
  quedaban bajo la plaza.
- **Letreros.** `sim/letreros_go2.py` arma textos sobre las fachadas con una
  fuente de trazos (cápsulas sin colisión) sobre una franja de vidrio oscuro,
  sin texturas ni librerías de imágenes. Se declaran en el mapa
  (`escenario.letreros`: `lado`, `texto`, `x_m`, `piso`, `altura_m`). El patio UADE
  lleva FACO, FACE, UBS, FASA y FAIN en la fachada este (la pared derecha) con la disposición
  escalonada de la foto. Letras disponibles: A B C D E F I L M N O P R S T U Y.
- **Cámara 3D.** Seguimiento suavizado a 6 m; la rueda del mouse acerca o aleja.

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
Las personas animadas de los mapas `parque_go2_*` siguen una trayectoria del JSON;
no se detectan con una red neuronal.

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

## Banderas

Cada objeto `bandera` del mapa JSON puede indicar `"textura": "<nombre>"`; se
carga `entorno/sim/assets/banderas/bandera_<nombre>.png` (proporción 14:9, tela
de 1,26 x 0,81 m sin colisión, a 3,3 m del piso). Si el archivo no existe, se
avisa y se dibuja tela lisa.

- `argentina`: tres franjas iguales (celeste, blanca, celeste) y Sol de Mayo de
  32 rayos. Vectores de `flag-icons` 7.5.0 (MIT; ver `LICENCIA_flag-icons.txt`),
  recompuestos en 14:9 en `bandera_argentina.svg`.
- `uade`: escudo oficial aportado por la autora (`escudo_uade_original.png`,
  300 x 300 px) centrado sobre paño blanco 14:9, sin deformarlo. El color del
  paño es una suposición; si la bandera institucional usa otro, reemplazar
  `bandera_uade.png` (idealmente con el escudo en SVG o mayor resolución).

## Iluminación y entorno

Para los mapas de parque, `crear_escena` aplica ajustes solo visuales
(`_AJUSTES_VISUALES`): sol inclinado con sombras, cielo claro, neblina suave y
suelo exterior verde sin reflejos. No cambian colisiones, coordenadas ni la
percepción (segmentación y profundidad no dependen de la luz).

## Mediciones reproducibles

`python herramientas/diagnostico_marcha.py` mide pies apoyados, arrastre por
apoyo y altura de pies/cuerpo sobre el modelo real de MuJoCo (reloj virtual).

`python -u herramientas/e2e_reloj_virtual.py 120 2` corre la misión sin ventana
con reloj virtual (resultado independiente de la CPU), repite con reinicio y
reporta repetibilidad, colisiones y despeje. `SIN_PELOTA=1` oculta la pelota a
la cámara para recorrer toda la ruta de búsqueda.

Desde `entorno`: `python herramientas/bench_percepcion.py` (tiempo de captura y
detección por cuadro) y `python -u herramientas/e2e_mision_sin_ventana.py 235`
(misión completa sin ventana). Ver el docstring de cada script.

## Pruebas

Desde la carpeta `entorno`:

    python -m unittest discover -s tests -v

Las pruebas unitarias cubren A*, huella, rutas bloqueadas, cámara, detección
segmentada, construcción del parque y estados de la misión. La apertura de la
ventana 3D depende del soporte OpenGL del equipo. El render offscreen de la
cámara se puede validar aunque el visor no abra.
