# Monitor Inteligente de Tráfico — Niveles 1, 2 y 3

> **Nivel 3 (Odometría visual, cámara en movimiento):** ver la [sección 9](#9-nivel-3--odometría-visual-cámara-en-movimiento). Usa el programa `odometria_visual.py`.
>
> **Nivel 2 (Analítica de tránsito):** ver la [sección 8](#8-nivel-2--analítica-de-tránsito). Funciona con el **mismo comando** del radar: al cerrar el programa se generan solos el historial CSV, el **mapa de calor por carril** y la **gráfica de vehículos por hora**.

**Nivel 1:** analiza un **video subido** (o la webcam), detecta y sigue los vehículos con **YOLO + ByteTrack**, convierte su movimiento a metros con una **calibración hecha con Google Maps** y muestra la **velocidad en km/h** de cada vehículo. Si un vehículo supera el límite, **su caja se vuelve roja**, parpadea y se guarda una **captura de la infracción**.

```
monitor_trafico/
├── calibrar.py              ← PASO 1: marcar la distancia real (Google Maps) sobre el video
├── radar_velocidad.py       ← PASO 2: el radar (Nivel 1) + registro y analítica al cerrar (Nivel 2)
├── analitica.py             ← Nivel 2: regenerar gráficos o JUNTAR varios clips
├── odometria_visual.py      ← Nivel 3: cámara en movimiento, antes vs después
├── requirements.txt
├── radar/
│   ├── calibracion.py       píxeles → metros (PPM u homografía), zona, ejes de la vía
│   ├── velocidad.py         velocidad por vehículo (regresión + suavizado) y control del límite
│   ├── conteo.py            [N2] línea virtual de conteo (cada vehículo se cuenta una vez)
│   ├── registro.py          [N2] historial.csv, conteo.csv, sesion.json, fondo de la calle
│   ├── tiempo.py            [N2] hora real de grabación (nombre del archivo / MP4 / --hora-inicio)
│   ├── analitica.py         [N2] carriles, mapa de calor, vehículos por hora, resumen
│   ├── odometria.py         [N3] FAST/ORB + BRIEF, emparejamiento, RANSAC, pose de la cámara
│   ├── graficos_n3.py       [N3] mapa de la trayectoria y comparación de velocidades
│   ├── dibujo.py            cajas, colores, panel de información
│   └── video.py             abrir video / explorador de archivos / FPS
└── pruebas/
    ├── test_radar.py, test_calibrar_interactivo.py, test_nivel2.py, test_nivel3.py   (27 pruebas)
    ├── generar_video_sintetico.py   / video_sintetico.mp4            (Nivel 1: 64.8 y 32.4 km/h)
    ├── generar_video_nivel2.py      / VID_20261120_075955.mp4        (Nivel 2: carriles y horas conocidos)
    └── generar_video_nivel3.py      / camara_movil.mp4               (Nivel 3: cámara a 18 km/h, buses quietos y en marcha)
```

---

## 0. Instalación

Python 3.10 o superior.

```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

La primera ejecución descarga automáticamente los pesos `yolo11n.pt` (unos 5 MB).
Con tarjeta NVIDIA, instala PyTorch con CUDA desde pytorch.org para ir de 5 a 10 veces más rápido.

> Si al abrir la ventana aparece el error `The function is not implemented`, tienes instalado `opencv-python-headless`. Ejecuta `pip uninstall opencv-python-headless opencv-python` y luego `pip install opencv-python`.

**Prueba rápida, sin video propio:**
```bash
python calibrar.py --video pruebas/video_sintetico.mp4 --modo ppm --puntos "100,650;1100,650" --metros 50 --salida pruebas/calib_sintetico.json
python radar_velocidad.py --video pruebas/video_sintetico.mp4 --calibracion pruebas/calib_sintetico.json --limite 50
```
Debe dar unos 64.8 km/h (en rojo, EXCESO) y unos 32.4 km/h (en verde).

---

## 1. Preparar el video

- La cámara debe estar **fija**: trípode o apoyada, sin zoom ni movimiento. Debe estar elevada (puente peatonal, 2.º a 4.º piso) y grabar de día.
- El video debe ser horizontal, de 720p o 1080p, a 30 fps, y durar entre 1 y 3 minutos.
- Necesitas saber **dónde se grabó**, para poder medir el lugar en Google Maps.
- **Videos de celular:** muchos graban con FPS *variable*, y eso falsea el tiempo. Conviértelos a FPS constante:
  ```bash
  ffmpeg -i video_celular.mp4 -vf fps=30 -c:v libx264 -crf 20 -an video_30fps.mp4
  ```
- Si el video es 4K, puedes bajarlo a 1080p para procesar más rápido (agrega `-vf scale=1920:-2`).

---

## 2. Medir la distancia real en Google Maps

1. Abre Google Maps en la PC y activa la **vista Satélite**. Ubica el lugar exacto de la grabación.
2. Haz **clic derecho** sobre el primer punto, elige **"Medir distancia"** y luego haz clic en el segundo punto. La distancia aparece abajo, en metros.
3. Elige puntos que se vean **en el suelo**, tanto en el video como en el satélite: las esquinas de un paso de cebra, las líneas de carril, el borde de la vereda o la **base** de un poste (nunca la punta).
4. Usa distancias **largas, de 15 m o más**. Un error de 0.5 m en 25 m es un 2 %; en 5 m es un 10 %.
5. Mide **una segunda distancia** distinta. La usarás para verificar la calibración.

> La imagen satelital puede estar desactualizada, por ejemplo si se repintaron las líneas. Street View sirve para confirmar postes y marcas.

---

## 3. Calibrar: `calibrar.py`

### ¿Qué modo elijo?

| Cómo ve la cámara la calle | Modo | Qué marcas en el suelo |
|---|---|---|
| **De costado**: los autos cruzan la imagen de izquierda a derecha y casi no cambian de tamaño | `ppm` | 2 puntos en el sentido del tráfico (por ejemplo, las bases de 2 postes) |
| **En diagonal o a lo largo de la calle**: los autos se achican al alejarse (la típica cámara de seguridad) | `homografia` | 4 esquinas de un rectángulo del suelo: ancho × largo conocidos |

**¿Por qué existen dos modos?** El factor *píxeles por metro* (PPM) es un único número. Con perspectiva, 1 m cerca de la cámara ocupa muchos más píxeles que 1 m lejos. En `pruebas/test_radar.py`, un auto a 60 km/h vista con perspectiva da **112 km/h cerca y 24 km/h lejos** con PPM, y **59 km/h** con homografía. La homografía calcula una "vista de pájaro" en metros y elimina ese error.

### Uso

```bash
python calibrar.py --video mi_video.mp4 --modo ppm
python calibrar.py --video mi_video.mp4 --modo homografia
python calibrar.py                      # abre el explorador para elegir el video
```

Se abre una ventana con una **lupa**, que se coloca en la esquina contraria al cursor, para marcar con precisión. Los videos pequeños se muestran ampliados.

> **Vista lateral con varios carriles (los carriles de atrás se ven más pequeños):** también usa homografía. Pon P1 y P4 al inicio de dos guiones de la línea discontinua de atrás, y P2 y P3 al inicio de dos guiones de la línea de adelante, abarcando el **mismo número de ciclos** (guion + espacio) en ambas. ANCHO = distancia entre las dos líneas. LARGO = esos N ciclos medidos en Google Maps. En la fase 2, dibuja la zona sobre los 4 carriles.

| Fase | Qué hacer |
|---|---|
| **1. Puntos** | **PPM:** 2 clics. **Homografía:** 4 clics en este orden: P1 cercano izquierdo → P2 cercano derecho (eso es el **ancho**) → P3 lejano del lado de P2 (eso es el **largo**) → P4 lejano del lado de P1. Pulsa ENTER y escribe las medidas en la **ventanita que aparece** (si no aparece, escríbelas en la terminal). |
| **2. Zona** (opcional) | Dibuja con clics el polígono donde se medirá la velocidad y pulsa ENTER. Si pulsas ENTER sin marcar nada, la zona por defecto es: en PPM, la imagen sin un 5 % de bordes; en homografía, el rectángulo de calibración. |
| **3. Verificación** | Haz 2 clics sobre **otra** distancia que mediste en Maps. Si el error es menor al 5 %, la calibración está bien. Pulsa ENTER para terminar. |

Teclas: **A / D** retroceden o avanzan 1 s (para elegir un cuadro sin autos tapando las marcas), **clic derecho o Z** deshace, **R** reinicia la fase y **Q** sale.

Se generan `calibracion.json`, `calibracion_preview.jpg` y, en homografía, `calibracion_vista_pajaro.jpg`. En esa última imagen, **las líneas de carril deben verse rectas y paralelas**; si se ven torcidas, el orden de los puntos o las medidas están mal.

Para calibrar sin ventana (repetible), pasa los puntos y las medidas por argumentos:
```bash
python calibrar.py --video v.mp4 --modo homografia --puntos "420,700;980,700;800,380;560,380" --ancho 7 --largo 30
```

---

## 4. Ejecutar el radar: `radar_velocidad.py`

```bash
python radar_velocidad.py --video mi_video.mp4 --calibracion calibracion.json --limite 50
```

| Opción | Por defecto | Para qué |
|---|---|---|
| `--limite` | 50 | Límite de velocidad en km/h |
| `--modelo` | `yolo11n.pt` | `yolo11s.pt` o `yolo11m.pt` son más precisos (mejor con GPU); `yolo26n.pt` también funciona |
| `--imgsz` | 640 | Usa 960 o 1280 si los autos se ven pequeños o lejanos |
| `--conf` | 0.30 | Confianza mínima de detección |
| `--salto` | 1 | Usa 2 para procesar 1 de cada 2 cuadros (el doble de rápido; el tiempo se sigue calculando bien) |
| `--ventana` | 0.6 | Segundos de historia para calcular la velocidad |
| `--confirmar` | 3 | Cuántas medidas seguidas sobre el límite hacen falta para marcar EXCESO (evita falsas alarmas) |
| `--fps` | (del archivo) | Fuerza los FPS si el archivo reporta un valor incorrecto |
| `--no-mostrar` | — | Procesa sin ventana (más rápido) |
| `--video 0` | — | Usa la webcam en vivo |

Teclas: **ESPACIO** pausa y **Q** sale.

**Colores de las cajas:** gris = aún midiendo · **verde** = normal · **ámbar** = 90 % o más del límite, o sobre el límite sin confirmar · **rojo parpadeante** = EXCESO confirmado.

### Resultados (`salidas/<nombre_video>/`)
- `<video>_radar.mp4`: el video con las anotaciones.
- `infracciones/exceso_id12_67kmh_t0012.30s.jpg`: una captura por cada infractor.
- `reporte_nivel1.txt`: una tabla por vehículo con la velocidad típica (mediana), la máxima, el tiempo y el estado.

---

## 5. Cómo funciona (para el informe)

1. **Detección y seguimiento.** YOLO detecta auto, moto, bus y camión (clases COCO 2, 3, 5 y 7). ByteTrack asigna un **ID** estable a cada vehículo entre cuadros.
2. **Punto de medición.** Se usa el **centro inferior de la caja**, `((x1+x2)/2, y2)`, porque es donde la llanta toca el suelo, que es el plano calibrado. El centro de la caja está a una altura desconocida y falsea la medida.
3. **Píxeles → metros.**
   - PPM: `PPM = distancia_px / distancia_m` y `X = x_px / PPM`.
   - Homografía: `[X, Y, 1]ᵀ ~ H · [x, y, 1]ᵀ`, con `H` (3×3) obtenida de 4 correspondencias usando `cv2.getPerspectiveTransform`.
4. **Tiempo.** Se calcula como `t = n_cuadro / FPS_del_video`, **no con el reloj de la PC**: si el procesamiento va más lento que el video, la velocidad sigue siendo correcta.
5. **Velocidad.** Se toman las posiciones en metros de los últimos 0.6 s y se ajusta una **recta por mínimos cuadrados** a X(t) y a Y(t). La pendiente es la velocidad: `v = √(vx² + vy²)`, y en km/h es `v · 3.6`. Esto es mucho más estable que usar solo dos puntos. Después se suaviza con una **media móvil exponencial**.
6. **Filtros de calidad.**
   - No se mide en cajas pegadas al borde (vehículos entrando o saliendo, con la caja recortada) ni fuera de la zona de medición.
   - Si un vehículo da un salto imposible entre dos cuadros (normalmente un cambio de ID), su historia se reinicia.
   - Se descartan las velocidades mayores a 250 km/h.
7. **Alerta.** Un vehículo se marca como EXCESO cuando supera el límite en **3 medidas seguidas**.

---

## 6. Validación

```bash
python -m pytest pruebas -q
```
Hay 13 pruebas del Nivel 1 (más 9 del Nivel 2): velocidad con PPM, homografía con perspectiva, zona de medición, saltos de ID, cambio de resolución, guardar y cargar, control del límite y el flujo interactivo de calibración.

Resultado de punta a punta con `pruebas/video_sintetico.mp4` (YOLO + ByteTrack reales):

| Vehículo | Real | Medido (mediana) | Error |
|---|---|---|---|
| A (carril inferior) | 64.8 km/h | 64.6 km/h | 0.3 % |
| B (carril superior) | 32.4 km/h | 32.3 km/h | 0.3 % |

**Con un video real:** la mejor validación es que un conocido pase en auto a una velocidad constante, por ejemplo 30 km/h según el velocímetro, y comparar. En videos reales, un error del 5 al 10 % es un buen resultado. Sus causas principales son la precisión de Google Maps, los clics de calibración y el temblor de las cajas de YOLO.

---

## 7. Problemas frecuentes

| Síntoma | Causa probable / solución |
|---|---|
| Todas las velocidades salen el doble o la mitad | FPS mal leídos (video de celular): convierte con ffmpeg o usa `--fps 30` |
| Los autos cercanos salen rápidos y los lejanos lentos | Hay perspectiva: usa `--modo homografia` |
| Velocidades absurdas en algunos autos | Revisa la verificación de la calibración. Sube `--conf`, o usa `--imgsz 960` o `--modelo yolo11s.pt` |
| No se detectan autos lejanos | Usa `--imgsz 1280` o un modelo más grande |
| Va muy lento | Usa `--no-mostrar`, `--salto 2`, GPU, o baja el video a 720p |
| Un auto aparece con "-- km/h" | Está fuera de la zona de medición o pegado al borde (es lo esperado) |

---

---

## 8. Nivel 2 — Analítica de tránsito

Usa **el mismo comando** del radar. Mientras procesa, guarda el historial; **al terminar el video, o al pulsar Q, Esc o Ctrl+C**, genera todo automáticamente:

```bash
python radar_velocidad.py --video VID_20261120_073015.mp4 --calibracion calibracion.json --limite 60
```

### Qué se genera (`salidas/<video>/`)

| Archivo | Contenido |
|---|---|
| `historial.csv` | **Historial de todos los vehículos:** cuadro, hora real, ID, clase, caja, punto de contacto, posición en metros (`s_m` a lo largo de la calle, `d_m` a lo ancho), si está en la zona, velocidad y estado. Se registra 5 veces por segundo por vehículo (`--registro-hz`). |
| `conteo.csv` | Una fila por vehículo que **cruzó la línea de conteo**: hora, clase, sentido, posición a lo ancho y velocidad |
| `vehiculos_contados.csv` | Lo mismo, pero con el **carril asignado** a cada vehículo |
| `carriles.csv` | Vehículos, porcentaje, sentido y velocidad media **por carril** |
| `mapa_calor.png` | **Mapa de calor** sobre la calle vacía, con los carriles numerados y el **más usado** resaltado |
| `vehiculos_por_hora.png` | **Vehículos por hora del día**, separados por clase, con la hora pico; debajo, el detalle por minutos si el video es corto |
| `resumen_nivel2.txt` | El resumen en texto (útil para el informe) |
| `fondo.jpg`, `sesion.json` | La calle sin autos (la mediana de varios cuadros) y los datos de la sesión |

Los CSV se abren en Excel. Si todo aparece en una sola columna, usa *Datos → Desde texto/CSV*.

### La hora real (para la gráfica por hora)
El programa la busca en este orden:
1. `--hora-inicio "2026-11-20 07:30"`, o solo `"07:30"` para usar la fecha de hoy.
2. El **nombre del archivo** del celular: `VID_20261120_073015.mp4`, `PXL_...`, `WhatsApp Video 2026-11-20 at 07.30.15`.
3. La fecha interna del **MP4**. Verifícala: algunos celulares la guardan con otra zona horaria.

Si no la encuentra, la gráfica usa minutos desde el inicio del video, y el resumen te avisa.

### Cómo funciona (para el informe)
1. **Línea virtual de conteo.** Cruza la calle por el centro de la zona de medición y se dibuja en el video. Un vehículo se cuenta **una sola vez**: en el instante en que su punto de contacto con el suelo cruza la línea. Ese instante se interpola entre cuadros. Así no se cuenta doble cuando el tracker cambia el ID de un auto.
2. **Ejes de la vía.** Con la calibración, cada punto se pasa a metros `(s, d)`. `s` va a lo largo de la calle y `d` a lo ancho. En PPM, el eje `s` va de P1 a P2; en homografía, de P2 a P3.
3. **Carriles automáticos.** Se arma el histograma de la posición `d` donde cada vehículo cruzó la línea, se suaviza, y **cada pico es un carril** (separados por al menos 2 m). Después se refina con k-means 1D. Puedes forzar el número con `--carriles 4`. El carril 1 es el de más arriba en la imagen (o el de más a la izquierda si la calle va de arriba hacia abajo).
4. **Mapa de calor.** Cada trayectoria se dibuja como una franja, y **cada vehículo suma 1** en los píxeles por donde pasó. El color indica cuántos vehículos pasaron por ese punto. Se usa una sola escala de azul, de claro a oscuro, sobre la foto de la calle aclarada.
5. **Gráfica por hora.** Cuenta los vehículos por hora según la hora del cruce. Como cada hora puede tener **distinto tiempo grabado**, también calcula el **flujo equivalente (veh/h)** y con él elige la hora pico. Las horas sin grabación **no aparecen como 0**, porque no hay datos de ellas.

### Juntar varios clips (el día completo)
Graba clips desde el mismo punto a distintas horas. Procesa cada uno con el radar, usando **la misma calibración**, y luego júntalos:
```bash
python radar_velocidad.py --video VID_20261120_070012.mp4 --limite 60 --no-mostrar
python radar_velocidad.py --video VID_20261120_123005.mp4 --limite 60 --no-mostrar
python radar_velocidad.py --video VID_20261120_180130.mp4 --limite 60 --no-mostrar
python analitica.py salidas/VID_20261120_070012 salidas/VID_20261120_123005 salidas/VID_20261120_180130 --salida salidas/dia_completo
```
`analitica.py` también sirve para regenerar los gráficos de un solo video, por ejemplo con otro número de carriles:
```bash
python analitica.py salidas/tu_video --carriles 4
```

### Opciones del Nivel 2
| Opción | Por defecto | Para qué |
|---|---|---|
| `--hora-inicio` | (automática) | Hora real del inicio del video |
| `--carriles` | auto | Número de carriles si la detección automática no acierta |
| `--linea-conteo` | 0.5 | Posición de la línea a lo largo de la zona (0 = un extremo, 1 = el otro) |
| `--registro-hz` | 5 | Registros por segundo y vehículo en `historial.csv` (0 = todos los cuadros; los archivos crecen mucho) |
| `--sin-analitica` | — | No generar los gráficos al cerrar |

### Validación del Nivel 2
Para probarlo sin video propio, usa `pruebas/VID_20261120_075955.mp4`, un video sintético que empieza a las 07:59:55:
```bash
python calibrar.py --video pruebas/VID_20261120_075955.mp4 --modo ppm --puntos "100,650;1100,650" --metros 50 --salida pruebas/calib_n2.json
python radar_velocidad.py --video pruebas/VID_20261120_075955.mp4 --calibracion pruebas/calib_n2.json --limite 50
```

| | Esperado | Obtenido |
|---|---|---|
| Vehículos contados | 8 | 8 |
| Carril de arriba (←) / de abajo (→) | 3 / 5 | 3 / 5, el de abajo como más usado |
| Hora 07:00 / 08:00 | 3 / 5 | 3 / 5 (la hora se leyó del nombre del archivo) |
| Velocidades | 48.6 / 64.8 km/h | 48.7 / 64.9 km/h |

**Limitación conocida:** en una vista **lateral**, los vehículos del carril de adelante tapan a los de atrás, y los carriles de atrás pueden quedar con algunos vehículos menos. La vista desde un puente peatonal, mirando a lo largo de la calle, evita ese problema.

---

## 9. Nivel 3 — Odometría visual (cámara en movimiento)

### El problema
Si aplicas el radar del Nivel 1 a una cámara que **se mueve**, todo lo que está quieto se desplaza en la imagen. El radar mide movimiento en la imagen, así que cree que **el suelo, las paredes y los autos estacionados se mueven** a la velocidad de la cámara.

### La solución (`radar/odometria.py`)
1. **FAST** detecta esquinas en el suelo y **BRIEF orientado** las describe con 256 bits; juntos forman **ORB**. Una rejilla reparte los puntos por toda la imagen para que no se amontonen.
2. Los puntos se **emparejan** con los del cuadro anterior por distancia de Hamming, y se aplica el test de Lowe (la mejor pareja debe ser claramente mejor que la segunda).
3. Los puntos emparejados se pasan a **metros sobre el suelo** con la calibración. Como el celular va **a altura y ángulo fijos**, la calibración sirve para todos los cuadros. Vista desde arriba, la escena solo se **traslada y gira** de un cuadro a otro.
4. **RANSAC** encuentra el movimiento rígido 2D (giro y traslación) que explica a la mayoría de los puntos. Deja fuera lo que no encaja: autos que se mueven y puntos de paredes. Después se refina con **Kabsch (SVD)**. YOLO también tapa autos y personas para que no aporten puntos.
5. **Encadenando** esos movimientos se obtiene la **pose de la cámara**: su trayectoria en metros, su velocidad y su giro.
6. Cada auto se lleva de las coordenadas de la cámara a las **coordenadas del mundo** con esa pose. Allí se mide con el **mismo estimador del Nivel 1**. Un auto estacionado queda siempre en el mismo lugar, así que marca 0 km/h; uno que circula muestra su velocidad real.

**¿De dónde sale la escala?** Con una sola cámara, la matriz esencial (5 puntos) da la **dirección** del movimiento, pero no **cuántos metros** se movió: la escala es ambigua. Aquí la escala sale del **plano del suelo calibrado**, porque se conocen las medidas reales del rectángulo marcado.

### Uso
```bash
# 1) Calibrar con un cuadro del inicio (quieto) del MISMO video. Rectángulo medido con wincha.
#    En la fase 2, dibuja una zona que cubra TODO el suelo visible: los puntos ORB se buscan solo ahí.
python calibrar.py --video caminata.mp4 --modo homografia --salida calib_movil.json

# 2) Calle con autos: antes y después, lado a lado
python odometria_visual.py --video caminata.mp4 --calibracion calib_movil.json

# 2b) Pasillo sin vehículos ni personas: solo odometría, mucho más rápido
python odometria_visual.py --video pasillo.mp4 --calibracion calib_pasillo.json --sin-yolo
```
Si el celular graba mirando **recto hacia el piso** (cenital), también sirve `--modo ppm`.

| Opción | Por defecto | Para qué |
|---|---|---|
| `--sin-yolo` | — | Solo odometría. Úsalo **solo si no hay vehículos ni personas** moviéndose en la imagen |
| `--orb-puntos` | 1500 | Puntos ORB por cuadro (más puntos = más robusto, pero más lento) |
| `--ransac-m` | 0.12 | Tolerancia de RANSAC en metros |
| `--umbral-estacionado` | 3 | Un vehículo con velocidad real menor a esta (km/h) se considera estacionado |
| `--tracker` | `botsort.yaml` | BoT-SORT compensa el movimiento de la cámara al seguir vehículos |
| `--vista` | `comparar` | `compensado` muestra solo el resultado |

### Qué se genera (`salidas/<video>_nivel3/`)
| Archivo | Contenido |
|---|---|
| `<video>_odometria.mp4` | **Antes y después** lado a lado. Izquierda: el radar del Nivel 1, con flechas naranjas del "movimiento" falso del suelo. Derecha: la versión compensada, con los puntos del suelo usados, la etiqueta ESTACIONADO o EN MOVIMIENTO y un **minimapa** de la trayectoria |
| `odometria.png` | **Mapa desde arriba** con la trayectoria de la cámara y los vehículos, y la **velocidad de la cámara** en el tiempo |
| `comparacion_velocidades.png` | Por cada vehículo: velocidad **aparente** (Nivel 1) frente a velocidad **real** (Nivel 3) |
| `odometria.csv` | Por cuadro: posición, giro, velocidad de la cámara, puntos ORB, parejas e inliers |
| `vehiculos_nivel3.csv`, `vehiculos_historial_nivel3.csv` | Velocidad aparente y real de cada vehículo, y su posición en el mundo |
| `reporte_nivel3.txt` | Resumen para el informe |

### Validación del Nivel 3
Se usó `pruebas/camara_movil.mp4`, un video sintético: la cámara avanza a 18 km/h con balanceo, pasa junto a buses estacionados y hay un bus circulando a 27 km/h.
```bash
python calibrar.py --video pruebas/camara_movil.mp4 --modo homografia --puntos "340,700;940,700;760,420;520,420" --ancho 6 --largo 12 --zona "0,330;1279,330;1279,719;0,719" --salida pruebas/calib_movil.json
python odometria_visual.py --video pruebas/camara_movil.mp4 --calibracion pruebas/calib_movil.json
```
| | Real | Nivel 1 (sin compensar) | Nivel 3 (odometría) |
|---|---|---|---|
| Recorrido de la cámara | 60.0 m | — | 60.2 m |
| Velocidad de la cámara | 18.0 km/h | — | 18.1 km/h |
| Buses estacionados | 0 km/h | 16 a 20 km/h ❌ | 1.2 a 2.1 km/h ✔ (bajo el umbral de 3 km/h) |
| Bus en movimiento | 27.0 km/h | 9.1 km/h ❌ | 26.7 km/h ✔ |

Con estacionados no da 0 exacto: queda entre 1 y 2 km/h por el temblor de las cajas de YOLO. Por eso existe el umbral.

**Limitaciones (para el informe):**
- Se asume que el celular mantiene **altura e inclinación constantes**. El balanceo al caminar agrega unos pocos puntos de error.
- Los suelos **sin textura** (lisos, brillantes) dan pocos puntos ORB, y el programa avisa con "odometría PERDIDA".
- El error se **acumula** (deriva): en recorridos largos, el mapa se va torciendo poco a poco.
