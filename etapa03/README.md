# Monitor Inteligente de Tráfico — Nivel 1: Radar de Velocidad (cámara fija)

Analiza un **video subido** (o la webcam), detecta y sigue los vehículos con **YOLO + ByteTrack**, convierte su movimiento a metros con una **calibración hecha con Google Maps** y muestra la **velocidad en km/h** de cada vehículo. Si un vehículo supera el límite, **su caja se vuelve roja**, parpadea y se guarda una **captura de la infracción**.

```
radar_nivel1/
├── calibrar.py              ← PASO 1: marcar la distancia real (Google Maps) sobre el video
├── radar_velocidad.py       ← PASO 2: el radar
├── requirements.txt
├── radar/
│   ├── calibracion.py       píxeles → metros (PPM u homografía) + zona de medición
│   ├── velocidad.py         velocidad por vehículo (regresión + suavizado) y control del límite
│   ├── dibujo.py            cajas, colores, panel de información
│   └── video.py             abrir video / explorador de archivos / FPS
└── pruebas/
    ├── test_radar.py                 pruebas con velocidades conocidas
    ├── test_calibrar_interactivo.py  simula clics y teclas de la calibración
    ├── generar_video_sintetico.py    crea un video con velocidades conocidas
    └── video_sintetico.mp4           (64.8 km/h y 32.4 km/h)
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

Se abre una ventana con una **lupa** en la esquina para marcar con precisión:

| Fase | Qué hacer |
|---|---|
| **1. Puntos** | **PPM:** 2 clics. **Homografía:** 4 clics en este orden: P1 cercano izquierdo → P2 cercano derecho (eso es el **ancho**) → P3 lejano del lado de P2 (eso es el **largo**) → P4 lejano del lado de P1. Pulsa ENTER y escribe las medidas **en la terminal**. |
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
Hay 11 pruebas: velocidad con PPM, homografía con perspectiva, zona de medición, saltos de ID, cambio de resolución, guardar y cargar, control del límite y el flujo interactivo de calibración.

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

**Siguiente paso: Nivel 2.** El estimador ya guarda los datos de cada vehículo (`EstimadorVelocidad.stats`). En el Nivel 2 se agregará el registro cuadro a cuadro en CSV, el mapa de calor por carril y la gráfica de autos por hora.
