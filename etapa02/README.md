# Etapa 02 – Tracking 2D, trayectorias, velocidad y cámara en movimiento

Código para exponer las 5 partes de la Etapa 02. Todo funciona con **la cámara web** (`--fuente 0`) o con **videos**
(hay videos reales de prueba en `datos/`).

| Parte | Tema | Script |
|---|---|---|
| 1 | ¿Por qué la VO / pose 3D es más difícil que el tracking 2D? | `parte1_dificultad_vo.py` |
| 2 | Tracking 2D con YOLO + ByteTrack / BoT-SORT y conteo con línea virtual | `parte2_tracking_conteo.py` |
| 3 | Guardar y analizar trayectorias (ID, frame, X, Y) | `parte3_analisis_trayectorias.py` |
| 4 | Velocidad en tiempo real (px/s → m/s), zonas y mapa de calor | `parte4_velocidad_zonas.py` + `calibrar.py` |
| 5 | Cámara en movimiento: solución con odometría visual | `parte5_camara_movil.py` (+ `extra_autopista/`) |

`comun.py` contiene las piezas compartidas (tracker, línea de conteo, registro CSV, calibración, velocímetro,
mapa de calor, zonas y odometría visual 2D).

---

## 1. Instalación (Windows)

Se necesita Python 3.9 a 3.12. En una terminal (cmd o PowerShell) dentro de esta carpeta:

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

En Linux/macOS: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`.

El modelo `yolo11n.pt` ya está incluido. La primera vez que se usa BoT-SORT o ByteTrack, ultralytics puede instalar
`lap` automáticamente.

**Consejos:** si la cámara web no abre, pruebe `--fuente 1`. Si va lento, agregue `--imgsz 480`.
Teclas comunes: `q` o `ESC` salir, `p` pausa.

---

## 2. Comandos por parte

### Parte 1 – Por qué la VO es más difícil
```bat
python parte1_dificultad_vo.py --modo simulacion
python parte1_dificultad_vo.py --modo vivo --fuente 0 --yolo
```
* `simulacion` genera en `salidas/parte1/` tres experimentos: deriva acumulada (VO) vs. error acotado (2D),
  ambigüedad de escala y mundo dinámico.
* `vivo`: VO monocular con la cámara web. Mueva la cámara y vuelva al punto de inicio: la trayectoria no regresa a
  cero (deriva) y no tiene unidades (escala desconocida). `r` reinicia. Con `--yolo` se ve el tracking 2D a la vez.

### Parte 2 – Tracking + conteo
```bat
python parte2_tracking_conteo.py --fuente 0
python parte2_tracking_conteo.py --fuente datos/peatones_vtest.avi --linea 0.5,0.3,0.5,0.78 --guardar-video
python parte2_tracking_conteo.py --fuente datos/trafico_urbano.mp4 --clases car,truck,bus,motorcycle --linea 0,0.6,1,0.6 --tracker botsort
```
* `--tracker bytetrack|botsort`, `--clases person,car,...`, `--linea x1,y1,x2,y2` (fracciones de la imagen).
* En vivo: tecla `l` y **dos clics** para dibujar otra línea; `r` reinicia el conteo.
* Salidas en `salidas/parte2/<video>_<tracker>/`: `trayectorias.csv` (frame, t_s, id, clase, conf, cx, cy, caja),
  `conteo_eventos.csv`, `resumen.json`, `fondo.jpg` y `video_resultado.mp4` (con `--guardar-video`).

### Parte 3 – Análisis de trayectorias
```bat
python parte3_analisis_trayectorias.py --carpeta salidas/parte2/peatones_vtest_bytetrack
python parte3_analisis_trayectorias.py --carpeta salidas/parte2/webcam_bytetrack --k 3
```
Genera en `analisis/`: `trayectorias.png`, `cubo_espacio_tiempo.png`, `direcciones.png`, `origen_destino.png`,
`patrones.png` (k-means), `anomalias.png`, `actividad.png`, `estadisticas_por_id.csv` y `resumen_analisis.json`.

### Parte 4 – Velocidad, zonas y mapa de calor
```bat
python parte4_velocidad_zonas.py --fuente 0 --persona
python parte4_velocidad_zonas.py --fuente datos/trafico_urbano.mp4 --clases car,truck,bus --calibracion calib/trafico_urbano.json --unidad km/h --guardar-video
python parte4_velocidad_zonas.py --fuente datos/peatones_vtest.avi --persona --filas 2 --cols 3
```
Conversión px → metros (elija una):
* `--ppm 38.5` factor único (solo válido cerca de donde se midió),
* `--calibracion archivo.json` homografía con 4 puntos del suelo (**recomendado**),
* `--persona` escala local con la altura media de una persona (1.70 m), sin calibrar.

Crear la calibración de su propia cámara/video:
```bat
python calibrar.py --fuente 0 --modo homografia --ancho 7 --largo 12 --salida calib/mi_camara.json
python calibrar.py --fuente 0 --modo ppm --distancia 3.5 --salida calib/mi_ppm.json
```
(clic en 4 esquinas de un rectángulo del suelo: cerca-izq, cerca-der, lejos-izq, lejos-der).
Teclas: `h` mapa de calor, `z` zonas, `v` vista de pájaro. Salidas en `salidas/parte4/`.

### Parte 5 – Cámara en movimiento
```bat
python parte5_camara_movil.py --fuente datos/trafico_urbano_pan.mp4 --clases car,truck,bus --calibracion calib/trafico_urbano_pan.json --unidad km/h --linea 0,0.6,1,0.6 --gt datos/trafico_urbano_pan_gt.json --guardar-video
python parte5_camara_movil.py --fuente 0 --persona
```
Muestra lado a lado **sin VO** (todo en coordenadas de pantalla) y **con VO** (todo en el sistema del frame 0, fijo
al mundo). `r` toma el frame actual como nueva referencia. Con `--gt` calcula el error contra la verdad.
Salidas: `sin_vs_con_vo.mp4`, `trayectorias_sin_vo.png`, `trayectorias_con_vo.png`, `mapa_calor_mundo.png`,
`resumen.json`.

`extra_autopista/` contiene el caso avanzado con un **video real de autopista** (cámara que avanza): odometría visual
monocular + plano de la vía para obtener la velocidad real de los otros autos. Ver `extra_autopista/LEEME.txt`.

---

## 3. Resultados obtenidos con los videos de prueba

| Parte | Resultado |
|---|---|
| 1 | Con 1 px de ruido: error 2D ≈ 1 px (acotado); error de la VO de 2 a 7 m en una vuelta de 126 m. Desde ~20 % de puntos sobre un objeto en movimiento, la dirección estimada falla ~70°. |
| 2 | Plaza: 795 frames, 34 IDs (ByteTrack) / 32 (BoT-SORT), 20 cruces (15 A→B, 5 B→A). Avenida: 37 vehículos. |
| 3 | 31 trayectorias, 4 517 filas; duración media 15.1 s; 4 flujos; 7 comportamientos a revisar. |
| 4 | Avenida: 51 vehículos, mediana 28.5 km/h (homografía). Plaza: mediana 0.9 m/s (escala por persona). |
| 5 | Video con paneo: error de velocidad mediano 10.7 → 1.9 km/h. Autopista: vecinos a +3 km/h relativos → 104–109 km/h reales. |

Las simulaciones 3D para exponer están en `simulaciones_3d_etapa02.html` (se abre con Chrome o Edge, funciona sin
internet). Teclas 1 a 5 cambian de parte; espacio pausa.
