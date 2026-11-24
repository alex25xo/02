"""
NIVEL 2 - DEFINIR LOS CARRILES A MANO (más preciso que la detección automática)

Haz clic sobre cada LÍNEA PINTADA que limita los carriles, bordes incluidos,
cerca de la línea de conteo (la blanca). N carriles = N+1 clics.
Ejemplo con 4 carriles (5 clics, de arriba hacia abajo):
    1) borde de arriba de la pista   2) línea discontinua de arriba
    3) línea amarilla central         4) línea discontinua de abajo
    5) borde de abajo de la pista

Mientras marcas, cada línea se dibuja a lo largo de la calle: debe quedar ENCIMA
de la línea pintada. Si se tuerce, la calibración no está bien alineada.

Uso:
    python definir_carriles.py --video ..\\VPNV2\\D22N1657A1704.mp4 --calibracion calibracion_nivel2.json
    -> crea carriles_nivel2.json. Luego agrega   --carriles-def carriles_nivel2.json   a procesar_lote.py

Sin ventana (repetible):  --bordes "x,y;x,y;x,y"   (un punto sobre cada línea)

Teclas: clic izquierdo = marcar | clic derecho / Z = deshacer | A / D = cambiar de cuadro
        R = borrar todo | ENTER = guardar | Q / ESC = salir sin guardar
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from radar.calibracion import Calibracion
from radar.conteo import ContadorLinea
from radar.dibujo import ajustar_para_pantalla, etiqueta, ventana_cerrada
from radar.video import abrir, fps_confiable, resolver_fuente

VENTANA = "Definir carriles - Nivel 2"
COLORES = [(214, 120, 42), (52, 104, 235), (122, 175, 27), (0, 161, 237), (214, 80, 98), (160, 160, 160)]


def linea_borde(calib: Calibracion, d: float, s_rango, W, H, n=120) -> np.ndarray:
    """Puntos (px) de la línea d = constante, a lo largo de la calle."""
    ss = np.linspace(s_rango[0], s_rango[1], n)
    px = calib.desde_via(np.c_[ss, np.full_like(ss, d)])
    ok = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
    return px[ok]


def numeracion(calib, contador, bordes) -> list[int]:
    """Orden de los carriles en la imagen: 1 = el de más arriba (o más a la izquierda)."""
    b = sorted(bordes)
    centros = [(b[i] + b[i + 1]) / 2 for i in range(len(b) - 1)]
    px = calib.desde_via(np.c_[np.full(len(centros), contador.s_linea), centros])
    clave = px[:, 1] if contador.via_horizontal else px[:, 0]
    orden = np.argsort(clave)
    num = [0] * len(centros)
    for k, idx in enumerate(orden):
        num[idx] = k + 1
    return num


def dibujar(frame, calib, contador, clics, s_rango):
    img = frame.copy()
    H, W = img.shape[:2]
    g = max(2, W // 640)
    if calib.zona_px:
        cv2.polylines(img, [np.array(calib.zona_px, np.int32).reshape(-1, 1, 2)], True,
                      (255, 200, 0), 1, cv2.LINE_AA)
    if contador.segmento_px:
        (xa, ya), (xb, yb) = contador.segmento_px
        cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (255, 255, 255), g, cv2.LINE_AA)
    bordes = sorted(calib.a_via(clics)[:, 1].tolist()) if clics else []
    if len(bordes) >= 2:
        capa = img.copy()
        nums = numeracion(calib, contador, bordes)
        for i in range(len(bordes) - 1):
            a = linea_borde(calib, bordes[i], s_rango, W, H)
            b = linea_borde(calib, bordes[i + 1], s_rango, W, H)
            if len(a) > 1 and len(b) > 1:
                poli = np.vstack([a, b[::-1]]).astype(np.int32)
                cv2.fillPoly(capa, [poli], COLORES[(nums[i] - 1) % len(COLORES)])
        cv2.addWeighted(capa, 0.3, img, 0.7, 0, dst=img)
        for i in range(len(bordes) - 1):
            centro = (bordes[i] + bordes[i + 1]) / 2
            c = linea_borde(calib, centro, s_rango, W, H)
            if len(c):
                j = int(np.argmin(c[:, 0] if contador.via_horizontal else c[:, 1]))
                etiqueta(img, f"Carril {nums[i]}", int(c[j][0]) + 10, int(c[j][1]) + 8,
                         (40, 40, 40), 0.7, 2)
    for d in bordes:
        pts = linea_borde(calib, d, s_rango, W, H).astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (0, 255, 255), g, cv2.LINE_AA)
    for p in clics:
        cv2.circle(img, tuple(int(v) for v in p), g + 4, (0, 0, 255), -1, cv2.LINE_AA)
    return img, bordes


def guardar(ruta: Path, bordes, clics, calib, fuente, frame, contador, s_rango):
    datos = {
        "bordes_d": sorted(bordes), "n_carriles": len(bordes) - 1,
        "clics_px": [list(map(float, p)) for p in clics], "video": str(fuente),
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "calibracion": {"modo": calib.modo, "resolucion": list(calib.resolucion),
                        "ppm": calib.ppm, "H": calib.H.tolist() if calib.H is not None else None},
    }
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    img, _ = dibujar(frame, calib, contador, clics, s_rango)
    cv2.imwrite(str(ruta.with_name(ruta.stem + "_preview.jpg")), img)
    print(f"\n{len(bordes) - 1} carriles guardados en: {ruta.resolve()}")
    print(f"Vista previa: {ruta.with_name(ruta.stem + '_preview.jpg').resolve()}")
    print(f"Ahora agrega a tus comandos:  --carriles-def {ruta.name}")


def main():
    ap = argparse.ArgumentParser(description="Definir los carriles a mano (Nivel 2)")
    ap.add_argument("--video", help="Video del mismo encuadre que la calibracion")
    ap.add_argument("--calibracion", default="calibracion.json")
    ap.add_argument("--salida", default="carriles_nivel2.json")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--bordes", help='Sin ventana: "x,y;x,y;..." un punto sobre cada linea')
    args = ap.parse_args()

    if not Path(args.calibracion).exists():
        sys.exit(f"No existe '{args.calibracion}'")
    fuente, es_webcam = resolver_fuente(args.video)
    cap = abrir(fuente)
    fps, _ = fps_confiable(cap)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    if not ok:
        sys.exit("No se pudo leer el video")
    H, W = frame.shape[:2]
    try:
        calib = Calibracion.cargar(args.calibracion).ajustar_resolucion(W, H)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")
    contador = ContadorLinea(calib, (W, H))
    zona = calib.zona_px or [(0, 0), (W, 0), (W, H), (0, H)]
    sd = calib.a_via(zona)
    margen = 0.1 * (sd[:, 0].max() - sd[:, 0].min())
    s_rango = (sd[:, 0].min() - margen, sd[:, 0].max() + margen)

    if args.bordes:                                    # ---------- sin ventana ----------
        clics = [tuple(map(float, p.split(","))) for p in args.bordes.replace(" ", "").split(";") if p]
        if len(clics) < 2:
            sys.exit("Se necesitan al menos 2 puntos (bordes) para 1 carril")
        bordes = sorted(calib.a_via(clics)[:, 1].tolist())
        guardar(Path(args.salida), bordes, clics, calib, fuente, frame, contador, s_rango)
        return

    clics: list[tuple[float, float]] = []             # ---------- interactivo ----------
    estado = {"escala": 1.0}

    def raton(evento, x, y, flags, _):
        X, Y = x / estado["escala"], y / estado["escala"]
        if evento == cv2.EVENT_LBUTTONDOWN:
            clics.append((X, Y))
        elif evento == cv2.EVENT_RBUTTONDOWN and clics:
            clics.pop()

    cv2.namedWindow(VENTANA, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(VENTANA, raton)
    idx = args.frame
    while True:
        img, bordes = dibujar(frame, calib, contador, clics, s_rango)
        vis, estado["escala"] = ajustar_para_pantalla(img, ampliar_max=2.0)
        n = max(0, len(clics) - 1)
        lineas = [f"Clic sobre cada LINEA que limita los carriles (bordes incluidos), cerca de la linea blanca",
                  f"Marcadas: {len(clics)} lineas = {n} carriles.  4 carriles = 5 clics (de arriba a abajo)",
                  "Las lineas amarillas deben quedar ENCIMA de las pintadas.  ENTER guardar | Z deshacer | A/D cuadro"]
        for i, l in enumerate(lineas):
            etiqueta(vis, l, 8, 26 + 26 * i, (30, 30, 30), 0.55)
        cv2.imshow(VENTANA, vis)
        k = cv2.waitKey(20) & 0xFF
        if ventana_cerrada(VENTANA) or k in (27, ord("q")):
            print("Cancelado: no se guardo nada.")
            break
        if k in (ord("z"), 8) and clics:
            clics.pop()
        elif k == ord("r"):
            clics.clear()
        elif k in (ord("a"), ord("d")) and not es_webcam:
            nuevo = max(0, idx + (1 if k == ord("d") else -1) * int(round(fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, nuevo)
            ok, fr = cap.read()
            if ok:
                frame, idx = fr, nuevo
        elif k in (13, 10):
            if len(clics) < 2:
                print("Marca al menos 2 lineas.")
                continue
            guardar(Path(args.salida), bordes, clics, calib, fuente, frame, contador, s_rango)
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
