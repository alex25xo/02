"""
NIVEL 2 - VIDEO DEMOSTRATIVO (sin volver a usar YOLO)

Toma lo que el radar YA guardó al procesar un clip (historial.csv, conteo.csv,
sesion.json) y tus carriles (carriles_nivel2.json), y lo dibuja sobre el video
original:
  - los carriles pintados de colores, con su nombre
  - cada vehículo con su caja, su velocidad y el carril por el que va
  - los excesos de velocidad en ROJO
  - la línea de conteo, que destella cuando un vehículo la cruza
  - contadores en vivo: total, excesos y barras de vehículos por carril

Como no vuelve a detectar nada, es rápido y muestra EXACTAMENTE los mismos datos
de los reportes.

Uso (desde la carpeta del proyecto):
    python video_nivel2.py salidas\\D22N1730A1737 --carriles-def carriles_nivel2.json --duracion 90
    python video_nivel2.py salidas\\D22N1730A1737 --carriles-def carriles_nivel2.json            (clip completo)
    python video_nivel2.py salidas\\D22N1730A1737 --carriles-def carriles_nivel2.json --desde 120 --duracion 60

Si movieron el video de lugar:  --video ..\\VPNV2\\D22N1730A1737.mp4
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from radar.analitica import asignar_por_bordes, cargar_bordes, detectar_carriles
from radar.calibracion import Calibracion
from radar.conteo import ContadorLinea
from radar.dibujo import ajustar_para_pantalla, etiqueta

COLORES = [(214, 120, 42), (52, 104, 235), (122, 175, 27), (0, 161, 237), (214, 80, 98), (150, 150, 150)]
ROJO = (0, 0, 235)
FUENTE = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------- #
# Datos guardados por el radar
# ---------------------------------------------------------------------- #
def leer_historial(ruta: Path):
    """id -> dict con arrays ordenados por tiempo (muestras ~5 por segundo)."""
    filas: dict[int, list] = {}
    with open(ruta, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            v = r["vel_kmh"]
            filas.setdefault(int(r["id"]), []).append((
                float(r["t_video_s"]), float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]),
                float(r["d_m"]), float(v) if v else math.nan, r["estado"], r["clase"]))
    hist = {}
    for tid, m in filas.items():
        m.sort(key=lambda x: x[0])
        hist[tid] = {"t": [x[0] for x in m], "caja": np.array([x[1:5] for x in m]),
                     "d": np.array([x[5] for x in m]), "v": np.array([x[6] for x in m]),
                     "estado": [x[7] for x in m], "clase": max(set(x[8] for x in m), key=[x[8] for x in m].count)}
    return hist


def leer_conteo(ruta: Path):
    with open(ruta, encoding="utf-8-sig") as f:
        ev = [{"id": int(r["id"]), "t": float(r["t_video_s"]), "d": float(r["d_m"]),
               "sentido": int(r["sentido"])} for r in csv.DictReader(f)]
    return sorted(ev, key=lambda e: e["t"])


def estado_en(h, t, gap_max=0.7, extra=0.25):
    """Caja (interpolada), d, velocidad y estado del vehículo en el instante t, o None."""
    ts = h["t"]
    i = bisect.bisect_right(ts, t) - 1
    if i < 0:
        return None
    if i < len(ts) - 1 and ts[i + 1] - ts[i] <= gap_max:
        a = (t - ts[i]) / (ts[i + 1] - ts[i])
        caja = h["caja"][i] * (1 - a) + h["caja"][i + 1] * a
        d = h["d"][i] * (1 - a) + h["d"][i + 1] * a
    elif t - ts[i] <= extra:
        caja, d = h["caja"][i], h["d"][i]
    else:
        return None
    v = h["v"][i]
    return caja, d, (None if math.isnan(v) else float(v)), h["estado"][i]


# ---------------------------------------------------------------------- #
# Carriles: bordes (a mano o automáticos), numeración y capa de colores
# ---------------------------------------------------------------------- #
def linea_borde(calib, d, s_rango, W, H, n=120):
    ss = np.linspace(s_rango[0], s_rango[1], n)
    px = calib.desde_via(np.c_[ss, np.full_like(ss, d)])
    ok = (px[:, 0] >= -W) & (px[:, 0] < 2 * W) & (px[:, 1] >= -H) & (px[:, 1] < 2 * H)
    return px[ok]


def preparar_carriles(calib, contador, bordes, eventos, W, H):
    zona = calib.zona_px or [(0, 0), (W, 0), (W, H), (0, H)]
    sd = calib.a_via(zona)
    s_rango = (sd[:, 0].min(), sd[:, 0].max())
    b = sorted(bordes)
    centros = [(b[i] + b[i + 1]) / 2 for i in range(len(b) - 1)]
    px = calib.desde_via(np.c_[np.full(len(centros), contador.s_linea), centros])
    orden = np.argsort(px[:, 1] if contador.via_horizontal else px[:, 0])
    numero = {int(idx): k + 1 for k, idx in enumerate(orden)}          # índice -> número de carril
    # sentido mayoritario por carril (con todo el clip)
    sentidos = {}
    if eventos:
        idx = asignar_por_bordes([e["d"] for e in eventos], b)
        for e, k in zip(eventos, idx):
            sentidos.setdefault(int(k), []).append(e["sentido"])
    flecha = {k: (contador.flecha(max(set(v), key=v.count)) if v else "")
              for k, v in sentidos.items()}
    # capa de colores (estática): una banda por carril
    capa = np.zeros((H, W, 3), np.uint8)
    mascara = np.zeros((H, W), np.uint8)
    etiquetas = []
    for i in range(len(b) - 1):
        a_ = linea_borde(calib, b[i], s_rango, W, H)
        c_ = linea_borde(calib, b[i + 1], s_rango, W, H)
        if len(a_) < 2 or len(c_) < 2:
            continue
        poli = np.vstack([a_, c_[::-1]]).astype(np.int32)
        color = COLORES[(numero[i] - 1) % len(COLORES)]
        cv2.fillPoly(capa, [poli], color)
        cv2.fillPoly(mascara, [poli], 255)
        cen = linea_borde(calib, centros[i], s_rango, W, H)
        cen = cen[(cen[:, 0] >= 0) & (cen[:, 0] < W) & (cen[:, 1] >= 0) & (cen[:, 1] < H)]
        if len(cen):
            j = int(np.argmin(cen[:, 0] if contador.via_horizontal else cen[:, 1]))
            etiquetas.append((numero[i], tuple(int(v) for v in cen[j])))
    lineas = [linea_borde(calib, d, s_rango, W, H).astype(np.int32) for d in b]
    return {"bordes": b, "numero": numero, "flecha": flecha, "capa": capa, "mascara": mascara,
            "etiquetas": etiquetas, "lineas": lineas, "n": len(b) - 1}


# ---------------------------------------------------------------------- #
# Dibujo
# ---------------------------------------------------------------------- #
def panel_carriles(img, car, cuenta, esc, x0, y0):
    """Barras en vivo: vehículos contados por carril."""
    n = car["n"]
    fila, ancho_barra = int(34 * esc), int(230 * esc)
    w_panel, h_panel = int(430 * esc), int(46 * esc) + fila * n
    sub = img[y0:y0 + h_panel, x0:x0 + w_panel]
    cv2.addWeighted(sub, 0.3, np.full_like(sub, 20), 0.7, 0, dst=sub)
    cv2.putText(img, "VEHICULOS POR CARRIL", (x0 + int(12 * esc), y0 + int(28 * esc)), FUENTE, 0.62 * esc,
                (255, 255, 255), max(1, int(2 * esc)), cv2.LINE_AA)
    top = max([cuenta.get(k, 0) for k in range(n)] + [1])
    total = max(1, sum(cuenta.values()))
    for k_idx in sorted(range(n), key=lambda i: car["numero"][i]):
        num = car["numero"][k_idx]
        y = y0 + int(46 * esc) + fila * (num - 1)
        color = COLORES[(num - 1) % len(COLORES)]
        cv2.putText(img, f"C{num} {car['flecha'].get(k_idx, '')}", (x0 + int(12 * esc), y + int(22 * esc)), FUENTE,
                    0.58 * esc, (255, 255, 255), max(1, int(1.5 * esc)), cv2.LINE_AA)
        c = cuenta.get(k_idx, 0)
        largo = int(ancho_barra * c / top)
        bx = x0 + int(92 * esc)
        cv2.rectangle(img, (bx, y + int(6 * esc)), (bx + max(2, largo), y + int(28 * esc)), color, -1)
        cv2.putText(img, f"{c} ({100 * c / total:.0f}%)", (bx + max(2, largo) + int(8 * esc), y + int(23 * esc)),
                    FUENTE, 0.55 * esc, (255, 255, 255), max(1, int(1.5 * esc)), cv2.LINE_AA)


def dibujar_vehiculo(img, caja, tid, clase, v, estado, num_carril, contado, parpadeo, esc):
    x1, y1, x2, y2 = (int(round(c)) for c in caja)
    exceso = estado == "exceso"
    color = ROJO if exceso else (COLORES[(num_carril - 1) % len(COLORES)] if num_carril else (180, 180, 180))
    if exceso and parpadeo:
        sub = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if sub.size:
            cv2.addWeighted(sub, 0.7, np.full_like(sub, (0, 0, 220)), 0.3, 0, dst=sub)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, max(2, int((3 if exceso else 2) * esc)), cv2.LINE_AA)
    cv2.circle(img, ((x1 + x2) // 2, y2), max(3, int(4 * esc)), color, -1, cv2.LINE_AA)
    vel = f"{v:.0f} km/h" if v is not None else "-- km/h"
    texto = f"#{tid} {clase} {vel}" + (f" | C{num_carril}" if num_carril else "")
    etiqueta(img, texto, x1, y1 - 3, color, 0.55 * esc, max(1, int(1.5 * esc)))
    if exceso:
        etiqueta(img, "EXCESO DE VELOCIDAD", x1, y2 + int(24 * esc), (0, 0, 200), 0.55 * esc, max(1, int(2 * esc)))
    if contado:
        cx, cy = x2 - int(10 * esc), y1 + int(10 * esc)
        cv2.circle(img, (cx, cy), max(5, int(8 * esc)), (60, 200, 60), -1, cv2.LINE_AA)
        cv2.putText(img, "OK", (cx - int(7 * esc), cy + int(4 * esc)), FUENTE, 0.32 * esc, (255, 255, 255),
                    1, cv2.LINE_AA)


# ---------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Video demostrativo del Nivel 2 (sin YOLO)")
    ap.add_argument("carpeta", help="Carpeta de salidas del clip, ej. salidas\\D22N1730A1737")
    ap.add_argument("--carriles-def", default=None, help="carriles_nivel2.json (sin esto: carriles automaticos)")
    ap.add_argument("--video", default=None, help="Ruta del video original si no esta donde se proceso")
    ap.add_argument("--desde", type=float, default=0.0, help="Segundo del clip donde empieza el video")
    ap.add_argument("--duracion", type=float, default=None, help="Segundos a generar (por defecto, todo)")
    ap.add_argument("--ancho", type=int, default=1280, help="Ancho del video de salida (alto proporcional)")
    ap.add_argument("--salida", default=None, help="Archivo de salida (.mp4)")
    ap.add_argument("--mostrar", action="store_true", help="Ver la ventana mientras se genera (mas lento)")
    args = ap.parse_args()

    carpeta = Path(args.carpeta)
    for nombre in ("sesion.json", "historial.csv", "conteo.csv"):
        if not (carpeta / nombre).exists():
            sys.exit(f"Falta {nombre} en {carpeta}. Primero procesa el clip con radar_velocidad.py o procesar_lote.py")
    meta = json.loads((carpeta / "sesion.json").read_text(encoding="utf-8"))
    c = meta["calibracion"]
    calib = Calibracion(c["modo"], tuple(c["resolucion"]), ppm=c.get("ppm"), H=c.get("H"),
                        puntos_px=c.get("puntos_px"), medidas_m=c.get("medidas_m"), zona_px=c.get("zona_px"))
    ruta_video = Path(args.video or meta["video"])
    if not ruta_video.exists():
        sys.exit(f"No encuentro el video original: {ruta_video}\nIndicalo con --video ruta\\al\\video.mp4")
    cap = cv2.VideoCapture(str(ruta_video))
    fps = float(meta.get("fps") or cap.get(cv2.CAP_PROP_FPS) or 30)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (W, H) != calib.resolucion:
        calib = calib.ajustar_resolucion(W, H)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hora0 = datetime.fromisoformat(meta["hora_inicio"]) if meta.get("hora_inicio") else None
    limite = meta.get("limite_kmh", 60)

    hist = leer_historial(carpeta / "historial.csv")
    eventos = leer_conteo(carpeta / "conteo.csv")
    contador = ContadorLinea(calib, (W, H), fraccion=meta.get("linea_conteo", {}).get("fraccion", 0.5))

    bordes = cargar_bordes(args.carriles_def, calib) if args.carriles_def else None
    origen_carriles = "definidos a mano"
    if not bordes:
        centros = detectar_carriles([e["d"] for e in eventos]) if eventos else np.array([0.0])
        medios = [(a + b) / 2 for a, b in zip(centros[:-1], centros[1:])]
        bordes = [centros[0] - 2.0] + medios + [centros[-1] + 2.0]
        origen_carriles = "automaticos"
    car = preparar_carriles(calib, contador, bordes, eventos, W, H)
    idx_ev = asignar_por_bordes([e["d"] for e in eventos], car["bordes"]) if eventos else []
    for e, k in zip(eventos, idx_ev):
        e["carril"] = int(k)
    t_conteo = {e["id"]: e["t"] for e in eventos}
    t_exceso = {}
    for tid, h in hist.items():
        for t, est in zip(h["t"], h["estado"]):
            if est == "exceso":
                t_exceso[tid] = t
                break

    # rango a generar
    f0 = max(0, int(args.desde * fps))
    f1 = total if args.duracion is None else min(total, f0 + int(args.duracion * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    esc = W / 1280.0
    salida = Path(args.salida) if args.salida else carpeta / f"{carpeta.name}_demo_nivel2.mp4"
    w_out = min(args.ancho, W)
    h_out = int(round(H * w_out / W / 2) * 2)
    escritor = None
    try:                                                 # silencia el aviso si H.264 no está disponible
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except AttributeError:
        pass
    for codec in ("avc1", "mp4v"):                       # H.264 si el equipo lo tiene; si no, MPEG-4
        escritor = cv2.VideoWriter(str(salida), cv2.VideoWriter_fourcc(*codec), fps, (w_out, h_out))
        if escritor.isOpened():
            break
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_WARNING)
    except AttributeError:
        pass
    if not escritor.isOpened():
        sys.exit("No se pudo crear el archivo de video de salida.")
    print(f"Clip: {carpeta.name} | {W}x{H} @ {fps:.0f} fps | carriles {origen_carriles}: {car['n']}")
    print(f"Generando segundos {f0 / fps:.0f} a {f1 / fps:.0f} -> {salida}")

    ts_ev = [e["t"] for e in eventos]
    ids_orden = sorted(hist, key=lambda k: hist[k]["t"][0])
    t_ini = [hist[k]["t"][0] for k in ids_orden]
    reloj = time.perf_counter()
    ultimo = reloj
    for f in range(f0, f1):
        ok, img = cap.read()
        if not ok:
            break
        t = f / fps
        # carriles pintados (solo sobre la pista)
        mezcla = cv2.addWeighted(img, 0.72, car["capa"], 0.28, 0)
        cv2.copyTo(mezcla, car["mascara"], img)
        for ln in car["lineas"]:
            if len(ln) > 1:
                cv2.polylines(img, [ln.reshape(-1, 1, 2)], False, (255, 255, 255), max(1, int(1.5 * esc)), cv2.LINE_AA)
        for num, (x, y) in car["etiquetas"]:
            etiqueta(img, f"Carril {num}", x + int(8 * esc), y + int(8 * esc),
                     COLORES[(num - 1) % len(COLORES)], 0.6 * esc, max(1, int(2 * esc)))
        # línea de conteo (destella en verde al cruzar alguien)
        j = bisect.bisect_right(ts_ev, t)
        recien = j > 0 and t - ts_ev[j - 1] < 0.4
        if contador.segmento_px:
            (xa, ya), (xb, yb) = contador.segmento_px
            cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (0, 0, 0), max(3, int(5 * esc)), cv2.LINE_AA)
            cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (60, 230, 60) if recien else (255, 255, 255),
                     max(2, int(3 * esc)), cv2.LINE_AA)
        # vehículos activos
        parpadeo = (f // max(1, int(fps / 4))) % 2 == 0
        hasta = bisect.bisect_right(t_ini, t + 0.05)
        for tid in ids_orden[max(0, hasta - 400):hasta]:
            h = hist[tid]
            if t > h["t"][-1] + 0.3:
                continue
            r = estado_en(h, t)
            if r is None:
                continue
            caja, d, v, est = r
            k = int(asignar_por_bordes([d], car["bordes"])[0])
            dentro = car["bordes"][0] - 1.0 <= d <= car["bordes"][-1] + 1.0
            num = car["numero"].get(k) if dentro else None
            dibujar_vehiculo(img, caja, tid, h["clase"], v, est, num, t_conteo.get(tid, 1e9) <= t, parpadeo, esc)
        # contadores
        cuenta, n_tot = {}, j
        for e in eventos[:j]:
            cuenta[e["carril"]] = cuenta.get(e["carril"], 0) + 1
        n_exc = sum(1 for te in t_exceso.values() if te <= t)
        hora = (hora0 + timedelta(seconds=t)).strftime("%d/%m/%Y %H:%M:%S") if hora0 else f"t = {t:6.1f} s"
        lineas = ["NIVEL 2 - CONTEO POR CARRIL + RADAR", f"{hora}   limite {limite:.0f} km/h",
                  f"Contados: {n_tot}    Excesos: {n_exc}"]
        x0, y0 = int(12 * esc), int(12 * esc)
        alto = int((24 + 30 * len(lineas)) * esc)
        sub = img[y0:y0 + alto, x0:x0 + int(470 * esc)]
        cv2.addWeighted(sub, 0.3, np.full_like(sub, 20), 0.7, 0, dst=sub)
        for i, l in enumerate(lineas):
            cv2.putText(img, l, (x0 + int(12 * esc), y0 + int((32 + 30 * i) * esc)), FUENTE,
                        (0.68 if i == 0 else 0.62) * esc, (255, 255, 255), max(1, int(2 * esc)), cv2.LINE_AA)
        panel_carriles(img, car, cuenta, esc, W - int(445 * esc), int(12 * esc))

        out = cv2.resize(img, (w_out, h_out), interpolation=cv2.INTER_AREA) if w_out != W else img
        escritor.write(out)
        if args.mostrar:
            cv2.imshow("Video demostrativo - Nivel 2", ajustar_para_pantalla(out)[0])
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
        ahora = time.perf_counter()
        if ahora - ultimo > 3:
            hecho = (f - f0 + 1) / max(1, f1 - f0)
            print(f"  {100 * hecho:5.1f}%  ({(f - f0 + 1) / (ahora - reloj):.0f} cuadros/s)")
            ultimo = ahora
    cap.release()
    escritor.release()
    cv2.destroyAllWindows()
    print(f"\nListo: {salida.resolve()}")


if __name__ == "__main__":
    main()
