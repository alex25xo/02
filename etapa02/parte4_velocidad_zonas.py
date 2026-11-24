"""
PARTE 4 - Velocidad en tiempo real (px/s -> m/s) + zonas + mapa de calor
=======================================================================
La velocidad que sale "gratis" del tracking esta en px/s. Para pasarla a m/s hace falta saber
cuantos metros mide un pixel EN EL SUELO, y eso cambia con la profundidad (perspectiva):
  --ppm 38.5                    factor unico px/m (solo valido si la camara mira perpendicular al plano)
  --calibracion calib/x.json    homografia imagen->suelo (4 puntos de medidas conocidas)  <- recomendado
  --persona                     escala local = 1.70 m / altura de la caja de cada persona (sin calibrar)
Para crear el JSON:  python calibrar.py --fuente VIDEO --modo homografia --ancho 7 --largo 12

Ejemplos:
  python parte4_velocidad_zonas.py --fuente 0 --persona                       (camara web, personas)
  python parte4_velocidad_zonas.py --fuente datos/trafico_urbano.mp4 --clases car,truck,bus --calibracion calib/trafico_urbano.json --unidad km/h
  python parte4_velocidad_zonas.py --fuente datos/peatones_vtest.avi --persona --filas 2 --cols 3

Teclas: q/ESC salir | p pausa | h mapa de calor | z zonas | v vista de pajaro (si hay homografia)
"""
import argparse
import json
import os
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import pandas as pd

from comun import (TrackerYOLO, RegistroTrayectorias, Calibracion, Velocimetro, MapaCalor, Zonas, SalidaVideo,
                   abrir_fuente, Reloj, dibujar_objeto, dibujar_estela, color_id, panel, tecla, aplicar_H)


def color_vel(v, vmax):
    """verde (lento) -> amarillo -> rojo (rapido)"""
    r = float(np.clip(v / vmax, 0, 1))
    return (0, int(255 * (1 - r) + 60 * r), int(255 * min(1, 2 * r)))


class VistaPajaro:
    """Vista cenital (desde arriba) usando la homografia: aqui las distancias en pixeles SI son metros."""

    def __init__(self, H, w, h, alto_px=360):
        ys, xs = np.mgrid[int(h * 0.3):h:8, 0:w:8]
        pts = np.float32(np.column_stack([xs.ravel(), ys.ravel()])).reshape(-1, 1, 2)
        m = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        self.x0, self.x1 = np.percentile(m[:, 0], [3, 97])
        self.y0, self.y1 = np.percentile(m[:, 1], [2, 98])
        self.esc = alto_px / (self.y1 - self.y0)                     # px de la vista por metro
        self.size = (int((self.x1 - self.x0) * self.esc), alto_px)
        S = np.array([[self.esc, 0, -self.x0 * self.esc], [0, -self.esc, self.y1 * self.esc], [0, 0, 1]])
        self.M = S @ H

    def a_vista(self, X, Y):
        return int((X - self.x0) * self.esc), int((self.y1 - Y) * self.esc)

    def render(self, frame, objetos_m, unidad):
        v = cv2.warpPerspective(frame, self.M, self.size)
        for m in range(int(np.ceil(self.y0 / 5) * 5), int(self.y1), 5):
            y = self.a_vista(0, m)[1]
            cv2.line(v, (0, y), (8, y), (255, 255, 255), 1)
            cv2.putText(v, f"{m} m", (10, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        for tid, X, Y, vel in objetos_m:
            p = self.a_vista(X, Y)
            cv2.circle(v, p, 6, color_id(tid), -1, cv2.LINE_AA)
            if vel is not None:
                cv2.putText(v, f"{vel:.0f}", (p[0] + 7, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(v, (0, 0), (v.shape[1] - 1, v.shape[0] - 1), (255, 255, 255), 1)
        cv2.putText(v, "vista de pajaro", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fuente", default="0")
    ap.add_argument("--modelo", default="yolo11n.pt")
    ap.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "botsort"])
    ap.add_argument("--clases", default="person")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=640)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--ppm", type=float, help="pixeles por metro")
    g.add_argument("--calibracion", help="JSON de calibrar.py (ppm u homografia)")
    g.add_argument("--persona", action="store_true", help="escala con la altura media de una persona (1.70 m)")
    ap.add_argument("--unidad", default="m/s", choices=["m/s", "km/h"])
    ap.add_argument("--ventana", type=float, default=1.0, help="segundos usados para estimar la velocidad")
    ap.add_argument("--filas", type=int, default=3)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--zonas", help="JSON con poligonos {'zonas':[{'nombre':..,'puntos':[[x,y],..]}]}")
    ap.add_argument("--salida", default=None)
    ap.add_argument("--guardar-video", action="store_true")
    ap.add_argument("--no-mostrar", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    if a.calibracion:
        calib = Calibracion.cargar(a.calibracion)
    elif a.ppm:
        calib = Calibracion("ppm", ppm=a.ppm)
    elif a.persona:
        calib = Calibracion("persona")
    else:
        calib = Calibracion("px")
    f_unidad = 3.6 if a.unidad == "km/h" else 1.0
    unidad = a.unidad if calib.metrica else "px/s"
    vmax = 60 if a.unidad == "km/h" else 3.0

    nombre = "webcam" if a.fuente.isdigit() else os.path.splitext(os.path.basename(a.fuente))[0]
    salida = a.salida or os.path.join("salidas", "parte4", f"{nombre}_{calib.tipo}")
    os.makedirs(salida, exist_ok=True)

    cap, fps, es_webcam = abrir_fuente(a.fuente)
    reloj = Reloj(fps, es_webcam)
    tracker = TrackerYOLO(a.modelo, a.tracker, a.clases.split(","), a.conf, a.imgsz)
    registro = RegistroTrayectorias(os.path.join(salida, "velocidades.csv"),
                                    extra=["px_suelo", "py_suelo", "X_m", "Y_m", "vel_px_s", "vel", "unidad", "zona"])
    vel_px, vel_m = Velocimetro(a.ventana), Velocimetro(a.ventana)
    video = SalidaVideo(os.path.join(salida, "video_resultado.mp4"), fps) if a.guardar_video else None

    mostrar = not a.no_mostrar
    ventana = "Parte 4 - Velocidad y zonas"
    if mostrar:
        cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    ver_calor, ver_zonas, ver_pajaro = False, True, True
    estelas = defaultdict(lambda: deque(maxlen=30))
    calor = zonas = pajaro = None
    frame_idx, pausa, t_proc = 0, False, deque(maxlen=30)
    fondo_acum = []
    vis = None
    while True:
        if not pausa:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if calor is None:
                calor = MapaCalor(w, h, sigma=max(6, w // 80))
                zonas = Zonas(w, h, a.filas, a.cols, a.zonas)
                if calib.tipo == "homografia":
                    pajaro = VistaPajaro(calib.H, w, h)
            t = reloj.t(frame_idx)
            t0 = time.time()
            objetos = tracker.procesar(frame)
            t_proc.append(time.time() - t0)

            vis = frame.copy()
            en_metros = []
            for o in objetos:
                tid = o["id"]
                pie = o["pie"]                                   # el punto que toca el suelo
                vpx = vel_px.actualizar(tid, t, *o["centro"])    # px/s (sin calibrar)
                X, Y = calib.a_mundo(pie, caja=o["caja"])
                vm = vel_m.actualizar(tid, t, X, Y) if calib.metrica else vpx
                v_mostrar = vm * f_unidad if (vm is not None and calib.metrica) else vm
                z = zonas.actualizar(tid, pie, v_mostrar)
                calor.agregar(pie)
                estelas[tid].append(pie)
                registro.guardar(frame_idx, t, o, px_suelo=pie[0], py_suelo=pie[1],
                                 X_m=X if calib.metrica else "", Y_m=Y if calib.metrica else "",
                                 vel_px_s=vpx if vpx is not None else "", vel=v_mostrar if v_mostrar is not None else "",
                                 unidad=unidad, zona=z or "")
                if calib.metrica:
                    en_metros.append((tid, X, Y, v_mostrar))
                col = color_vel(v_mostrar, vmax if calib.metrica else 150) if v_mostrar is not None else (200, 200, 200)
                txt = f"#{tid} " + (f"{v_mostrar:.1f} {unidad}" if v_mostrar is not None else "...")
                if calib.metrica and vpx is not None:
                    txt += f" ({vpx:.0f} px/s)"
                dibujar_estela(vis, list(estelas[tid]), color_id(tid))
                dibujar_objeto(vis, o, txt, col)

            base = vis
            if ver_calor:
                base = calor.colorear(vis)
            if ver_zonas:
                base = zonas.dibujar(base, unidad)
            vis = base
            if pajaro is not None and ver_pajaro:
                pv = pajaro.render(frame, en_metros, unidad)
                ph = int(0.62 * h)
                pv = cv2.resize(pv, (int(pv.shape[1] * ph / pv.shape[0]), ph))
                vis[10:10 + ph, w - pv.shape[1] - 10:w - 10] = pv
            panel(vis, [f"Calibracion: {calib.tipo}  | {1 / max(np.mean(t_proc), 1e-3):.1f} FPS",
                        f"Objetos: {len(objetos)}  | unidad: {unidad}",
                        "h: calor  z: zonas  v: vista pajaro"])
            if video:
                video.escribir(vis)
            if frame_idx % max(1, int(fps)) == 0 and len(fondo_acum) < 40:
                fondo_acum.append(frame)
            frame_idx += 1
            if a.max_frames and frame_idx >= a.max_frames:
                break
        if mostrar:
            cv2.imshow(ventana, vis)
        k = tecla(mostrar, 1 if not pausa else 30)
        if k in (ord("q"), 27):
            break
        if k == ord("p"):
            pausa = not pausa
        if k == ord("h"):
            ver_calor = not ver_calor
        if k == ord("z"):
            ver_zonas = not ver_zonas
        if k == ord("v"):
            ver_pajaro = not ver_pajaro

    cap.release(); registro.cerrar()
    if video:
        video.cerrar()
    if mostrar:
        cv2.destroyAllWindows()

    # ---------------------------------------------------------------- resultados finales
    fondo = np.median(np.stack(fondo_acum), axis=0).astype(np.uint8) if fondo_acum else np.zeros((h, w, 3), np.uint8)
    cv2.imwrite(os.path.join(salida, "fondo.jpg"), fondo)
    cv2.imwrite(os.path.join(salida, "mapa_calor.png"), calor.colorear(fondo, 0.75))
    cv2.imwrite(os.path.join(salida, "zonas.png"), zonas.dibujar(fondo, unidad))
    pd.DataFrame(zonas.tabla(fps)).to_csv(os.path.join(salida, "zonas.csv"), index=False)

    df = pd.read_csv(os.path.join(salida, "velocidades.csv"))
    df["vel"] = pd.to_numeric(df["vel"], errors="coerce")
    df["vel_px_s"] = pd.to_numeric(df["vel_px_s"], errors="coerce")
    por_id = df.dropna(subset=["vel"]).groupby("id").agg(clase=("clase", "first"), muestras=("vel", "size"),
                                                          vel_mediana=("vel", "median"), vel_max=("vel", "max"),
                                                          vel_px_s_mediana=("vel_px_s", "median"))
    por_id = por_id[por_id["muestras"] >= 8].round(2)
    por_id.to_csv(os.path.join(salida, "velocidad_por_id.csv"))
    resumen = {"fuente": a.fuente, "calibracion": calib.tipo, "unidad": unidad, "objetos_con_velocidad": int(len(por_id)),
               "vel_mediana_global": round(float(por_id["vel_mediana"].median()), 2) if len(por_id) else None,
               "zonas": zonas.tabla(fps)}
    json.dump(resumen, open(os.path.join(salida, "resumen.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"Resultados en: {salida}")


if __name__ == "__main__":
    main()
