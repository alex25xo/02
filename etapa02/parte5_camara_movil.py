"""
PARTE 5 - Tracking con la CAMARA EN MOVIMIENTO: compensacion con odometria visual
================================================================================
Problema: si la camara se mueve, cada pixel se desplaza aunque el objeto este quieto:
    movimiento observado = movimiento del objeto + movimiento de la camara
  -> la linea virtual "viaja" con la imagen (cuenta mal), las velocidades se inflan y el mapa de calor se borronea.
Solucion (odometria visual 2D):
  1. Se estima el movimiento de la camara con puntos del FONDO (se excluyen las cajas de YOLO).
  2. RANSAC -> homografia H_k entre frames; se acumula G_k (frame 0 -> frame k).
  3. Cada posicion se lleva al sistema del frame 0 (anclado al mundo):  p_ref = inv(G_k) * p
  4. Linea, velocidad (con la calibracion del frame 0) y mapa de calor se calculan en ese sistema fijo.

Ejemplos:
  python parte5_camara_movil.py --fuente datos/trafico_urbano_pan.mp4 --clases car,truck,bus --calibracion calib/trafico_urbano_pan.json --unidad km/h --linea 0,0.6,1,0.6 --gt datos/trafico_urbano_pan_gt.json
  python parte5_camara_movil.py --fuente 0 --persona          (mueva la camara web mientras alguien camina)

Teclas: q/ESC salir | p pausa | r reiniciar la referencia (frame actual = nuevo frame 0)
"""
import argparse
import json
import os
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import pandas as pd

from comun import (TrackerYOLO, LineaConteo, Calibracion, Velocimetro, MapaCalor, OdometriaVisual2D, SalidaVideo,
                   RegistroTrayectorias, abrir_fuente, Reloj, parse_linea, dibujar_objeto, dibujar_estela, color_id,
                   panel, etiqueta, tecla, aplicar_H)


def transformar(H, pts):
    if len(pts) == 0:
        return []
    p = cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), H).reshape(-1, 2)
    return [tuple(x) for x in p]


def cinta(img, texto, color):
    cv2.rectangle(img, (0, img.shape[0] - 30), (img.shape[1], img.shape[0]), color, -1)
    cv2.putText(img, texto, (10, img.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fuente", default="0")
    ap.add_argument("--modelo", default="yolo11n.pt")
    ap.add_argument("--tracker", default="botsort", choices=["bytetrack", "botsort"])
    ap.add_argument("--clases", default="person")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--linea", default="0.5,0.1,0.5,0.9", help="linea en el frame 0 (fracciones)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--calibracion", help="JSON de calibrar.py hecho sobre el frame 0")
    g.add_argument("--ppm", type=float)
    g.add_argument("--persona", action="store_true")
    ap.add_argument("--unidad", default="m/s", choices=["m/s", "km/h"])
    ap.add_argument("--modelo-vo", default="homografia", choices=["homografia", "afin"])
    ap.add_argument("--gt", help="(opcional) JSON con el movimiento real de la camara para evaluar")
    ap.add_argument("--salida", default=None)
    ap.add_argument("--guardar-video", action="store_true")
    ap.add_argument("--no-mostrar", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    calib = (Calibracion.cargar(a.calibracion) if a.calibracion else Calibracion("ppm", ppm=a.ppm) if a.ppm
             else Calibracion("persona") if a.persona else Calibracion("px"))
    f_u = 3.6 if (a.unidad == "km/h" and calib.metrica) else 1.0
    unidad = a.unidad if calib.metrica else "px/s"
    nombre = "webcam" if a.fuente.isdigit() else os.path.splitext(os.path.basename(a.fuente))[0]
    salida = a.salida or os.path.join("salidas", "parte5", nombre)
    os.makedirs(salida, exist_ok=True)

    gt = None
    if a.gt:
        d = json.load(open(a.gt, encoding="utf-8"))
        gt = [np.array(A) for A in d["transformaciones"]]       # A_k: video sin mover -> frame k

    cap, fps, es_webcam = abrir_fuente(a.fuente)
    reloj = Reloj(fps, es_webcam)
    tracker = TrackerYOLO(a.modelo, a.tracker, a.clases.split(","), a.conf, a.imgsz)
    vo = OdometriaVisual2D(modelo=a.modelo_vo)
    v_sin, v_con, v_gt = Velocimetro(1.0), Velocimetro(1.0), Velocimetro(1.0)
    registro = RegistroTrayectorias(os.path.join(salida, "trayectorias.csv"),
                                    extra=["x_ref", "y_ref", "vel_sin_vo", "vel_con_vo", "vel_real", "unidad"])
    video = SalidaVideo(os.path.join(salida, "sin_vs_con_vo.mp4"), fps) if a.guardar_video else None
    mostrar = not a.no_mostrar
    if mostrar:
        cv2.namedWindow("Parte 5 - camara en movimiento", cv2.WINDOW_NORMAL)

    linea_sin = linea_con = None
    est_img = defaultdict(lambda: deque(maxlen=40))
    est_ref = defaultdict(lambda: deque(maxlen=40))
    tray_img, tray_ref = defaultdict(list), defaultdict(list)
    errores_vo = []
    frame_idx, pausa = 0, False
    lado = None
    calor = None
    while True:
        if not pausa:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if linea_sin is None:
                p1, p2 = parse_linea(a.linea, w, h)
                linea_sin, linea_con = LineaConteo(p1, p2), LineaConteo(p1, p2)
                calor = MapaCalor(2 * w, 2 * h, sigma=max(6, w // 80))      # lienzo del mundo con margen
                off = np.array([[1, 0, w / 2], [0, 1, h / 2], [0, 0, 1.0]])
            t = reloj.t(frame_idx)
            objetos = tracker.procesar(frame)
            G = vo.procesar(frame, [o["caja"] for o in objetos])             # frame 0 -> frame actual
            A = vo.a_referencia                                              # frame actual -> frame 0
            if gt is not None and frame_idx < len(gt):
                G_real = gt[frame_idx] @ np.linalg.inv(gt[0])
                esq = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                errores_vo.append(float(np.mean(np.linalg.norm(
                    cv2.perspectiveTransform(esq, G) - cv2.perspectiveTransform(esq, G_real), axis=2))))

            izq, der = frame.copy(), frame.copy()
            for o in objetos:
                tid = o["id"]
                p_img = o["pie"] if calib.tipo == "homografia" else o["centro"]
                p_ref = aplicar_H(A, p_img)
                # ---- SIN VO: todo en coordenadas de la imagen actual
                linea_sin.actualizar(tid, p_img, o["clase"], frame_idx, t)
                Xs, Ys = calib.a_mundo(p_img, caja=o["caja"])
                vs = v_sin.actualizar(tid, t, Xs, Ys)
                # ---- CON VO: todo en el sistema del frame 0 (fijo al mundo)
                linea_con.actualizar(tid, p_ref, o["clase"], frame_idx, t)
                Xc, Yc = calib.a_mundo(p_ref, caja=o["caja"])
                vc = v_con.actualizar(tid, t, Xc, Yc)
                calor.agregar(aplicar_H(off, p_ref))
                vr = None
                if gt is not None and frame_idx < len(gt):                   # referencia: coordenadas sin movimiento
                    p_orig = aplicar_H(np.linalg.inv(gt[frame_idx]), p_img)
                    Xr, Yr = calib.a_mundo(aplicar_H(gt[0], p_orig), caja=o["caja"])
                    vr = v_gt.actualizar(tid, t, Xr, Yr)
                est_img[tid].append(p_img); est_ref[tid].append(p_ref)
                tray_img[tid].append(p_img); tray_ref[tid].append(p_ref)
                registro.guardar(frame_idx, t, o, x_ref=p_ref[0], y_ref=p_ref[1],
                                 vel_sin_vo=vs * f_u if vs is not None else "", vel_con_vo=vc * f_u if vc is not None else "",
                                 vel_real=vr * f_u if vr is not None else "", unidad=unidad)
                col = color_id(tid)
                dibujar_estela(izq, list(est_img[tid]), col)
                dibujar_objeto(izq, o, f"#{tid} " + (f"{vs * f_u:.0f} {unidad}" if vs is not None else "..."), col)
                dibujar_estela(der, transformar(G, list(est_ref[tid])), col)     # la estela queda pegada al suelo
                dibujar_objeto(der, o, f"#{tid} " + (f"{vc * f_u:.0f} {unidad}" if vc is not None else "..."), col)

            # puntos usados por la VO (verde = fondo/inlier, rojo = descartado por RANSAC)
            info = vo.info
            if "a" in info:
                for (x, y), ok_ in zip(info["b"], info["inl"]):
                    cv2.circle(der, (int(x), int(y)), 2, (60, 220, 60) if ok_ else (40, 40, 230), -1)
            linea_sin.dibujar(izq)
            q1, q2 = transformar(G, [linea_con.p1, linea_con.p2])
            linea_con.dibujar(der, q1, q2)
            panel(izq, [f"Conteo: {linea_sin.total}", "linea fija en la PANTALLA"], escala=0.5)
            panel(der, [f"Conteo: {linea_con.total}", "linea fija en el MUNDO",
                        f"VO: {info['inliers']}/{info['puntos']} inliers | {info['flujo_px']:.1f} px/frame"], escala=0.5)
            cinta(izq, "SIN odometria visual", (40, 40, 200))
            cinta(der, "CON odometria visual (compensada)", (170, 100, 30))
            vis = np.hstack([izq, np.full((h, 6, 3), 255, np.uint8), der])
            if video:
                video.escribir(vis)
            frame_idx += 1
            if a.max_frames and frame_idx >= a.max_frames:
                break
        if mostrar:
            cv2.imshow("Parte 5 - camara en movimiento", vis)
        k = tecla(mostrar, 1 if not pausa else 30)
        if k in (ord("q"), 27):
            break
        if k == ord("p"):
            pausa = not pausa
        if k == ord("r"):                                   # el frame actual pasa a ser la nueva referencia
            q1, q2 = transformar(vo.G, [linea_con.p1, linea_con.p2])
            linea_con = LineaConteo(q1, q2)
            vo = OdometriaVisual2D(modelo=a.modelo_vo)
            est_ref.clear()

    cap.release(); registro.cerrar()
    if video:
        video.cerrar()
    if mostrar:
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------ figuras finales
    cap, _, _ = abrir_fuente(a.fuente) if not es_webcam else (None, 0, 0)
    ref = None
    if cap is not None:
        ok, ref = cap.read(); cap.release()
    if ref is not None:
        for nombre_f, tray, titulo in (("trayectorias_sin_vo.png", tray_img, "SIN VO: coordenadas de pantalla"),
                                       ("trayectorias_con_vo.png", tray_ref, "CON VO: sistema del frame 0 (mundo)")):
            lienzo = cv2.copyMakeBorder(ref, h // 2, h // 2, w // 2, w // 2, cv2.BORDER_CONSTANT, value=(40, 40, 40))
            for tid, pts in tray.items():
                if len(pts) > 5:
                    dibujar_estela(lienzo, [(x + w / 2, y + h / 2) for x, y in pts], color_id(tid), 2)
            cam = [(x + w / 2, y + h / 2) for x, y in vo.tray]
            etiqueta(lienzo, titulo, (12, 28), (30, 30, 30), escala=0.7, grosor=2)
            cv2.imwrite(os.path.join(salida, nombre_f), lienzo)
        lienzo = cv2.copyMakeBorder(ref, h // 2, h // 2, w // 2, w // 2, cv2.BORDER_CONSTANT, value=(40, 40, 40))
        cv2.imwrite(os.path.join(salida, "mapa_calor_mundo.png"), calor.colorear(lienzo))

    df = pd.read_csv(os.path.join(salida, "trayectorias.csv"))
    res = {"fuente": a.fuente, "frames": frame_idx, "conteo_sin_vo": linea_sin.total, "conteo_con_vo": linea_con.total,
           "desplazamiento_camara_px": [round(float(x), 1) for x in np.subtract(vo.tray[-1], vo.tray[0])]}
    for c in ("vel_sin_vo", "vel_con_vo", "vel_real"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df["vel_real"].notna().any():
        m = df.dropna(subset=["vel_sin_vo", "vel_con_vo", "vel_real"])
        res["error_vel_sin_vo"] = round(float((m["vel_sin_vo"] - m["vel_real"]).abs().mean()), 2)
        res["error_vel_con_vo"] = round(float((m["vel_con_vo"] - m["vel_real"]).abs().mean()), 2)
        res["unidad"] = unidad
    if errores_vo:
        res["error_vo_px_medio"] = round(float(np.mean(errores_vo)), 2)
        res["error_vo_px_final"] = round(float(errores_vo[-1]), 2)
        np.savetxt(os.path.join(salida, "error_vo_por_frame.csv"), np.array(errores_vo), fmt="%.3f", header="error_px")
    json.dump(res, open(os.path.join(salida, "resumen.json"), "w", encoding="utf-8"), indent=2)
    print(json.dumps(res, indent=2))
    print(f"Resultados en: {salida}")


if __name__ == "__main__":
    main()
