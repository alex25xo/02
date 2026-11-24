"""
PARTE 2 - Tracking 2D con YOLO + ByteTrack / BoT-SORT, conteo con linea virtual y registro de trayectorias
=========================================================================================================
Ejemplos:
  python parte2_tracking_conteo.py --fuente 0                                   (camara web)
  python parte2_tracking_conteo.py --fuente datos/peatones_vtest.avi --linea 0.55,0.25,0.55,0.95
  python parte2_tracking_conteo.py --fuente datos/trafico_urbano.mp4 --clases car,truck,bus,motorcycle --linea 0,0.62,1,0.62 --tracker botsort

Teclas: q/ESC salir | p pausa | l dibujar una nueva linea con 2 clics | r reiniciar conteo
Salidas (carpeta --salida):
  trayectorias.csv     frame, t_s, id, clase, conf, cx, cy, x1, y1, x2, y2   <- para la Parte 3
  conteo_eventos.csv   frame, t_s, id, clase, sentido                         <- cada cruce de la linea
  resumen.json         totales por clase y sentido
  fondo.jpg            imagen de fondo (mediana) para dibujar las trayectorias
"""
import argparse
import csv
import json
import os
import time
from collections import defaultdict, deque

import cv2
import numpy as np

from comun import (TrackerYOLO, LineaConteo, SelectorLinea, RegistroTrayectorias, SalidaVideo, abrir_fuente, Reloj,
                   parse_linea, dibujar_objeto, dibujar_estela, color_id, panel, tecla)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fuente", default="0", help="0 = camara web, o ruta de un video")
    ap.add_argument("--modelo", default="yolo11n.pt")
    ap.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "botsort"])
    ap.add_argument("--clases", default="person", help="clases COCO separadas por coma (person,car,bus,...)")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--linea", default="0.5,0.1,0.5,0.9", help="x1,y1,x2,y2 en fracciones de la imagen")
    ap.add_argument("--punto", default="centro", choices=["centro", "pie"], help="punto del objeto que cruza la linea")
    ap.add_argument("--estela", type=int, default=40, help="frames de estela dibujada")
    ap.add_argument("--salida", default=None)
    ap.add_argument("--guardar-video", action="store_true")
    ap.add_argument("--no-mostrar", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    nombre = "webcam" if a.fuente.isdigit() else os.path.splitext(os.path.basename(a.fuente))[0]
    salida = a.salida or os.path.join("salidas", "parte2", f"{nombre}_{a.tracker}")
    os.makedirs(salida, exist_ok=True)

    cap, fps, es_webcam = abrir_fuente(a.fuente)
    reloj = Reloj(fps, es_webcam)
    tracker = TrackerYOLO(a.modelo, a.tracker, a.clases.split(","), a.conf, a.imgsz)
    registro = RegistroTrayectorias(os.path.join(salida, "trayectorias.csv"))
    video = SalidaVideo(os.path.join(salida, "video_resultado.mp4"), fps) if a.guardar_video else None

    mostrar = not a.no_mostrar
    ventana = "Parte 2 - Tracking + conteo"
    if mostrar:
        cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
        selector = SelectorLinea(ventana)

    linea = None
    estelas = defaultdict(lambda: deque(maxlen=a.estela))
    ids_vistos = set()
    muestras_fondo = []
    frame_idx, pausa, t_proc = 0, False, deque(maxlen=30)
    vis = None
    while True:
        if not pausa:
            ok, frame = cap.read()
            if not ok:
                break
            if linea is None:
                h, w = frame.shape[:2]
                linea = LineaConteo(*parse_linea(a.linea, w, h))
            t = reloj.t(frame_idx)
            t0 = time.time()
            objetos = tracker.procesar(frame)
            t_proc.append(time.time() - t0)

            vis = frame.copy()
            for o in objetos:
                p = o["centro"] if a.punto == "centro" else o["pie"]
                linea.actualizar(o["id"], p, o["clase"], frame_idx, t)
                estelas[o["id"]].append(p)
                ids_vistos.add(o["id"])
                registro.guardar(frame_idx, t, o)
                dibujar_estela(vis, list(estelas[o["id"]]), color_id(o["id"]))
                dibujar_objeto(vis, o)
                cv2.circle(vis, tuple(map(int, p)), 4, (255, 255, 255), -1, cv2.LINE_AA)
            linea.dibujar(vis)
            ab, ba = linea.totales()
            lineas = [f"{a.tracker.upper()} | {1 / max(np.mean(t_proc), 1e-3):.1f} FPS | frame {frame_idx}",
                      f"Activos: {len(objetos)} | IDs totales: {len(ids_vistos)}",
                      f"CONTEO: {linea.total}   (A->B {ab} | B->A {ba})"]
            for c, v in linea.por_clase.items():
                lineas.append(f"  {c}: {v['A->B'] + v['B->A']}")
            panel(vis, lineas, escala=0.55 if w >= 900 else 0.45)
            if video:
                video.escribir(vis)
            if frame_idx % max(1, int(fps)) == 0 and len(muestras_fondo) < 40:
                muestras_fondo.append(frame)
            frame_idx += 1
            if a.max_frames and frame_idx >= a.max_frames:
                break

        if mostrar:
            if selector.activo:
                aviso = vis.copy()
                panel(aviso, [f"Haga 2 clics para la nueva linea ({len(selector.puntos)}/2)"], org=(10, aviso.shape[0] - 50))
                cv2.imshow(ventana, aviso)
            else:
                cv2.imshow(ventana, vis)
            nueva = selector.nueva_linea()
            if nueva:
                linea = LineaConteo(*nueva)
        k = tecla(mostrar, 1 if not pausa else 30)
        if k in (ord("q"), 27):
            break
        if k == ord("p"):
            pausa = not pausa
        if k == ord("l") and mostrar:
            selector.iniciar()
        if k == ord("r") and linea is not None:
            linea = LineaConteo(linea.p1, linea.p2)

    cap.release()
    registro.cerrar()
    if video:
        video.cerrar()
    if mostrar:
        cv2.destroyAllWindows()

    # ---------------- salidas para el analisis (Parte 3)
    if muestras_fondo:
        cv2.imwrite(os.path.join(salida, "fondo.jpg"), np.median(np.stack(muestras_fondo), axis=0).astype(np.uint8))
    with open(os.path.join(salida, "conteo_eventos.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_s", "id", "clase", "sentido"])
        w.writerows(linea.eventos if linea else [])
    ab, ba = linea.totales() if linea else (0, 0)
    resumen = {"fuente": a.fuente, "tracker": a.tracker, "frames": frame_idx, "fps": fps,
               "ids_distintos": len(ids_vistos), "filas_trayectorias": registro.n,
               "linea_px": [linea.p1, linea.p2] if linea else None,
               "conteo_total": linea.total if linea else 0, "A->B": ab, "B->A": ba,
               "por_clase": {k: dict(v) for k, v in (linea.por_clase.items() if linea else [])}}
    json.dump(resumen, open(os.path.join(salida, "resumen.json"), "w", encoding="utf-8"), indent=2)
    print(json.dumps(resumen, indent=2))
    print(f"\nResultados en: {salida}")


if __name__ == "__main__":
    main()
