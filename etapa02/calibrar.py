"""
calibrar.py — Factor de escala (px/m) u homografía imagen -> suelo
=================================================================
Paso previo del Nivel 1. Se mide una distancia REAL en la calle (Google Maps:
clic derecho > "Medir distancia"; p. ej. entre dos postes o el ancho de un
paso peatonal) y se marca la misma distancia en la imagen.

Modo ppm (lo que pide el enunciado):
    python calibrar.py --fuente 0 --modo ppm --distancia 12.5 --salida calib/mi_calle.json
    -> clic en 2 puntos; ppm = distancia_px / distancia_m

Modo homografía (recomendado: corrige la perspectiva):
    python calibrar.py --fuente video.mp4 --modo homografia --ancho 7 --largo 12 --salida calib/h.json
    -> clic en 4 esquinas de un rectángulo del suelo, en orden:
       cerca-izq, cerca-der, lejos-izq, lejos-der

Sin interfaz gráfica (puntos ya conocidos):
    python calibrar.py --modo ppm --puntos "310,277;310,219" --distancia 6 --salida calib/x.json
"""
import argparse
import json
import os

import cv2
import numpy as np


def pedir_clics(frame, n, titulo):
    pts = []
    vis = frame.copy()

    def cb(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < n:
            pts.append((x, y))
            cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(vis, str(len(pts)), (x + 6, y - 6), 0, 0.6, (0, 0, 255), 2)
            if len(pts) > 1:
                cv2.line(vis, pts[-2], pts[-1], (0, 255, 255), 1)

    cv2.namedWindow(titulo)
    cv2.setMouseCallback(titulo, cb)
    while len(pts) < n:
        cv2.imshow(titulo, vis)
        if (cv2.waitKey(20) & 0xFF) in (27, ord("q")):
            raise SystemExit("Cancelado")
    cv2.imshow(titulo, vis)
    cv2.waitKey(600)
    cv2.destroyWindow(titulo)
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fuente", default="0")
    ap.add_argument("--frame", type=int, default=0, help="frame del video a usar")
    ap.add_argument("--modo", choices=["ppm", "homografia"], default="ppm")
    ap.add_argument("--distancia", type=float, help="distancia real (m) entre los 2 puntos (modo ppm)")
    ap.add_argument("--ancho", type=float, help="ancho real (m) del rectángulo (modo homografía)")
    ap.add_argument("--largo", type=float, help="largo real (m) del rectángulo (modo homografía)")
    ap.add_argument("--puntos", default=None, help="'x,y;x,y;...' para no usar el mouse")
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()

    n = 2 if a.modo == "ppm" else 4
    frame = None
    if a.puntos:
        pts = [tuple(map(float, p.split(","))) for p in a.puntos.split(";")]
    else:
        cap = cv2.VideoCapture(int(a.fuente) if a.fuente.isdigit() else a.fuente)
        cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("No se pudo leer la fuente")
        pts = pedir_clics(frame, n, f"Marque {n} puntos (ESC cancela)")
    if len(pts) != n:
        raise SystemExit(f"Se requieren {n} puntos")

    if a.modo == "ppm":
        d = a.distancia or float(input("Distancia real entre los puntos (m): "))
        dpx = float(np.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]))
        datos = {"tipo": "ppm", "ppm": round(dpx / d, 4), "puntos_imagen": pts, "distancia_m": d,
                 "distancia_px": round(dpx, 2),
                 "nota": "El factor solo es válido cerca de estos puntos y en esa dirección."}
        print(f"Distancia en imagen: {dpx:.1f} px  ->  {dpx / d:.3f} px/m")
    else:
        an = a.ancho or float(input("Ancho real del rectángulo (m): "))
        la = a.largo or float(input("Largo real del rectángulo (m): "))
        mundo = [[0, 0], [an, 0], [0, la], [an, la]]
        datos = {"tipo": "homografia", "puntos_imagen": pts, "puntos_mundo_m": mundo,
                 "nota": "Orden: cerca-izq, cerca-der, lejos-izq, lejos-der. X lateral, Y a lo largo de la vía."}
        H, _ = cv2.findHomography(np.float32(pts), np.float32(mundo))
        print("Homografía imagen->suelo:\n", np.round(H, 5))
    os.makedirs(os.path.dirname(os.path.abspath(a.salida)), exist_ok=True)
    with open(a.salida, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)
    print("Guardado en", a.salida)
    if frame is not None and a.modo == "homografia":
        vista = cv2.warpPerspective(frame, np.array([[20, 0, 0], [0, -20, 20 * (la + 20)], [0, 0, 1]]) @ H,
                                    (int(20 * (an + 4)), int(20 * (la + 22))))
        cv2.imshow("Vista de pajaro (20 px/m) - cualquier tecla", vista)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
