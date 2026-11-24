"""
NIVEL 3 - ODOMETRIA VISUAL (camara en movimiento)
Monitor Inteligente de Trafico y Navegacion

Muestra lado a lado:
  IZQUIERDA  "ANTES":   el MISMO radar del Nivel 1 aplicado a una camara que se mueve.
                        Cree que el suelo y los autos estacionados se mueven.
  DERECHA    "DESPUES": odometria visual (FAST/ORB + BRIEF + RANSAC) estima el
                        movimiento de la camara y lo resta: el suelo queda quieto,
                        los autos estacionados marcan ~0 km/h y los que circulan,
                        su velocidad real. Ademas: velocidad y trayectoria de la camara.

Antes, calibra la camara con un cuadro de ESTE video (el celular a altura/angulo fijos):
    python calibrar.py --video caminata.mp4 --modo homografia --salida calib_movil.json

Uso:
    python odometria_visual.py --video caminata.mp4 --calibracion calib_movil.json
    python odometria_visual.py --video pasillo.mp4 --calibracion calib_pasillo.json --sin-yolo
    python odometria_visual.py --video 0 --calibracion calib_webcam.json      # webcam en vivo

Teclas: ESPACIO pausa | Q / ESC salir (los reportes se generan igual)
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from radar.calibracion import Calibracion
from radar.dibujo import (ajustar_para_pantalla, dibujar_panel, dibujar_vehiculo, dibujar_zona,
                          etiqueta, ventana_cerrada)
from radar.odometria import OdometriaVisual, aplicar
from radar.velocidad import ControlLimite, EstimadorVelocidad
from radar.video import abrir, fps_confiable, info_video, resolver_fuente

CLASES_VEHICULO = {2: "auto", 3: "moto", 5: "bus", 7: "camion"}
CLASE_PERSONA = 0
VENTANA = "Nivel 3 - Odometria visual (antes vs despues)"


def parse_args():
    ap = argparse.ArgumentParser(description="Odometria visual: radar con camara en movimiento")
    ap.add_argument("--video", help="Ruta del video o indice de webcam. Sin valor: abre explorador")
    ap.add_argument("--calibracion", default="calib_movil.json",
                    help="Calibracion hecha con calibrar.py sobre un cuadro de este video")
    ap.add_argument("--limite", type=float, default=50.0, help="Limite de velocidad (km/h)")
    ap.add_argument("--umbral-estacionado", type=float, default=3.0,
                    help="Por debajo de esta velocidad real (km/h) el vehiculo esta quieto")
    ap.add_argument("--modelo", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--tracker", default="botsort.yaml",
                    help="botsort.yaml (compensa el movimiento de camara) o bytetrack.yaml")
    ap.add_argument("--dispositivo", default=None)
    ap.add_argument("--sin-yolo", action="store_true",
                    help="Solo odometria (ej. pasillo sin vehiculos): mucho mas rapido")
    ap.add_argument("--orb-puntos", type=int, default=1500, help="Puntos ORB por cuadro")
    ap.add_argument("--ransac-m", type=float, default=0.12,
                    help="Tolerancia de RANSAC en metros sobre el suelo")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--salto", type=int, default=1, help="Procesar 1 de cada N cuadros")
    ap.add_argument("--vista", choices=["comparar", "compensado"], default="comparar",
                    help="comparar = antes|despues lado a lado; compensado = solo el resultado")
    ap.add_argument("--salida", default="salidas")
    ap.add_argument("--no-mostrar", action="store_true")
    ap.add_argument("--sin-video-salida", action="store_true")
    ap.add_argument("--max-segundos", type=float, default=None)
    return ap.parse_args()


# ---------------------------------------------------------------------- #
# Dibujo propio del Nivel 3
# ---------------------------------------------------------------------- #
def dibujar_flujo_aparente(img, r, dt, max_flechas=160):
    """Flechas naranjas: cómo 'se mueve' el suelo en la imagen (movimiento de 0.25 s)."""
    if len(r.px_actual) == 0:
        return
    idx = np.flatnonzero(r.inliers) if r.inliers.any() else np.arange(len(r.px_actual))
    if len(idx) > max_flechas:
        idx = idx[np.linspace(0, len(idx) - 1, max_flechas).astype(int)]
    k = 0.25 / max(dt, 1e-3)
    for i in idx:
        a, b = r.px_prev[i], r.px_actual[i]
        ini = b - (b - a) * k
        cv2.arrowedLine(img, tuple(int(v) for v in ini), tuple(int(v) for v in b), (0, 140, 255),
                        1, cv2.LINE_AA, tipLength=0.3)


def dibujar_puntos_compensados(img, r, max_puntos=300):
    """Puntos del suelo usados: verdes = coinciden con el movimiento de la cámara (quietos)."""
    n = len(r.px_actual)
    if n == 0:
        return
    paso = max(1, n // max_puntos)
    for i in range(0, n, paso):
        p = tuple(int(v) for v in r.px_actual[i])
        if r.inliers[i]:
            cv2.circle(img, p, 2, (60, 200, 60), -1, cv2.LINE_AA)
        else:
            cv2.drawMarker(img, p, (0, 0, 220), cv2.MARKER_TILTED_CROSS, 5, 1)


def dibujar_minimapa(img, trayectoria, vehiculos_mundo, estados, lado=220):
    """Mapa desde arriba en la esquina: trayectoria de la cámara y vehículos."""
    H, W = img.shape[:2]
    x0, y0 = W - lado - 10, 10
    capa = img.copy()
    cv2.rectangle(capa, (x0, y0), (x0 + lado, y0 + lado), (25, 25, 25), -1)
    cv2.addWeighted(capa, 0.7, img, 0.3, 0, dst=img)
    tr = np.array([p[1:] for p in trayectoria]) if trayectoria else np.zeros((1, 2))
    pts = [tr] + [np.array(v) for v in vehiculos_mundo.values() if len(v)]
    todo = np.vstack(pts)
    cmin, cmax = todo.min(axis=0), todo.max(axis=0)
    centro = (cmin + cmax) / 2
    extension = max(6.0, float((cmax - cmin).max()) * 1.15)
    esc = (lado - 20) / extension

    def a_px(p):
        p = np.asarray(p, float).reshape(-1, 2)
        return np.stack([x0 + lado / 2 + (p[:, 0] - centro[0]) * esc,
                         y0 + lado / 2 - (p[:, 1] - centro[1]) * esc], axis=1).astype(np.int32)

    for tid, pos in vehiculos_mundo.items():
        if not len(pos):
            continue
        q = a_px(pos)
        if estados.get(tid) == "en movimiento":
            cv2.polylines(img, [q.reshape(-1, 1, 2)], False, (122, 175, 27), 2, cv2.LINE_AA)
            cv2.circle(img, tuple(q[-1]), 4, (122, 175, 27), -1)
        elif estados.get(tid) == "estacionado":
            c = tuple(np.median(q, axis=0).astype(int))
            cv2.rectangle(img, (c[0] - 4, c[1] - 4), (c[0] + 4, c[1] + 4), (52, 104, 235), -1)
    q = a_px(tr)
    if len(q) > 1:
        cv2.polylines(img, [q.reshape(-1, 1, 2)], False, (214, 120, 42), 2, cv2.LINE_AA)
    cv2.circle(img, tuple(q[-1]), 5, (255, 255, 255), -1)
    cv2.putText(img, "MAPA (vista de arriba)", (x0 + 6, y0 + lado - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, f"{extension:.0f} m", (x0 + lado - 46, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (220, 220, 220), 1, cv2.LINE_AA)


def titulo_barra(img, texto, color):
    """Franja con el título en la parte inferior del panel."""
    h = img.shape[0]
    cv2.rectangle(img, (0, h - 32), (img.shape[1], h), color, -1)
    cv2.putText(img, texto, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)


def estado_movimiento(st, umbral) -> str | None:
    if st.n_medidas < 5:
        return None
    return "estacionado" if st.mediana_kmh < umbral else "en movimiento"


# ---------------------------------------------------------------------- #
def main():
    args = parse_args()
    fuente, es_webcam = resolver_fuente(args.video)
    cap = abrir(fuente)
    info = info_video(cap)
    fps, fps_ok = fps_confiable(cap, args.fps)
    W, H = info["ancho"], info["alto"]
    total = info["frames"] if not es_webcam and info["frames"] > 0 else None
    if not fps_ok:
        print(f"AVISO: FPS reportados = {info['fps']}; se usa {fps}. Corrige con --fps.")

    if not Path(args.calibracion).exists():
        sys.exit(f"No existe '{args.calibracion}'. Calibra con un cuadro de este video:\n"
                 f"  python calibrar.py --video \"{fuente}\" --modo homografia --salida {args.calibracion}")
    try:
        calib = Calibracion.cargar(args.calibracion).ajustar_resolucion(W, H)
    except ValueError as e:
        sys.exit(f"ERROR de calibracion: {e}")
    if not calib.zona_px or (calib.modo == "homografia" and calib.zona_px == calib.puntos_px):
        print("CONSEJO: en calibrar.py (fase 2) dibuja una zona que cubra TODO el suelo visible: "
              "la odometria busca puntos ORB solo dentro de la zona.")

    modelo = None
    if not args.sin_yolo:
        from ultralytics import YOLO
        modelo = YOLO(args.modelo)

    nombre = "webcam" if es_webcam else Path(str(fuente)).stem
    carpeta = Path(args.salida) / f"{nombre}_nivel3"
    carpeta.mkdir(parents=True, exist_ok=True)

    vo = OdometriaVisual(calib, n_puntos=args.orb_puntos, ransac_m=args.ransac_m)
    est_ap = EstimadorVelocidad(calib)          # el radar del Nivel 1, tal cual
    est_re = EstimadorVelocidad(calib)          # el mismo, pero con posiciones del MUNDO
    ctl_ap = ControlLimite(args.limite)
    ctl_re = ControlLimite(args.limite)
    tray_px: dict[int, deque] = defaultdict(lambda: deque(maxlen=40))
    tray_mundo: dict[int, deque] = defaultdict(lambda: deque(maxlen=400))
    escritor = None
    ruta_video = carpeta / f"{nombre}_odometria.mp4"
    margen = max(6, int(0.02 * max(W, H)))
    filas_csv = []

    print(f"\nVideo: {fuente} | {W}x{H} | {fps:.2f} fps" + (f" | {total} cuadros" if total else ""))
    print(f"Calibracion: {calib.descripcion()} | YOLO: {'no' if modelo is None else args.modelo}"
          f" | ORB: {args.orb_puntos} puntos | tracker: {args.tracker}")

    idx, t, t_prev = -1, 0.0, None
    t0_reloj = time.perf_counter()
    ultimo_print = t0_reloj
    pausado = False
    try:
        while True:
            if pausado:
                k = cv2.waitKey(50) & 0xFF
                if k == ord(" "):
                    pausado = False
                elif k in (27, ord("q")) or ventana_cerrada(VENTANA):
                    break
                continue
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % max(1, args.salto):
                continue
            t = (time.perf_counter() - t0_reloj) if es_webcam else idx / fps
            if args.max_segundos and t > args.max_segundos:
                break
            dt = (t - t_prev) if t_prev is not None else 1.0 / fps
            t_prev = t

            # 1) Detección: vehículos (se miden) y personas (solo se excluyen de la odometría)
            detecciones, excluir = [], []
            if modelo is not None:
                res = modelo.track(frame, persist=True, tracker=args.tracker,
                                   classes=[CLASE_PERSONA, *CLASES_VEHICULO], conf=args.conf,
                                   imgsz=args.imgsz, device=args.dispositivo, verbose=False)[0]
                if res.boxes is not None and len(res.boxes):
                    cajas = res.boxes.xyxy.cpu().numpy()
                    clases = res.boxes.cls.int().cpu().tolist()
                    ids = (res.boxes.id.int().cpu().tolist() if res.boxes.id is not None
                           else [None] * len(clases))
                    for caja, c, tid in zip(cajas, clases, ids):
                        excluir.append(caja)
                        if c in CLASES_VEHICULO and tid is not None:
                            detecciones.append((caja, CLASES_VEHICULO[c], tid))

            # 2) Odometría visual: movimiento de la cámara (sin usar autos ni personas)
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            r = vo.procesar(gris, t, excluir)

            antes = frame.copy()
            despues = frame.copy()
            dibujar_zona(despues, calib, alpha=0.04)
            dibujar_flujo_aparente(antes, r, dt)
            dibujar_puntos_compensados(despues, r)
            pose_inv = np.linalg.inv(vo.pose)

            # 3) Vehículos: misma medición en coordenadas de la cámara vs del mundo
            for caja, clase, tid in detecciones:
                x1, y1, x2, y2 = caja
                punto = ((x1 + x2) / 2.0, float(y2))
                completa = (x1 > margen and y1 > margen and x2 < W - margen and y2 < H - margen)
                local = calib.a_metros([punto])[0]
                mundo = vo.a_mundo(local)[0]
                v_ap = est_ap.actualizar(tid, t, punto, clase, medir=completa)
                v_re = est_re.actualizar(tid, t, punto, clase, medir=completa, pos_m=mundo)
                e_ap, _ = ctl_ap.evaluar(tid, v_ap)
                e_re, _ = ctl_re.evaluar(tid, v_re)
                tray_px[tid].append(punto)
                if completa and calib.en_zona(punto):
                    tray_mundo[tid].append(tuple(mundo))
                dibujar_vehiculo(antes, caja, tid, clase, v_ap, e_ap, tray_px[tid])
                # Rastro en el MUNDO, reproyectado a la imagen actual: un auto quieto deja un punto
                rastro = None
                if len(tray_mundo[tid]) > 1:
                    loc = aplicar(pose_inv, np.array(tray_mundo[tid])[-60:])
                    loc = loc[calib.visibles(loc)]            # solo lo que está delante
                    if len(loc) > 1:
                        rastro = calib.a_pixeles(loc)
                dibujar_vehiculo(despues, caja, tid, clase, v_re, e_re, rastro)
                em = estado_movimiento(est_re.stats[tid], args.umbral_estacionado)
                if em:
                    etiqueta(despues, em.upper(), x1, y2 + (44 if e_re == "exceso" else 22),
                             (90, 90, 90) if em == "estacionado" else (190, 110, 20), 0.5, 1)
                filas_csv.append([idx, f"{t:.3f}", tid, clase, f"{punto[0]:.1f}", f"{punto[1]:.1f}",
                                  f"{local[0]:.2f}", f"{local[1]:.2f}", f"{mundo[0]:.2f}", f"{mundo[1]:.2f}",
                                  "" if v_ap is None else f"{v_ap:.1f}", "" if v_re is None else f"{v_re:.1f}"])
            est_ap.purgar(t)
            est_re.purgar(t)

            # 4) Paneles de información
            med_ap = est_ap.vehiculos_medidos()
            med_re = est_re.vehiculos_medidos()
            n_falsos = sum(1 for s in med_ap.values() if s.mediana_kmh >= args.umbral_estacionado)
            estados = {k: estado_movimiento(s, args.umbral_estacionado) for k, s in med_re.items()}
            n_est = sum(1 for e in estados.values() if e == "estacionado")
            n_mov = sum(1 for e in estados.values() if e == "en movimiento")
            dibujar_panel(antes, [
                f"Suelo 'se mueve' a: {r.v_suelo_aparente_kmh:5.1f} km/h",
                f"Vehiculos que cree en movimiento: {n_falsos}",
            ])
            vo_txt = {"ok": "OK", "inicio": "iniciando", "perdida": "PERDIDA (pocos puntos)"}[r.estado]
            dibujar_panel(despues, [
                f"Camara: {r.v_camara_kmh:5.1f} km/h   giro {r.giro_deg:+5.1f} deg",
                f"Recorrido: {vo.distancia_recorrida():6.1f} m   odometria {vo_txt}",
                f"Suelo compensado: {r.v_suelo_compensada_kmh:4.1f} km/h (ruido)",
                f"ORB {r.n_puntos} | parejas {r.n_parejas} | inliers {r.n_inliers}",
                f"Estacionados {n_est} | En movimiento {n_mov}",
            ], alerta=r.estado == "perdida")
            dibujar_minimapa(despues, vo.trayectoria, tray_mundo, estados)
            titulo_barra(antes, "ANTES: radar del Nivel 1 (sin compensar)", (40, 90, 200))
            titulo_barra(despues, "DESPUES: odometria visual ORB + RANSAC (Nivel 3)", (150, 100, 20))
            if args.vista == "comparar":
                a, _ = ajustar_para_pantalla(antes, 960, 960)
                b, _ = ajustar_para_pantalla(despues, 960, 960)
                salida = np.hstack([a, b])
            else:
                salida = despues

            if not args.sin_video_salida:
                if escritor is None:
                    escritor = cv2.VideoWriter(str(ruta_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                               fps / max(1, args.salto), (salida.shape[1], salida.shape[0]))
                escritor.write(salida)
            if not args.no_mostrar:
                vis, _ = ajustar_para_pantalla(salida, 1600, 900)
                cv2.imshow(VENTANA, vis)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord("q")) or ventana_cerrada(VENTANA):
                    break
                if k == ord(" "):
                    pausado = True

            ahora = time.perf_counter()
            if ahora - ultimo_print > 3:
                prog = f"{100 * (idx + 1) / total:5.1f}% " if total else ""
                print(f"  {prog}t={t:5.1f}s | camara {r.v_camara_kmh:5.1f} km/h | recorrido "
                      f"{vo.distancia_recorrida():5.1f} m | inliers {r.n_inliers} | "
                      f"{(idx + 1) / (ahora - t0_reloj):.1f} cuadros/s")
                ultimo_print = ahora
    except KeyboardInterrupt:
        print("\nInterrumpido (Ctrl+C). Se guarda lo procesado.")
    finally:
        cap.release()
        if escritor is not None:
            escritor.release()
        cv2.destroyAllWindows()

    generar_salidas(carpeta, vo, est_ap, est_re, tray_mundo, filas_csv, args, fuente)
    if escritor is not None:
        print(f"  - Video antes/despues: {ruta_video.name}")


def generar_salidas(carpeta, vo, est_ap, est_re, tray_mundo, filas_csv, args, fuente):
    hist = vo.historial
    if not hist:
        print("No se proceso ningun cuadro.")
        return
    # ---- CSV ----
    with open(carpeta / "odometria.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["t_video_s", "x_m", "y_m", "giro_deg", "v_camara_kmh", "v_camara_bruta_kmh",
                    "v_suelo_aparente_kmh", "v_suelo_compensada_kmh", "puntos_orb", "parejas",
                    "inliers", "estado"])
        for h, (_, x, y) in zip(hist, vo.trayectoria):
            w.writerow([f"{h.t:.3f}", f"{x:.3f}", f"{y:.3f}", f"{h.giro_deg:.2f}", f"{h.v_camara_kmh:.2f}",
                        f"{h.v_camara_bruta_kmh:.2f}", f"{h.v_suelo_aparente_kmh:.2f}",
                        f"{h.v_suelo_compensada_kmh:.2f}", h.n_puntos, h.n_parejas, h.n_inliers, h.estado])
    with open(carpeta / "vehiculos_historial_nivel3.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_video_s", "id", "clase", "px_x", "px_y", "x_local_m", "y_local_m",
                    "x_mundo_m", "y_mundo_m", "v_aparente_kmh", "v_real_kmh"])
        w.writerows(filas_csv)

    med_re = est_re.vehiculos_medidos()
    filas = []
    for tid, st in sorted(med_re.items()):
        sa = est_ap.stats.get(tid)
        if st.n_medidas < 5 or sa is None or sa.n_medidas < 5:
            continue
        filas.append({"id": tid, "clase": st.clase, "aparente": sa.mediana_kmh, "real": st.mediana_kmh,
                      "estado": estado_movimiento(st, args.umbral_estacionado),
                      "medidas": st.n_medidas})
    with open(carpeta / "vehiculos_nivel3.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "clase", "v_aparente_nivel1_kmh", "v_real_nivel3_kmh", "estado", "medidas"])
        for r in filas:
            w.writerow([r["id"], r["clase"], f"{r['aparente']:.1f}", f"{r['real']:.1f}", r["estado"], r["medidas"]])

    # ---- Resumen ----
    ok = [h for h in hist if h.estado == "ok"]
    dur = hist[-1].t - hist[0].t
    dist = vo.distancia_recorrida()
    v_media = dist / dur * 3.6 if dur > 0 else 0.0
    L = ["RESUMEN - ODOMETRIA VISUAL (Nivel 3)", "=" * 64,
         f"Video: {fuente}",
         f"Cuadros: {len(hist)} | odometria OK en {100 * len(ok) / max(1, len(hist) - 1):.0f}% | "
         f"inliers medios {np.mean([h.n_inliers for h in ok]) if ok else 0:.0f}",
         f"Recorrido de la camara: {dist:.2f} m en {dur:.1f} s -> velocidad media {v_media:.1f} km/h",
         f"Desplazamiento neto (inicio -> fin): "
         f"{np.hypot(*np.subtract(vo.trayectoria[-1][1:], vo.trayectoria[0][1:])):.2f} m | "
         f"giro final {vo.giro_deg():+.1f} grados",
         f"El suelo 'se movia' (sin compensar) a {np.median([h.v_suelo_aparente_kmh for h in ok]) if ok else 0:.1f} km/h;"
         f" tras compensar: {np.median([h.v_suelo_compensada_kmh for h in ok]) if ok else 0:.1f} km/h (ruido)"]
    if filas:
        L += ["", "VEHICULOS (velocidad tipica = mediana)",
              f"{'ID':>5} {'clase':<8} {'aparente N1':>12} {'real N3':>9}  estado", "-" * 50]
        for r in filas:
            L.append(f"{r['id']:>5} {r['clase']:<8} {r['aparente']:>10.1f}   {r['real']:>8.1f}  {r['estado']}")
        n_est = sum(r["estado"] == "estacionado" for r in filas)
        n_falsos = sum(r["aparente"] >= args.umbral_estacionado for r in filas if r["estado"] == "estacionado")
        L += ["", f"Vehiculos estacionados: {n_est}. El radar del Nivel 1 creia que {n_falsos} de ellos "
                  "se movian; con odometria visual quedan quietos."]
    texto = "\n".join(L)
    (carpeta / "reporte_nivel3.txt").write_text(texto + "\n", encoding="utf-8")
    print("\n" + texto)

    # ---- Gráficos ----
    try:
        from radar.graficos_n3 import figura_comparacion, figura_odometria
        vehiculos = {}
        for r in filas:
            m = np.array(tray_mundo.get(r["id"], []))
            if len(m):
                vehiculos[r["id"]] = {"estado": r["estado"], "mundo": m, "clase": r["clase"]}
        sub = (f"Recorrido {dist:.1f} m en {dur:.1f} s · velocidad media {v_media:.1f} km/h · "
               f"giro final {vo.giro_deg():+.1f}° · odometría OK en {100 * len(ok) / max(1, len(hist) - 1):.0f}% de los cuadros")
        figura_odometria(vo.trayectoria, hist, vehiculos, carpeta / "odometria.png", sub)
        if filas:
            figura_comparacion(filas, args.umbral_estacionado, carpeta / "comparacion_velocidades.png")
    except Exception as e:
        traceback.print_exc()
        print(f"No se pudieron generar los graficos: {e}")
    print(f"\nResultados en: {carpeta.resolve()}")
    print("  - odometria.csv, vehiculos_nivel3.csv, vehiculos_historial_nivel3.csv, reporte_nivel3.txt")
    print("  - odometria.png" + (", comparacion_velocidades.png" if filas else ""))


if __name__ == "__main__":
    main()
