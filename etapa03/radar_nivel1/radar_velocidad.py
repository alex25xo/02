"""
NIVEL 1 - RADAR DE VELOCIDAD (camara fija)
Monitor Inteligente de Trafico y Navegacion

Detecta y sigue vehiculos con YOLO + ByteTrack, convierte su posicion a metros
con la calibracion (PPM u homografia) y muestra la velocidad en km/h.
Si un vehiculo supera el limite, su caja se vuelve ROJA (alerta) y se guarda
una captura de la infraccion.

Uso:
    python radar_velocidad.py --video mi_video.mp4 --calibracion calibracion.json --limite 50
    python radar_velocidad.py                       # abre el explorador para elegir el video
    python radar_velocidad.py --video 0             # webcam en vivo

Teclas: ESPACIO pausa | Q / ESC salir
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2

from radar.calibracion import Calibracion
from radar.dibujo import ajustar_para_pantalla, dibujar_panel, dibujar_vehiculo, dibujar_zona
from radar.velocidad import ControlLimite, EstimadorVelocidad
from radar.video import abrir, fps_confiable, info_video, resolver_fuente

# Clases COCO de vehiculos
CLASES_VEHICULO = {2: "auto", 3: "moto", 5: "bus", 7: "camion"}
VENTANA = "Radar de velocidad - Nivel 1"


def parse_args():
    ap = argparse.ArgumentParser(description="Radar de velocidad con YOLO (camara fija)")
    ap.add_argument("--video", help="Ruta del video o indice de webcam. Sin valor: abre explorador")
    ap.add_argument("--calibracion", default="calibracion.json", help="Archivo de calibrar.py")
    ap.add_argument("--limite", type=float, default=50.0, help="Limite de velocidad en km/h")
    ap.add_argument("--modelo", default="yolo11n.pt",
                    help="Pesos YOLO (yolo11n.pt rapido en CPU; yolo11s/m.pt mas precisos)")
    ap.add_argument("--conf", type=float, default=0.30, help="Confianza minima de deteccion")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="Tamano de inferencia (960-1280 ayuda con autos lejanos/pequenos)")
    ap.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml o botsort.yaml")
    ap.add_argument("--dispositivo", default=None, help="cpu, 0 (GPU), mps ... (auto por defecto)")
    ap.add_argument("--fps", type=float, default=None,
                    help="Forzar FPS si el archivo reporta un valor incorrecto")
    ap.add_argument("--salto", type=int, default=1,
                    help="Procesar 1 de cada N cuadros (2 = el doble de rapido)")
    ap.add_argument("--ventana", type=float, default=0.6,
                    help="Segundos de historia para calcular la velocidad")
    ap.add_argument("--confirmar", type=int, default=3,
                    help="Medidas seguidas sobre el limite para confirmar exceso")
    ap.add_argument("--salida", default="salidas", help="Carpeta de resultados")
    ap.add_argument("--no-mostrar", action="store_true", help="No abrir ventana (mas rapido)")
    ap.add_argument("--sin-video-salida", action="store_true", help="No guardar el video anotado")
    ap.add_argument("--max-segundos", type=float, default=None,
                    help="Procesar solo los primeros N segundos (pruebas rapidas)")
    return ap.parse_args()


def formatear_tiempo(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:04.1f}"


def main():
    args = parse_args()

    # ---------------- Video ---------------- #
    fuente, es_webcam = resolver_fuente(args.video)
    cap = abrir(fuente)
    info = info_video(cap)
    fps, fps_ok = fps_confiable(cap, args.fps)
    W, H = info["ancho"], info["alto"]
    if not fps_ok:
        print(f"AVISO: el video reporta FPS={info['fps']}. Se usara {fps}. "
              "Corrigelo con --fps si sabes el valor real.")
    total = info["frames"] if not es_webcam and info["frames"] > 0 else None

    # ---------------- Calibracion ---------------- #
    if not Path(args.calibracion).exists():
        sys.exit(f"No existe '{args.calibracion}'. Primero ejecuta:\n"
                 f"  python calibrar.py --video \"{fuente}\" --modo ppm")
    calib = Calibracion.cargar(args.calibracion)
    try:
        calib = calib.ajustar_resolucion(W, H)
    except ValueError as e:
        sys.exit(f"ERROR de calibracion: {e}")
    if calib.origen and not es_webcam and Path(calib.origen).name != Path(str(fuente)).name:
        print(f"AVISO: la calibracion se hizo con '{Path(calib.origen).name}'. "
              "Si la camara cambio de posicion, vuelve a calibrar.")

    # ---------------- Modelo ---------------- #
    from ultralytics import YOLO          # import tardio: el --help responde rapido
    modelo = YOLO(args.modelo)

    # ---------------- Salidas ---------------- #
    nombre = "webcam" if es_webcam else Path(str(fuente)).stem
    carpeta = Path(args.salida) / nombre
    (carpeta / "infracciones").mkdir(parents=True, exist_ok=True)
    escritor = None
    if not args.sin_video_salida:
        fps_salida = fps / max(1, args.salto)
        ruta_video = carpeta / f"{nombre}_radar.mp4"
        escritor = cv2.VideoWriter(str(ruta_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                   fps_salida, (W, H))

    estimador = EstimadorVelocidad(calib, ventana_s=args.ventana)
    control = ControlLimite(args.limite, confirmar=args.confirmar)
    trayectorias: dict[int, deque] = defaultdict(lambda: deque(maxlen=40))

    print(f"\nVideo: {fuente} | {W}x{H} | {fps:.2f} fps"
          + (f" | {total} cuadros ({total / fps:.1f} s)" if total else ""))
    print(f"Calibracion: {calib.descripcion()} | Limite: {args.limite:.0f} km/h | "
          f"Modelo: {args.modelo} | Tracker: {args.tracker}")
    if calib.modo == "ppm":
        print("Nota: el modo PPM asume que la camara ve la calle de costado o desde arriba. "
              "Con perspectiva fuerte usa --modo homografia en calibrar.py.")

    # Una caja a menos de este margen del borde suele estar recortada (auto entrando
    # o saliendo): su centro inferior no representa el movimiento real.
    margen_borde = max(6, int(0.02 * max(W, H)))
    idx = -1
    t0_reloj = time.perf_counter()
    ultimo_print = t0_reloj
    pausado = False
    escala_vis = 1.0

    while True:
        if pausado:
            k = cv2.waitKey(50) & 0xFF
            if k == ord(" "):
                pausado = False
            elif k in (27, ord("q")):
                break
            continue

        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % max(1, args.salto):
            continue

        # Tiempo DEL VIDEO (no del procesamiento). En webcam se usa el reloj real.
        t = (time.perf_counter() - t0_reloj) if es_webcam else idx / fps
        if args.max_segundos and t > args.max_segundos:
            break

        # ------------- Deteccion + seguimiento ------------- #
        res = modelo.track(frame, persist=True, tracker=args.tracker,
                           classes=list(CLASES_VEHICULO), conf=args.conf, imgsz=args.imgsz,
                           device=args.dispositivo, verbose=False)[0]
        crudo = frame.copy()                 # para capturas limpias de infracciones
        dibujar_zona(frame, calib)
        parpadeo = (idx // max(1, int(fps / 4))) % 2 == 0

        hay_exceso = False
        if res.boxes is not None and res.boxes.id is not None:
            cajas = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.int().cpu().tolist()
            clases = res.boxes.cls.int().cpu().tolist()
            for caja, tid, c in zip(cajas, ids, clases):
                x1, y1, x2, y2 = caja
                clase = CLASES_VEHICULO.get(c, "vehiculo")
                punto = ((x1 + x2) / 2.0, float(y2))          # contacto con el suelo
                # Caja cortada por el borde -> su centro inferior no es fiable
                completa = (x1 > margen_borde and y1 > margen_borde and
                            x2 < W - margen_borde and y2 < H - margen_borde)
                kmh = estimador.actualizar(tid, t, punto, clase, medir=completa)
                estado, nuevo = control.evaluar(tid, kmh)
                trayectorias[tid].append(punto)
                hay_exceso |= estado == "exceso"
                dibujar_vehiculo(frame, caja, tid, clase, kmh, estado, trayectorias[tid], parpadeo)

                if nuevo:                      # primera vez que se confirma el exceso
                    st = estimador.stats[tid]
                    st.exceso, st.kmh_exceso = True, kmh
                    cap_img = crudo.copy()
                    dibujar_vehiculo(cap_img, caja, tid, clase, kmh, "exceso", None, True)
                    dibujar_panel(cap_img, [f"INFRACCION  t={formatear_tiempo(t)}",
                                            f"Vehiculo #{tid} ({clase}): {kmh:.1f} km/h",
                                            f"Limite: {args.limite:.0f} km/h"], alerta=True)
                    archivo = carpeta / "infracciones" / f"exceso_id{tid}_{kmh:.0f}kmh_t{t:07.2f}s.jpg"
                    cv2.imwrite(str(archivo), cap_img)
                    print(f"  [!] EXCESO  t={formatear_tiempo(t)}  #{tid} {clase}: "
                          f"{kmh:.1f} km/h (limite {args.limite:.0f})")

        estimador.purgar(t)
        n_medidos = len(estimador.vehiculos_medidos())
        dibujar_panel(frame, [
            f"RADAR DE VELOCIDAD | Limite {args.limite:.0f} km/h",
            f"t = {formatear_tiempo(t)}   {calib.descripcion()}",
            f"Vehiculos medidos: {n_medidos}   Excesos: {len(control.infractores)}",
        ], alerta=hay_exceso)

        if escritor is not None:
            escritor.write(frame)

        if not args.no_mostrar:
            vis, escala_vis = ajustar_para_pantalla(frame)
            cv2.imshow(VENTANA, vis)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            if k == ord(" "):
                pausado = True

        ahora = time.perf_counter()
        if ahora - ultimo_print > 3:
            vel_proc = (idx + 1) / (ahora - t0_reloj)
            prog = f"{100 * (idx + 1) / total:5.1f}% " if total else ""
            print(f"  {prog}cuadro {idx + 1}{'/' + str(total) if total else ''} | "
                  f"{vel_proc:.1f} cuadros/s | medidos {n_medidos} | excesos {len(control.infractores)}")
            ultimo_print = ahora

    cap.release()
    if escritor is not None:
        escritor.release()
    cv2.destroyAllWindows()

    # ---------------- Resumen ---------------- #
    medidos = estimador.vehiculos_medidos()
    lineas = [
        "RESUMEN - RADAR DE VELOCIDAD (Nivel 1)",
        f"Video: {fuente}",
        f"Duracion analizada: {formatear_tiempo(max(0, idx) / fps)}  |  {calib.descripcion()}"
        f"  |  Limite: {args.limite:.0f} km/h",
        f"Vehiculos con velocidad medida: {len(medidos)}  |  Excesos: {len(control.infractores)}",
        "",
        "(v_tipica = mediana de sus medidas; es la velocidad representativa del vehiculo)",
        f"{'ID':>5} {'clase':<8} {'v_tipica':>9} {'v_max':>8} {'medidas':>8} {'medido (s)':>14}  estado",
        "-" * 72,
    ]
    for tid, st in sorted(medidos.items()):
        lineas.append(f"{tid:>5} {st.clase:<8} {st.mediana_kmh:>8.1f}  {st.max_kmh:>7.1f} "
                      f"{st.n_medidas:>8}  {st.t_primera:>6.1f}-{st.t_ultima:<6.1f}  "
                      f"{'EXCESO' if st.exceso else 'ok'}")
    if medidos:
        vel = [s.mediana_kmh for s in medidos.values()]
        lineas += ["-" * 72, f"Velocidad tipica del trafico (promedio): {sum(vel) / len(vel):.1f} km/h"]
    reporte = "\n".join(lineas)
    print("\n" + reporte)
    (carpeta / "reporte_nivel1.txt").write_text(reporte + "\n", encoding="utf-8")
    print(f"\nResultados en: {carpeta.resolve()}")
    if escritor is not None:
        print(f"  - Video anotado: {ruta_video.name}")
    print(f"  - Capturas de infracciones: infracciones/ ({len(control.infractores)})")
    print("  - Reporte: reporte_nivel1.txt")


if __name__ == "__main__":
    main()
