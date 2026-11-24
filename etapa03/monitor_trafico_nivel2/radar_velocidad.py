"""
MONITOR INTELIGENTE DE TRAFICO - camara fija
  NIVEL 1: Radar de velocidad (YOLO + ByteTrack + calibracion -> km/h, alerta roja)
  NIVEL 2: Analitica de transito (historial CSV, conteo por linea virtual y, al
           cerrar, mapa de calor por carril + grafica de vehiculos por hora)

Uso:
    python radar_velocidad.py --video mi_video.mp4 --calibracion calibracion.json --limite 50
    python radar_velocidad.py --video mi_video.mp4 --hora-inicio "2026-11-20 07:30"
    python radar_velocidad.py                       # abre el explorador para elegir el video
    python radar_velocidad.py --video 0             # webcam en vivo

Teclas: ESPACIO pausa | Q / ESC salir (los reportes se generan igual al salir)
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path

import cv2

from radar.calibracion import Calibracion
from radar.conteo import ContadorLinea
from radar.dibujo import (ajustar_para_pantalla, dibujar_linea_conteo, dibujar_panel,
                          dibujar_vehiculo, dibujar_zona, ventana_cerrada)
from radar.registro import MuestreadorFondo, RegistroHistorial, guardar_sesion
from radar.tiempo import determinar_hora_inicio
from radar.velocidad import ControlLimite, EstimadorVelocidad
from radar.video import abrir, fps_confiable, info_video, resolver_fuente

# Clases COCO de vehiculos
CLASES_VEHICULO = {2: "auto", 3: "moto", 5: "bus", 7: "camion"}
VENTANA = "Monitor de trafico - Radar de velocidad"


def parse_args():
    ap = argparse.ArgumentParser(description="Radar de velocidad + analitica de transito (camara fija)")
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
    # ---- Nivel 2 ----
    n2 = ap.add_argument_group("Nivel 2 - analitica")
    n2.add_argument("--hora-inicio", default=None,
                    help='Hora real del inicio del video: "2026-11-20 07:30" o "07:30". '
                         "Si no se da, se lee del nombre del archivo o del MP4")
    n2.add_argument("--registro-hz", type=float, default=5.0,
                    help="Registros por segundo y vehiculo en historial.csv (0 = todos los cuadros)")
    n2.add_argument("--linea-conteo", type=float, default=0.5,
                    help="Posicion de la linea de conteo a lo largo de la zona (0-1, 0.5 = centro)")
    n2.add_argument("--carriles", type=int, default=None,
                    help="Numero de carriles (por defecto se detectan automaticamente)")
    n2.add_argument("--sin-analitica", action="store_true",
                    help="No generar los graficos al cerrar (se pueden hacer luego con analitica.py)")
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

    # ---------------- Hora real (Nivel 2) ---------------- #
    try:
        hora_inicio, fuente_hora = determinar_hora_inicio(fuente, args.hora_inicio, es_webcam)
    except ValueError as e:
        sys.exit(str(e))

    # ---------------- Modelo ---------------- #
    from ultralytics import YOLO          # import tardio: el --help responde rapido
    modelo = YOLO(args.modelo)

    # ---------------- Salidas ---------------- #
    nombre = "webcam" if es_webcam else Path(str(fuente)).stem
    carpeta = Path(args.salida) / nombre
    (carpeta / "infracciones").mkdir(parents=True, exist_ok=True)
    escritor = None
    ruta_video = carpeta / f"{nombre}_radar.mp4"
    if not args.sin_video_salida:
        escritor = cv2.VideoWriter(str(ruta_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                   fps / max(1, args.salto), (W, H))

    estimador = EstimadorVelocidad(calib, ventana_s=args.ventana)
    control = ControlLimite(args.limite, confirmar=args.confirmar)
    trayectorias: dict[int, deque] = defaultdict(lambda: deque(maxlen=40))

    # ---- Nivel 2: registro, linea de conteo y fondo ----
    registro = RegistroHistorial(carpeta, hora_inicio, args.registro_hz)
    contador = ContadorLinea(calib, (W, H), fraccion=args.linea_conteo)
    if contador.segmento_px is None:
        print("AVISO: la linea de conteo cae fuera de la zona de medicion; no se contaran vehiculos. "
              "Revisa la zona en calibrar.py o cambia --linea-conteo.")
    fondo = MuestreadorFondo(max(1, total // 30) if total else int(fps * 2))
    sesion = {
        "nombre": nombre, "video": str(fuente), "resolucion": [W, H], "fps": fps,
        "hora_inicio": hora_inicio.isoformat(sep=" ") if hora_inicio else None,
        "fuente_hora": fuente_hora, "duracion_s": 0.0,
        "calibracion": calib.a_dict(), "limite_kmh": args.limite, "modelo": args.modelo,
        "registro_hz": args.registro_hz, "fondo": "fondo.jpg",
        "linea_conteo": {"fraccion": args.linea_conteo, "s_m": contador.s_linea,
                         "segmento_px": contador.segmento_px},
    }
    guardar_sesion(carpeta / "sesion.json", sesion)

    print(f"\nVideo: {fuente} | {W}x{H} | {fps:.2f} fps"
          + (f" | {total} cuadros ({total / fps:.1f} s)" if total else ""))
    print(f"Calibracion: {calib.descripcion()} | Limite: {args.limite:.0f} km/h | "
          f"Modelo: {args.modelo} | Tracker: {args.tracker}")
    print(f"Hora de inicio: {hora_inicio:%Y-%m-%d %H:%M:%S} ({fuente_hora})" if hora_inicio
          else f"Hora de inicio: {fuente_hora}")
    if calib.modo == "ppm":
        print("Nota: el modo PPM asume que la camara ve la calle de costado o desde arriba. "
              "Con perspectiva fuerte usa --modo homografia en calibrar.py.")

    # Una caja a menos de este margen del borde suele estar recortada (auto entrando
    # o saliendo): su centro inferior no representa el movimiento real.
    margen_borde = max(6, int(0.02 * max(W, H)))
    idx = -1
    t = 0.0
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

            # Tiempo DEL VIDEO (no del procesamiento). En webcam se usa el reloj real.
            t = (time.perf_counter() - t0_reloj) if es_webcam else idx / fps
            if args.max_segundos and t > args.max_segundos:
                break

            # ------------- Deteccion + seguimiento ------------- #
            res = modelo.track(frame, persist=True, tracker=args.tracker,
                               classes=list(CLASES_VEHICULO), conf=args.conf, imgsz=args.imgsz,
                               device=args.dispositivo, verbose=False)[0]
            crudo = frame.copy()                 # para capturas limpias y para el fondo
            fondo.agregar(idx, crudo)
            dibujar_zona(frame, calib)
            dibujar_linea_conteo(frame, contador, t)
            parpadeo = (idx // max(1, int(fps / 4))) % 2 == 0

            hay_exceso = False
            if res.boxes is not None and res.boxes.id is not None:
                cajas = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.int().cpu().tolist()
                clases = res.boxes.cls.int().cpu().tolist()
                confs = res.boxes.conf.cpu().tolist()
                for caja, tid, c, conf in zip(cajas, ids, clases, confs):
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

                    # ---- Nivel 2: historial + conteo ----
                    sd = calib.a_via([punto])[0]
                    registro.registrar(idx, t, tid, clase, conf, caja, punto, sd,
                                       calib.en_zona(punto), kmh, estado)
                    ev = contador.actualizar(tid, t, punto, estimador.stats[tid].clase, kmh)
                    if ev is not None:
                        registro.registrar_conteo(ev, contador.flecha(ev.sentido))

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
            contador.purgar(t)
            n_medidos = len(estimador.vehiculos_medidos())
            hora_txt = registro.fecha(t)[11:19] if hora_inicio else formatear_tiempo(t)
            dibujar_panel(frame, [
                f"RADAR DE VELOCIDAD | Limite {args.limite:.0f} km/h",
                f"{hora_txt}   {calib.descripcion()}",
                f"Vehiculos medidos: {n_medidos}   Excesos: {len(control.infractores)}",
                f"Contados: {contador.total}  ({contador.flecha(1)} {contador.por_sentido[1]} | "
                f"{contador.flecha(-1)} {contador.por_sentido[-1]})",
            ], alerta=hay_exceso)

            if escritor is not None:
                escritor.write(frame)

            if not args.no_mostrar:
                vis, _ = ajustar_para_pantalla(frame)
                cv2.imshow(VENTANA, vis)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord("q")) or ventana_cerrada(VENTANA):
                    break
                if k == ord(" "):
                    pausado = True

            ahora = time.perf_counter()
            if ahora - ultimo_print > 3:
                vel_proc = (idx + 1) / (ahora - t0_reloj)
                prog = f"{100 * (idx + 1) / total:5.1f}% " if total else ""
                print(f"  {prog}cuadro {idx + 1}{'/' + str(total) if total else ''} | "
                      f"{vel_proc:.1f} cuadros/s | medidos {n_medidos} | contados {contador.total} | "
                      f"excesos {len(control.infractores)}")
                ultimo_print = ahora
    except KeyboardInterrupt:
        print("\nInterrumpido (Ctrl+C). Se guardara lo procesado hasta ahora.")
    finally:
        cap.release()
        if escritor is not None:
            escritor.release()
        cv2.destroyAllWindows()
        registro.cerrar()
        fondo.guardar(carpeta / "fondo.jpg", (W, H))
        sesion["duracion_s"] = round(t + 1.0 / fps, 3) if not es_webcam else round(t, 3)
        sesion["cuadros_procesados"] = idx + 1
        guardar_sesion(carpeta / "sesion.json", sesion)

    # ---------------- Resumen Nivel 1 ---------------- #
    medidos = estimador.vehiculos_medidos()
    lineas = [
        "RESUMEN - RADAR DE VELOCIDAD (Nivel 1)",
        f"Video: {fuente}",
        f"Duracion analizada: {formatear_tiempo(sesion['duracion_s'])}  |  {calib.descripcion()}"
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

    # ---------------- Nivel 2: analitica automatica al cerrar ---------------- #
    if not args.sin_analitica:
        print("\nGenerando analitica de transito (Nivel 2)...")
        try:
            from radar.analitica import generar_reportes
            generar_reportes([carpeta], carpeta, args.carriles)
        except Exception as e:  # la analitica nunca debe hacer perder los datos ya guardados
            traceback.print_exc()
            print(f"No se pudo generar la analitica ({e}). Los datos estan en historial.csv y "
                  f"conteo.csv; reintenta con: python analitica.py \"{carpeta}\"")

    print(f"\nResultados en: {carpeta.resolve()}")
    if escritor is not None:
        print(f"  - Video anotado: {ruta_video.name}")
    print(f"  - Infracciones: infracciones/ ({len(control.infractores)})  |  reporte_nivel1.txt")
    print("  - Nivel 2: historial.csv, conteo.csv, vehiculos_contados.csv, carriles.csv,")
    print("             mapa_calor.png, vehiculos_por_hora.png, resumen_nivel2.txt")


if __name__ == "__main__":
    main()
