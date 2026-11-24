"""
NIVEL 2 - Procesar VARIOS videos de un mismo punto (un día de grabación) y
juntar toda la analítica en un solo reporte.

La hora de inicio de cada video se toma del NOMBRE del archivo:
  - D22N1657A1704.mp4        -> día 22, de 16:57 a 17:04 (año y mes salen de --fecha)
  - VID_20261122_165700.mp4  -> formato de celular

Pasos recomendados:
  1) Revisar (no procesa nada): horarios detectados + calibración dibujada en cada video
       python procesar_lote.py --carpeta ..\\VPNV2 --fecha 2026-11-22 --revisar
  2) Procesar todo (sin ventana) y generar la analítica del día completo
       python procesar_lote.py --carpeta ..\\VPNV2 --fecha 2026-11-22 --limite 60 --salto 2
  3) (Opcional) Regenerar solo los gráficos, por ejemplo forzando 4 carriles
       python procesar_lote.py --carpeta ..\\VPNV2 --fecha 2026-11-22 --solo-analitica --carriles 4
     o con carriles dibujados a mano (definir_carriles.py):
       python procesar_lote.py --carpeta ..\\VPNV2 --fecha 2026-11-22 --solo-analitica --carriles-def carriles_nivel2.json

Si se interrumpe (Ctrl+C o se apaga la PC), al volver a ejecutarlo continúa con
los videos que faltan: los que ya se procesaron completos se saltan.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from radar.calibracion import Calibracion  # noqa: E402
from radar.tiempo import hora_desde_nombre  # noqa: E402

EXTENSIONES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
PATRON_DNA = re.compile(r"D(\d{1,2})N(\d{2})(\d{2})A(\d{2})(\d{2})", re.IGNORECASE)


def horario_desde_nombre(nombre: str, fecha: datetime | None):
    """(inicio, fin_segun_nombre) o (None, None)."""
    m = PATRON_DNA.search(Path(nombre).stem)
    if m:
        if fecha is None:
            sys.exit("Los nombres tipo D22N1657A1704 no traen año ni mes: agrega --fecha AAAA-MM-DD")
        dia, h1, m1, h2, m2 = map(int, m.groups())
        base = fecha.replace(day=dia, hour=0, minute=0, second=0, microsecond=0)
        ini = base + timedelta(hours=h1, minutes=m1)
        fin = base + timedelta(hours=h2, minutes=m2)
        if fin < ini:
            fin += timedelta(days=1)
        return ini, fin
    h = hora_desde_nombre(nombre)
    return (h, None) if h else (None, None)


def info_video(ruta: Path) -> dict:
    cap = cv2.VideoCapture(str(ruta))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    d = {"ancho": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "alto": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
         "fps": fps, "duracion_s": n / fps if fps > 0 else 0.0, "cap": cap}
    return d


def ya_procesado(carpeta: Path, duracion_s: float) -> bool:
    meta = carpeta / "sesion.json"
    if not meta.exists() or not (carpeta / "conteo.csv").exists():
        return False
    try:
        d = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return d.get("duracion_s", 0) >= 0.97 * duracion_s


def revisar_calibracion(videos, calib: Calibracion, carpeta_rev: Path) -> Path:
    """Dibuja la calibración sobre un cuadro de cada video y arma un mosaico."""
    from calibrar import dibujar_preview
    carpeta_rev.mkdir(parents=True, exist_ok=True)
    miniaturas = []
    for v in videos:
        cap = v["info"]["cap"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(2 * (v["info"]["fps"] or 30)))
        ok, frame = cap.read()
        if not ok:
            continue
        try:
            c = calib.ajustar_resolucion(frame.shape[1], frame.shape[0])
            img = dibujar_preview(frame, c)
        except ValueError as e:
            img = frame.copy()
            cv2.putText(img, f"NO COINCIDE: {e}"[:90], (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255), 3, cv2.LINE_AA)
        etiqueta = f"{v['ruta'].name}   inicio {v['inicio']:%H:%M}" if v["inicio"] else v["ruta"].name
        cv2.rectangle(img, (0, 0), (img.shape[1], 50), (30, 30, 30), -1)
        cv2.putText(img, etiqueta, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(carpeta_rev / f"{v['ruta'].stem}.jpg"), img)
        miniaturas.append(cv2.resize(img, (640, int(640 * img.shape[0] / img.shape[1]))))
    if not miniaturas:
        return carpeta_rev
    h = max(m.shape[0] for m in miniaturas)
    miniaturas = [cv2.copyMakeBorder(m, 0, h - m.shape[0], 0, 0, cv2.BORDER_CONSTANT) for m in miniaturas]
    while len(miniaturas) % 2:
        miniaturas.append(np.zeros_like(miniaturas[0]))
    filas = [np.hstack(miniaturas[i:i + 2]) for i in range(0, len(miniaturas), 2)]
    cv2.imwrite(str(carpeta_rev / "mosaico.jpg"), np.vstack(filas))
    return carpeta_rev


def fmt(seg: float) -> str:
    m, s = divmod(int(seg), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def main():
    ap = argparse.ArgumentParser(description="Nivel 2: procesar varios videos y juntar la analitica")
    ap.add_argument("--carpeta", required=True, help="Carpeta con los videos (ej. ..\\VPNV2)")
    ap.add_argument("--fecha", help="Fecha de grabacion AAAA-MM-DD (necesaria para nombres D22N1657A1704)")
    ap.add_argument("--calibracion", default="calibracion.json")
    ap.add_argument("--limite", type=float, default=50.0, help="Limite de velocidad km/h")
    ap.add_argument("--salto", type=int, default=1, help="Procesar 1 de cada N cuadros (2 = el doble de rapido)")
    ap.add_argument("--modelo", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--carriles", type=int, default=None, help="Forzar numero de carriles")
    ap.add_argument("--carriles-def", default=None,
                    help="Carriles dibujados a mano con definir_carriles.py (ej. carriles_nivel2.json)")
    ap.add_argument("--salida", default="salidas")
    ap.add_argument("--nombre", default=None, help="Nombre del reporte combinado (por defecto <carpeta>_completo)")
    ap.add_argument("--con-video", action="store_true", help="Guardar tambien el video anotado de cada clip")
    ap.add_argument("--mostrar", action="store_true", help="Ver la ventana mientras procesa (mas lento)")
    ap.add_argument("--rehacer", action="store_true", help="Volver a procesar aunque ya exista el resultado")
    ap.add_argument("--revisar", action="store_true", help="Solo mostrar horarios y revisar la calibracion")
    ap.add_argument("--solo-analitica", action="store_true", help="No procesar videos: solo juntar resultados")
    args = ap.parse_args()

    carpeta = Path(args.carpeta)
    if not carpeta.is_dir():
        sys.exit(f"No existe la carpeta {carpeta}")
    fecha = datetime.strptime(args.fecha, "%Y-%m-%d") if args.fecha else None
    if not Path(args.calibracion).exists():
        sys.exit(f"No existe '{args.calibracion}'. Copia tu calibracion aqui o crea una nueva:\n"
                 f"  python calibrar.py --video \"{next(iter(sorted(carpeta.iterdir())), '')}\" --modo homografia")
    calib = Calibracion.cargar(args.calibracion)
    if args.carriles_def and not Path(args.carriles_def).exists():
        sys.exit(f"No existe '{args.carriles_def}'. Crealo con definir_carriles.py")

    # ---------- Plan: videos y horarios ----------
    videos = []
    for ruta in sorted(p for p in carpeta.iterdir() if p.suffix.lower() in EXTENSIONES):
        ini, fin = horario_desde_nombre(ruta.name, fecha)
        videos.append({"ruta": ruta, "inicio": ini, "fin": fin, "info": info_video(ruta)})
    if not videos:
        sys.exit(f"No hay videos en {carpeta}")
    videos.sort(key=lambda v: (v["inicio"] is None, v["inicio"] or datetime.max, v["ruta"].name))

    print(f"\n{'VIDEO':<24} {'INICIO':<17} {'FIN (nombre)':<8} {'DURACION':>9} {'RESOLUCION':>11}")
    print("-" * 74)
    problemas = 0
    for v in videos:
        i = v["info"]
        ini = f"{v['inicio']:%Y-%m-%d %H:%M}" if v["inicio"] else "?? (sin hora)"
        fin = f"{v['fin']:%H:%M}" if v["fin"] else "-"
        print(f"{v['ruta'].name:<24} {ini:<17} {fin:<8} {fmt(i['duracion_s']):>9} {i['ancho']:>5}x{i['alto']:<5}")
        if v["inicio"] is None:
            problemas += 1
        elif v["fin"] and abs((v["fin"] - v["inicio"]).total_seconds() - i["duracion_s"]) > 90:
            print(f"   AVISO: el nombre dice {fmt((v['fin'] - v['inicio']).total_seconds())} pero el video dura "
                  f"{fmt(i['duracion_s'])}. Revisa el nombre.")
    total_s = sum(v["info"]["duracion_s"] for v in videos)
    print(f"\n{len(videos)} videos | {fmt(total_s)} de grabacion | calibracion: {calib.descripcion()} "
          f"({calib.resolucion[0]}x{calib.resolucion[1]})")
    if problemas:
        sys.exit("Hay videos sin hora de inicio. Renombralos (ej. D22N1657A1704.mp4) o procesalos a mano "
                 "con radar_velocidad.py --hora-inicio.")

    salida = Path(args.salida)
    if args.revisar:
        rev = revisar_calibracion(videos, calib, salida / "revision_calibracion")
        print(f"\nRevisa las imagenes en: {rev.resolve()}")
        print("  - mosaico.jpg: todos los videos juntos.")
        print("  Los puntos y lineas AMARILLOS de calibracion deben caer EXACTAMENTE sobre las mismas marcas del suelo en todos.")
        print("  Si en alguno esta corrido, la camara se movio: calibra ese grupo de videos por separado.")
        return

    # ---------- Procesar ----------
    carpetas_ok = []
    t_lote = time.perf_counter()
    for n, v in enumerate(videos, 1):
        destino = salida / v["ruta"].stem
        if args.solo_analitica:
            if (destino / "sesion.json").exists():
                carpetas_ok.append(destino)
            else:
                print(f"  (sin resultados aun para {v['ruta'].name}; se omite)")
            continue
        if not args.rehacer and ya_procesado(destino, v["info"]["duracion_s"]):
            print(f"\n[{n}/{len(videos)}] {v['ruta'].name}: ya procesado, se salta (usa --rehacer para repetir)")
            carpetas_ok.append(destino)
            continue
        print(f"\n[{n}/{len(videos)}] {v['ruta'].name}  (inicio {v['inicio']:%H:%M})")
        cmd = [sys.executable, str(AQUI / "radar_velocidad.py"), "--video", str(v["ruta"]),
               "--calibracion", args.calibracion, "--limite", str(args.limite),
               "--hora-inicio", v["inicio"].strftime("%Y-%m-%d %H:%M:%S"),
               "--salida", str(salida), "--salto", str(args.salto),
               "--modelo", args.modelo, "--imgsz", str(args.imgsz)]
        if not args.mostrar:
            cmd.append("--no-mostrar")
        if not args.con_video:
            cmd.append("--sin-video-salida")
        if args.carriles:
            cmd += ["--carriles", str(args.carriles)]
        t0 = time.perf_counter()
        try:
            r = subprocess.run(cmd, cwd=str(AQUI))
        except KeyboardInterrupt:
            print("\nLote detenido. Vuelve a ejecutar el mismo comando para continuar donde quedo.")
            return
        if r.returncode != 0:
            print(f"   ERROR procesando {v['ruta'].name} (codigo {r.returncode}). Se continua con el siguiente.")
            continue
        carpetas_ok.append(destino)
        hecho = sum(x["info"]["duracion_s"] for x in videos[:n])
        transcurrido = time.perf_counter() - t_lote
        if hecho > 0 and n < len(videos):
            restante = transcurrido / hecho * (total_s - hecho)
            print(f"   Listo en {fmt(time.perf_counter() - t0)}. Tiempo restante estimado: ~{fmt(restante)}")

    # ---------- Analítica combinada ----------
    if not carpetas_ok:
        sys.exit("No hay resultados para combinar.")
    nombre = args.nombre or f"{carpeta.name}_completo"
    print(f"\n=== Analitica combinada de {len(carpetas_ok)} videos -> {salida / nombre} ===")
    from radar.analitica import generar_reportes
    generar_reportes(carpetas_ok, salida / nombre, args.carriles, carriles_def=args.carriles_def)
    print(f"\nListo. Abre la carpeta: {(salida / nombre).resolve()}")
    print("  mapa_calor.png | vehiculos_por_hora.png | resumen_nivel2.txt | carriles.csv | vehiculos_contados.csv")


if __name__ == "__main__":
    main()
