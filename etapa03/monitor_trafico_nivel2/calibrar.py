"""
HERRAMIENTA DE CALIBRACION - Nivel 1 (Radar de velocidad)

Marca sobre un cuadro del video los puntos del suelo cuya distancia real
mediste en Google Maps, y genera `calibracion.json`.

Uso interactivo (recomendado):
    python calibrar.py --video mi_video.mp4 --modo ppm
    python calibrar.py --video mi_video.mp4 --modo homografia

Uso sin ventana (repetible, con puntos ya conocidos):
    python calibrar.py --video mi_video.mp4 --modo ppm --puntos "100,600;900,610" --metros 25
    python calibrar.py --video mi_video.mp4 --modo homografia \
        --puntos "420,700;980,700;800,380;560,380" --ancho 7 --largo 30

Controles en la ventana:
    Clic izquierdo  agregar punto            Clic derecho / Z   deshacer punto
    A / D           retroceder / avanzar 1 s (elegir un cuadro sin autos encima)
    ENTER           confirmar fase           R                  reiniciar fase
    Q / ESC         salir
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from radar.calibracion import Calibracion, calibrar_homografia, calibrar_ppm, vista_pajaro
from radar.dibujo import ajustar_para_pantalla, etiqueta, ventana_cerrada
from radar.video import abrir, fps_confiable, resolver_fuente

VENTANA = "Calibracion - Radar de velocidad"
AMARILLO, CIAN, MAGENTA, VERDE = (0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0)

NOMBRES_HOMO = [
    "P1: primera esquina del rectangulo",
    "P2: P1->P2 es el ANCHO (cruzando la calle)",
    "P3: P2->P3 es el LARGO (a lo largo de la calle)",
    "P4: cierra el rectangulo (frente a P3, del lado de P1)",
]
USAR_DIALOGO = True     # pedir las medidas en una ventanita (si falla, en la terminal)


# ---------------------------------------------------------------------- #
def parsear_puntos(texto: str) -> list[tuple[float, float]]:
    pts = []
    for par in texto.replace(" ", "").split(";"):
        if par:
            x, y = par.split(",")
            pts.append((float(x), float(y)))
    return pts


def _dialogo_numero(mensaje: str) -> float | None:
    """Ventana emergente para escribir la medida. None si el usuario cancela."""
    import tkinter as tk
    from tkinter import messagebox, simpledialog
    raiz = tk.Tk()
    raiz.withdraw()
    raiz.attributes("-topmost", True)
    try:
        while True:
            s = simpledialog.askstring("Medida real (Google Maps)", mensaje, parent=raiz)
            if s is None:
                return None
            try:
                v = float(s.strip().replace(",", "."))
                if v > 0:
                    return v
            except ValueError:
                pass
            messagebox.showwarning("Valor no valido", "Escribe un numero mayor que 0 (ej. 12.5)",
                                   parent=raiz)
    finally:
        raiz.destroy()


def pedir_numero(mensaje: str) -> float:
    """Pide un número > 0: primero con ventana emergente, si no se puede, en la terminal."""
    if USAR_DIALOGO:
        try:
            v = _dialogo_numero(mensaje.strip())
        except Exception:          # sin tkinter o sin entorno gráfico -> terminal
            v = -1.0
        if v is None:
            raise ValueError("Medida cancelada. Pulsa ENTER otra vez para escribirla.")
        if v > 0:
            print(f"{mensaje.strip()} {v}")
            return v
    while True:
        try:
            v = float(input(mensaje).strip().replace(",", "."))
            if v > 0:
                return v
        except ValueError:
            pass
        except EOFError:
            sys.exit("\nCancelado.")
        print("  -> Escribe un número mayor que 0 (ej. 12.5)")


def leer_cuadro(cap, indice: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, indice))
    ok, frame = cap.read()
    return frame if ok else None


def lupa(frame, x, y, radio=20, tam=160):
    """Recorte ampliado alrededor del cursor para marcar con precisión."""
    h, w = frame.shape[:2]
    x, y = int(x), int(y)
    x0, y0 = max(0, x - radio), max(0, y - radio)
    x1, y1 = min(w, x + radio), min(h, y + radio)
    rec = frame[y0:y1, x0:x1]
    if rec.size == 0:
        return None
    rec = cv2.copyMakeBorder(rec, radio - (y - y0), radio - (y1 - y), radio - (x - x0),
                             radio - (x1 - x), cv2.BORDER_CONSTANT)
    rec = cv2.resize(rec, (tam, tam), interpolation=cv2.INTER_NEAREST)
    c = tam // 2
    cv2.line(rec, (c, 0), (c, tam), (0, 0, 255), 1)
    cv2.line(rec, (0, c), (tam, c), (0, 0, 255), 1)
    cv2.rectangle(rec, (0, 0), (tam - 1, tam - 1), (255, 255, 255), 1)
    return rec


def dibujar_preview(frame, calib: Calibracion, verif=()):
    """Imagen con la calibración dibujada (para guardar/mostrar)."""
    img = frame.copy()
    g = max(2, img.shape[1] // 640)
    if calib.zona_px:
        pts = np.array(calib.zona_px, np.int32).reshape(-1, 1, 2)
        capa = img.copy()
        cv2.fillPoly(capa, [pts], (255, 200, 0))
        cv2.addWeighted(capa, 0.1, img, 0.9, 0, dst=img)
        cv2.polylines(img, [pts], True, (255, 200, 0), max(1, g // 2), cv2.LINE_AA)
        etiqueta(img, "zona de medicion", int(pts[:, 0, 0].min()) + 4, int(pts[:, 0, 1].min()) + 22,
                 (120, 90, 0), 0.5)
    pts = [tuple(map(int, p)) for p in calib.puntos_px]
    cerrar = calib.modo == "homografia"
    cv2.polylines(img, [np.array(pts, np.int32).reshape(-1, 1, 2)], cerrar, AMARILLO, g, cv2.LINE_AA)
    for i, p in enumerate(pts):
        cv2.circle(img, p, g + 3, AMARILLO, -1, cv2.LINE_AA)
        etiqueta(img, f"P{i + 1}", p[0] + 8, p[1] - 8, (0, 120, 120), 0.6)
    m = calib.medidas_m
    lados = ([(0, 1, m.get("distancia"))] if calib.modo == "ppm"
             else [(0, 1, m.get("ancho")), (1, 2, m.get("largo"))])
    for a, b, d in lados:
        cx, cy = (pts[a][0] + pts[b][0]) // 2, (pts[a][1] + pts[b][1]) // 2
        etiqueta(img, f"{d:g} m", cx - 20, cy - 10, (0, 90, 90), 0.6)
    etiqueta(img, calib.descripcion(), 10, img.shape[0] - 12, (40, 40, 40), 0.7, 2)
    for a, b in verif:
        a_i, b_i = tuple(map(int, a)), tuple(map(int, b))
        cv2.line(img, a_i, b_i, MAGENTA, g, cv2.LINE_AA)
        d = calib.distancia_m(a, b)
        etiqueta(img, f"{d:.2f} m", (a_i[0] + b_i[0]) // 2, (a_i[1] + b_i[1]) // 2 - 6,
                 (120, 0, 120), 0.6)
    return img


# ---------------------------------------------------------------------- #
class CalibradorInteractivo:
    def __init__(self, cap, frame, idx, fps, es_webcam, modo):
        self.cap, self.frame, self.idx, self.fps = cap, frame, idx, fps
        self.es_webcam, self.modo = es_webcam, modo
        self.puntos: list[tuple[float, float]] = []
        self.zona: list[tuple[float, float]] = []
        self.verif: list[tuple] = []
        self.pendiente = None          # primer punto de una medición de verificación
        self.fase = "calib"            # calib -> zona -> verificar
        self.calib: Calibracion | None = None
        self.raton = (0, 0)
        self.escala = 1.0

    # --- ratón ---------------------------------------------------------- #
    def on_mouse(self, evento, x, y, flags, _):
        X, Y = x / self.escala, y / self.escala
        self.raton = (X, Y)
        if evento == cv2.EVENT_LBUTTONDOWN:
            if self.fase == "calib":
                n = 2 if self.modo == "ppm" else 4
                if len(self.puntos) < n:
                    self.puntos.append((X, Y))
            elif self.fase == "zona":
                self.zona.append((X, Y))
            elif self.fase == "verificar":
                if self.pendiente is None:
                    self.pendiente = (X, Y)
                else:
                    a, b = self.pendiente, (X, Y)
                    self.verif.append((a, b))
                    self.pendiente = None
                    print(f"  Verificacion: {self.calib.distancia_m(a, b):.2f} m "
                          f"({np.hypot(b[0] - a[0], b[1] - a[1]):.0f} px)")
        elif evento == cv2.EVENT_RBUTTONDOWN:
            self.deshacer()

    def deshacer(self):
        lista = {"calib": self.puntos, "zona": self.zona}.get(self.fase)
        if lista:
            lista.pop()
        elif self.fase == "verificar" and self.verif:
            self.verif.pop()

    # --- render --------------------------------------------------------- #
    def instrucciones(self) -> list[str]:
        if self.fase == "calib":
            if self.modo == "ppm":
                if len(self.puntos) < 2:
                    return [f"FASE 1/3 - PPM: marca el punto {len(self.puntos) + 1} de 2 EN EL SUELO",
                            "Ej: base de 2 postes, extremos de un paso de cebra (en el sentido de los autos)",
                            "A/D: cambiar cuadro   Clic der/Z: deshacer"]
                return ["FASE 1/3 - Listo. Pulsa ENTER y escribe la distancia real (m)"]
            if len(self.puntos) < 4:
                return [f"FASE 1/3 - HOMOGRAFIA: marca {NOMBRES_HOMO[len(self.puntos)]}",
                        "Las 4 esquinas de un rectangulo del SUELO (ancho x largo conocidos)",
                        "A/D: cambiar cuadro   Clic der/Z: deshacer"]
            return ["FASE 1/3 - Listo. Pulsa ENTER y escribe ANCHO y LARGO (m)"]
        if self.fase == "zona":
            por_defecto = ("imagen sin bordes" if self.modo == "ppm" else "el rectangulo de calibracion")
            return ["FASE 2/3 - ZONA DE MEDICION (opcional): clics para dibujar un poligono",
                    f"ENTER = terminar (sin puntos = {por_defecto})"]
        return ["FASE 3/3 - VERIFICACION: haz 2 clics sobre otra distancia conocida",
                "Compara con Google Maps. Error < 5% = buena calibracion",
                "C: borrar mediciones    ENTER/Q: terminar"]

    def render(self):
        img = self.frame.copy()
        g = max(2, img.shape[1] // 640)
        if self.calib is not None:
            img = dibujar_preview(img, self.calib, self.verif)
        else:
            pts = [tuple(map(int, p)) for p in self.puntos]
            if len(pts) > 1:
                cv2.polylines(img, [np.array(pts, np.int32).reshape(-1, 1, 2)],
                              self.modo == "homografia" and len(pts) == 4, AMARILLO, g, cv2.LINE_AA)
            for i, p in enumerate(pts):
                cv2.circle(img, p, g + 3, AMARILLO, -1, cv2.LINE_AA)
                etiqueta(img, f"P{i + 1}", p[0] + 8, p[1] - 8, (0, 120, 120), 0.6)
        if self.fase == "zona" and self.zona:
            pts = np.array(self.zona, np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], len(self.zona) > 2, CIAN, g, cv2.LINE_AA)
            for p in self.zona:
                cv2.circle(img, tuple(map(int, p)), g + 2, CIAN, -1)
        if self.pendiente is not None:
            cv2.circle(img, tuple(map(int, self.pendiente)), g + 3, MAGENTA, -1)

        # Videos pequenos se amplian (hasta 2x) para marcar con mas precision
        vis, self.escala = ajustar_para_pantalla(img, ampliar_max=2.0)
        lineas = self.instrucciones()
        y = 26
        for linea in lineas:
            etiqueta(vis, linea, 8, y, (30, 30, 30), 0.55)
            y += 26
        if not self.es_webcam:
            etiqueta(vis, f"t = {self.idx / self.fps:.1f} s (cuadro {self.idx})", 8,
                     vis.shape[0] - 10, (30, 30, 30), 0.5)
        zoom = lupa(self.frame, *self.raton)
        alto_v, ancho_v = vis.shape[:2]
        if zoom is not None and ancho_v > 400 and alto_v > 26 * len(lineas) + 200:
            # La lupa va en la esquina opuesta al cursor para no tapar donde haces clic
            mx, my = self.raton[0] * self.escala, self.raton[1] * self.escala
            t = zoom.shape[0]
            x0 = 10 if mx > ancho_v / 2 else ancho_v - t - 10
            y0 = alto_v - t - 30 if my < alto_v / 2 else 26 * len(lineas) + 10
            vis[y0:y0 + t, x0:x0 + t] = zoom
        return vis

    # --- bucle ---------------------------------------------------------- #
    def ejecutar(self, args) -> Calibracion | None:
        cv2.namedWindow(VENTANA, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(VENTANA, self.on_mouse)
        alto, ancho = self.frame.shape[:2]
        while True:
            cv2.imshow(VENTANA, self.render())
            k = cv2.waitKey(20) & 0xFF
            if ventana_cerrada(VENTANA):           # cerraron la ventana con la X
                k = 27
            if k in (27, ord("q")):
                if self.fase == "verificar":
                    break
                print("Calibracion cancelada.")
                return None
            if k in (ord("z"), 8):
                self.deshacer()
            elif k == ord("r"):
                {"calib": self.puntos, "zona": self.zona, "verificar": self.verif}[self.fase].clear()
            elif k == ord("c") and self.fase == "verificar":
                self.verif.clear()
            elif k in (ord("a"), ord("d")) and self.fase == "calib" and not self.es_webcam:
                nuevo = self.idx + (1 if k == ord("d") else -1) * int(round(self.fps))
                fr = leer_cuadro(self.cap, nuevo)
                if fr is not None:
                    self.frame, self.idx = fr, max(0, nuevo)
            elif k in (13, 10):
                if self.fase == "calib":
                    n = 2 if self.modo == "ppm" else 4
                    if len(self.puntos) < n:
                        continue
                    self._mostrar_aviso("Escribe la medida en la ventanita (o en la TERMINAL)")
                    try:
                        self.calib = construir(self.modo, self.puntos, args, (ancho, alto))
                    except ValueError as e:
                        print(f"AVISO: {e}  (Z = deshacer punto, R = reiniciar)")
                        continue
                    self.fase = "zona"
                elif self.fase == "zona":
                    if 0 < len(self.zona) < 3:
                        print("La zona necesita al menos 3 puntos (o ninguno).")
                        continue
                    if self.zona:
                        self.calib = construir(self.modo, self.puntos, args, (ancho, alto),
                                               zona=self.zona, medidas=self.calib.medidas_m)
                    self.fase = "verificar"
                    print("\nCalibracion lista:", self.calib.descripcion())
                    print("FASE 3: verifica con otra distancia conocida (o ENTER para terminar).")
                    if self.calib.modo == "homografia":
                        bev = vista_pajaro(self.frame, self.calib)
                        if bev is not None:
                            cv2.imshow("Vista de pajaro (verificacion)", ajustar_para_pantalla(bev, 700, 700)[0])
                else:
                    break
        cv2.destroyAllWindows()
        return self.calib

    def _mostrar_aviso(self, texto):
        vis = self.render()
        etiqueta(vis, texto, 8, vis.shape[0] // 2, (0, 0, 180), 0.9, 2)
        cv2.imshow(VENTANA, vis)
        cv2.waitKey(50)


def construir(modo, puntos, args, resolucion, zona=None, medidas=None) -> Calibracion:
    """Crea la calibración pidiendo las medidas reales si no vinieron por argumentos."""
    origen = str(args.video)
    if modo == "ppm":
        metros = (medidas or {}).get("distancia") or args.metros or pedir_numero(
            "\nDistancia REAL entre P1 y P2 en metros (medida en Google Maps): ")
        c = calibrar_ppm(puntos[0], puntos[1], metros, resolucion, zona_px=zona, origen=origen)
        print(f"  -> {np.hypot(*np.subtract(puntos[1], puntos[0])):.1f} px / {metros} m "
              f"= {c.ppm:.2f} px/m")
        return c
    ancho = (medidas or {}).get("ancho") or args.ancho or pedir_numero(
        "\nANCHO real P1-P2 en metros (ej. ancho de la calzada): ")
    largo = (medidas or {}).get("largo") or args.largo or pedir_numero(
        "LARGO real P2-P3 en metros (ej. distancia entre 2 postes): ")
    return calibrar_homografia(puntos, ancho, largo, resolucion, zona_px=zona, origen=origen)


# ---------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Calibracion metrica para el radar de velocidad")
    ap.add_argument("--video", help="Ruta del video o indice de webcam (0). Sin valor: abre explorador")
    ap.add_argument("--modo", choices=["ppm", "homografia"], default="ppm")
    ap.add_argument("--frame", type=int, default=0, help="Cuadro inicial a mostrar")
    ap.add_argument("--salida", default="calibracion.json")
    ap.add_argument("--puntos", help='Puntos sin ventana: "x1,y1;x2,y2[;x3,y3;x4,y4]"')
    ap.add_argument("--zona", help='Poligono de la zona de medicion: "x,y;x,y;x,y;..."')
    ap.add_argument("--metros", type=float, help="(ppm) Distancia real P1-P2 en metros")
    ap.add_argument("--ancho", type=float, help="(homografia) Ancho real P1-P2 en metros")
    ap.add_argument("--largo", type=float, help="(homografia) Largo real P2-P3 en metros")
    args = ap.parse_args()

    fuente, es_webcam = resolver_fuente(args.video)
    args.video = fuente
    cap = abrir(fuente)
    fps, _ = fps_confiable(cap)
    frame = leer_cuadro(cap, args.frame) if not es_webcam else cap.read()[1]
    if frame is None:
        sys.exit("No se pudo leer un cuadro del video.")
    alto, ancho = frame.shape[:2]
    print(f"Video: {fuente}  |  {ancho}x{alto}  |  {fps:.2f} fps  |  modo: {args.modo}")

    if args.puntos:   # ---------- modo sin ventana ----------
        puntos = parsear_puntos(args.puntos)
        zona = parsear_puntos(args.zona) if args.zona else None
        n = 2 if args.modo == "ppm" else 4
        if len(puntos) != n:
            sys.exit(f"El modo {args.modo} necesita {n} puntos y diste {len(puntos)}.")
        calib = construir(args.modo, puntos, args, (ancho, alto), zona=zona)
    else:             # ---------- modo interactivo ----------
        herramienta = CalibradorInteractivo(cap, frame, args.frame, fps, es_webcam, args.modo)
        calib = herramienta.ejecutar(args)
        if calib is None:
            return
        frame = herramienta.frame          # el cuadro elegido con A/D

    cap.release()
    salida = Path(args.salida)
    calib.guardar(salida)
    prev = salida.with_name(salida.stem + "_preview.jpg")
    cv2.imwrite(str(prev), dibujar_preview(frame, calib))
    bev = vista_pajaro(frame, calib)
    if bev is not None:
        cv2.imwrite(str(salida.with_name(salida.stem + "_vista_pajaro.jpg")), bev)
    print(f"\nCalibracion guardada en: {salida.resolve()}")
    print(f"Vista previa: {prev.resolve()}")
    print(f"Resumen: {calib.descripcion()}"
          + (f" | zona de medicion con {len(calib.zona_px)} puntos" if calib.zona_px else ""))


if __name__ == "__main__":
    main()
