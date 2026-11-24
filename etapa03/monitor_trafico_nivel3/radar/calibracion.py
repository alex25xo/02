"""
Calibración métrica de una cámara fija (Nivel 1 - Radar de velocidad).

Convierte coordenadas de la imagen (píxeles) a coordenadas reales (metros)
sobre el plano del suelo. Hay dos modos:

  * "ppm"         Factor de escala único: píxeles por metro (lo que pide el
                  enunciado). Se obtiene marcando 2 puntos en el suelo cuya
                  distancia real se midió en Google Maps.
                  Es correcto cuando la cámara mira la calle DE COSTADO
                  (perpendicular) o DESDE ARRIBA. Con perspectiva fuerte
                  sobrestima/subestima según la distancia a la cámara.

  * "homografia"  Transformación de perspectiva (matriz 3x3) del plano del
                  suelo a una "vista de pájaro" en metros. Se obtiene marcando
                  4 esquinas de un rectángulo del suelo de medidas conocidas
                  (ej. ancho de la calzada x distancia entre dos postes).
                  Corrige la perspectiva: un metro lejos vale lo mismo que un
                  metro cerca.

En ambos casos los puntos SIEMPRE se marcan en el suelo (base de los postes,
líneas pintadas, bordes de vereda), nunca en la parte alta de un objeto.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

MODOS = ("ppm", "homografia")


class Calibracion:
    """Guarda la calibración y transforma puntos de píxeles a metros."""

    def __init__(self, modo: str, resolucion: tuple[int, int], *,
                 ppm: float | None = None,
                 H: np.ndarray | None = None,
                 puntos_px: list | None = None,
                 medidas_m: dict | None = None,
                 zona_px: list | None = None,
                 origen: str = ""):
        if modo not in MODOS:
            raise ValueError(f"Modo desconocido '{modo}'. Usa uno de {MODOS}")
        if modo == "ppm" and (ppm is None or ppm <= 0):
            raise ValueError("El modo 'ppm' necesita un factor ppm > 0")
        if modo == "homografia" and H is None:
            raise ValueError("El modo 'homografia' necesita la matriz H")

        self.modo = modo
        self.resolucion = (int(resolucion[0]), int(resolucion[1]))  # (ancho, alto)
        self.ppm = float(ppm) if ppm is not None else None
        self.H = np.asarray(H, dtype=np.float64) if H is not None else None
        self.puntos_px = [tuple(map(float, p)) for p in (puntos_px or [])]
        self.medidas_m = dict(medidas_m or {})
        self.zona_px = [tuple(map(float, p)) for p in (zona_px or [])] or None
        self.origen = origen
        self._zona_np = (np.array(self.zona_px, dtype=np.float32).reshape(-1, 1, 2)
                         if self.zona_px else None)

    # ------------------------------------------------------------------ #
    # Conversión píxeles -> metros
    # ------------------------------------------------------------------ #
    def a_metros(self, puntos_px) -> np.ndarray:
        """Convierte N puntos (x, y) en píxeles a (X, Y) en metros. Devuelve Nx2."""
        pts = np.asarray(puntos_px, dtype=np.float64).reshape(-1, 2)
        if self.modo == "ppm":
            return pts / self.ppm
        return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), self.H).reshape(-1, 2)

    def a_pixeles(self, puntos_m) -> np.ndarray:
        """Inversa de a_metros: (X, Y) en metros -> (x, y) en píxeles."""
        pts = np.asarray(puntos_m, dtype=np.float64).reshape(-1, 2)
        if self.modo == "ppm":
            return pts * self.ppm
        H_inv = np.linalg.inv(self.H)
        return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H_inv).reshape(-1, 2)

    def visibles(self, puntos_m) -> np.ndarray:
        """
        (Nivel 3) Para puntos del suelo en metros: True si quedan DELANTE de la
        cámara y dentro de la imagen. Un punto detrás de la cámara también tiene
        "píxeles" matemáticos, pero no se debe dibujar.
        """
        pts = np.asarray(puntos_m, dtype=np.float64).reshape(-1, 2)
        W, H = self.resolucion
        if self.modo == "ppm":
            px, ok = pts * self.ppm, np.ones(len(pts), bool)
        else:
            H_inv = np.linalg.inv(self.H)
            q = np.c_[pts, np.ones(len(pts))] @ H_inv.T
            ref = self.a_metros(self.puntos_px[:1])[0]
            signo = np.sign((H_inv @ np.array([ref[0], ref[1], 1.0]))[2])
            ok = q[:, 2] * signo > 1e-12
            px = q[:, :2] / np.where(ok, q[:, 2], 1.0)[:, None]
        return ok & (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)

    # ------------------------------------------------------------------ #
    # Ejes de la vía (Nivel 2): s = a lo largo de la calle, d = a lo ancho
    # ------------------------------------------------------------------ #
    def ejes_via(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Devuelve (origen, u, n) en metros:
          u = dirección A LO LARGO de la calle (sentido del tráfico)
          n = dirección A LO ANCHO (perpendicular): sirve para saber el carril
        PPM: u va de P1 a P2 (se marcan en el sentido del tráfico).
        Homografía: P2->P3 es el LARGO (eje Y) y P1->P2 el ANCHO (eje X).
        """
        if self.modo == "ppm":
            p1, p2 = self.a_metros(self.puntos_px[:2])
            u = (p2 - p1) / np.linalg.norm(p2 - p1)
            return p1, u, np.array([-u[1], u[0]])
        return np.zeros(2), np.array([0.0, 1.0]), np.array([1.0, 0.0])

    def a_via(self, puntos_px) -> np.ndarray:
        """Píxeles -> coordenadas de la vía (s, d) en metros. Devuelve Nx2."""
        origen, u, n = self.ejes_via()
        r = self.a_metros(puntos_px) - origen
        return np.stack([r @ u, r @ n], axis=1)

    def desde_via(self, sd) -> np.ndarray:
        """Coordenadas de la vía (s, d) en metros -> píxeles. Devuelve Nx2."""
        origen, u, n = self.ejes_via()
        sd = np.asarray(sd, dtype=np.float64).reshape(-1, 2)
        return self.a_pixeles(origen + sd[:, :1] * u + sd[:, 1:] * n)

    def distancia_m(self, p1, p2) -> float:
        """Distancia real (m) entre dos puntos de la imagen que están en el suelo."""
        a, b = self.a_metros([p1, p2])
        return float(np.hypot(*(b - a)))

    def en_zona(self, punto_px) -> bool:
        """True si el punto está dentro de la zona de medición (o si no hay zona)."""
        if self._zona_np is None:
            return True
        x, y = float(punto_px[0]), float(punto_px[1])
        return cv2.pointPolygonTest(self._zona_np, (x, y), False) >= 0

    # ------------------------------------------------------------------ #
    # Cambio de resolución (ej. calibraste en 1080p y procesas en 720p)
    # ------------------------------------------------------------------ #
    def ajustar_resolucion(self, ancho: int, alto: int) -> "Calibracion":
        w0, h0 = self.resolucion
        if (ancho, alto) == (w0, h0):
            return self
        sx, sy = ancho / w0, alto / h0
        if abs(sx - sy) > 0.01:
            raise ValueError(
                f"La relación de aspecto cambió ({w0}x{h0} -> {ancho}x{alto}). "
                "Vuelve a calibrar con un cuadro de este mismo video.")
        escalar = lambda pts: [(x * sx, y * sy) for x, y in pts] if pts else pts
        H = None
        if self.H is not None:
            S_inv = np.diag([1 / sx, 1 / sy, 1.0])
            H = self.H @ S_inv
        return Calibracion(self.modo, (ancho, alto),
                           ppm=self.ppm * sx if self.ppm else None, H=H,
                           puntos_px=escalar(self.puntos_px),
                           medidas_m=self.medidas_m,
                           zona_px=escalar(self.zona_px), origen=self.origen)

    # ------------------------------------------------------------------ #
    # Guardar / cargar
    # ------------------------------------------------------------------ #
    def a_dict(self) -> dict:
        return {
            "modo": self.modo,
            "resolucion": list(self.resolucion),
            "ppm": self.ppm,
            "H": self.H.tolist() if self.H is not None else None,
            "puntos_px": [list(p) for p in self.puntos_px],
            "medidas_m": self.medidas_m,
            "zona_px": [list(p) for p in self.zona_px] if self.zona_px else None,
            "origen": self.origen,
            "fecha": datetime.now().isoformat(timespec="seconds"),
        }

    def guardar(self, ruta: str | Path) -> None:
        Path(ruta).write_text(json.dumps(self.a_dict(), indent=2, ensure_ascii=False),
                              encoding="utf-8")

    @staticmethod
    def cargar(ruta: str | Path) -> "Calibracion":
        d = json.loads(Path(ruta).read_text(encoding="utf-8"))
        return Calibracion(d["modo"], tuple(d["resolucion"]), ppm=d.get("ppm"),
                           H=d.get("H"), puntos_px=d.get("puntos_px"),
                           medidas_m=d.get("medidas_m"), zona_px=d.get("zona_px"),
                           origen=d.get("origen", ""))

    def descripcion(self) -> str:
        if self.modo == "ppm":
            return f"PPM = {self.ppm:.2f} px/m"
        m = self.medidas_m
        return f"Homografia {m.get('ancho', '?')} x {m.get('largo', '?')} m"


# ---------------------------------------------------------------------- #
# Constructores
# ---------------------------------------------------------------------- #
def zona_sin_bordes(resolucion, fraccion: float = 0.05) -> list[tuple[float, float]]:
    """
    Rectángulo de la imagen sin un margen en cada borde. Los vehículos que están
    entrando o saliendo de cuadro tienen la caja recortada y su punto de medición
    se mueve más lento que el auto real, así que no se miden ahí.
    """
    w, h = resolucion
    mx, my = w * fraccion, h * fraccion
    return [(mx, my), (w - mx, my), (w - mx, h - my), (mx, h - my)]


def calibrar_ppm(p1, p2, metros: float, resolucion, zona_px=None, origen="") -> Calibracion:
    """
    Factor de escala a partir de 2 puntos del suelo separados `metros` metros.
    Sin zona explícita se mide en toda la imagen menos un 5 % en cada borde.
    """
    if metros <= 0:
        raise ValueError("La distancia real debe ser > 0 m")
    d_px = math.dist(p1, p2)
    if d_px < 10:
        raise ValueError("Los dos puntos están demasiado juntos (< 10 px)")
    zona_px = zona_px or zona_sin_bordes(resolucion)
    return Calibracion("ppm", resolucion, ppm=d_px / metros, puntos_px=[p1, p2],
                       medidas_m={"distancia": metros}, zona_px=zona_px, origen=origen)


def calibrar_homografia(puntos4, ancho_m: float, largo_m: float, resolucion,
                        zona_px=None, origen="") -> Calibracion:
    """
    Homografía suelo->metros a partir de 4 esquinas de un rectángulo real.

    Orden de los puntos (recorriendo el perímetro):
        P1 -> P2 : lado de ANCHO (a lo ancho de la calle, el más cercano a la cámara)
        P2 -> P3 : lado de LARGO (a lo largo de la calle)
        P3 -> P4 : ancho lejano
        P4 -> P1 : largo del otro lado
    """
    if len(puntos4) != 4:
        raise ValueError("La homografía necesita exactamente 4 puntos")
    if ancho_m <= 0 or largo_m <= 0:
        raise ValueError("Las medidas reales deben ser > 0 m")
    src = np.array(puntos4, dtype=np.float32)
    dst = np.array([[0, 0], [ancho_m, 0], [ancho_m, largo_m], [0, largo_m]], dtype=np.float32)
    # Validación geométrica: polígono convexo y con área razonable
    if not cv2.isContourConvex(src.reshape(-1, 1, 2)):
        raise ValueError("Los 4 puntos no forman un cuadrilátero convexo. "
                         "Márcalos en orden recorriendo el perímetro (sin cruzarse).")
    if cv2.contourArea(src) < 500:
        raise ValueError("El cuadrilátero es demasiado pequeño en la imagen")
    H = cv2.getPerspectiveTransform(src, dst)
    zona = zona_px if zona_px else [tuple(p) for p in puntos4]
    return Calibracion("homografia", resolucion, H=H, puntos_px=[tuple(p) for p in puntos4],
                       medidas_m={"ancho": ancho_m, "largo": largo_m},
                       zona_px=zona, origen=origen)


def vista_pajaro(frame: np.ndarray, calib: Calibracion, px_por_m: float = 20.0,
                 margen_m: float = 4.0) -> np.ndarray | None:
    """
    Imagen "desde arriba" del rectángulo de calibración (solo homografía).
    Sirve para verificar: las líneas de carril deben verse paralelas y rectas.
    """
    if calib.modo != "homografia":
        return None
    ancho = calib.medidas_m["ancho"]
    largo = calib.medidas_m["largo"]
    s = px_por_m
    # metros -> lienzo (el lado lejano queda arriba)
    M = np.array([[s, 0, margen_m * s],
                  [0, -s, (largo + margen_m) * s],
                  [0, 0, 1]], dtype=np.float64)
    W = int((ancho + 2 * margen_m) * s)
    Hh = int((largo + 2 * margen_m) * s)
    if W * Hh > 4000 * 4000:  # evita imágenes gigantes
        return vista_pajaro(frame, calib, px_por_m=px_por_m / 2, margen_m=margen_m)
    out = cv2.warpPerspective(frame, M @ calib.H, (W, Hh))
    # rectángulo de calibración y cuadrícula cada 5 m
    p0 = (int(margen_m * s), int(margen_m * s))
    p1 = (int((ancho + margen_m) * s), int((largo + margen_m) * s))
    for y_m in np.arange(0, largo + 0.01, 5):
        y = int((largo + margen_m - y_m) * s)
        cv2.line(out, (0, y), (W, y), (80, 80, 80), 1)
        cv2.putText(out, f"{y_m:.0f} m", (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(out, p0, p1, (0, 255, 255), 2)
    return out
