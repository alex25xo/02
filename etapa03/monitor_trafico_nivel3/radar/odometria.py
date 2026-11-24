"""
NIVEL 3 - Odometría visual sobre el plano del suelo (cámara en movimiento).

Problema
--------
Con la cámara en movimiento, todo lo que está quieto (suelo, paredes, autos
estacionados) se desplaza en la imagen. El radar del Nivel 1 mide movimiento en
la imagen, así que cree que todo se mueve a la velocidad de la cámara.

Solución: estimar el movimiento de la cámara y restarlo
-------------------------------------------------------
1. FAST encuentra esquinas en el suelo y BRIEF (rotado = ORB) describe cada una
   con 256 bits. Se reparten con una rejilla para no amontonarlas.
2. Se emparejan con el cuadro anterior por distancia de Hamming + test de Lowe.
3. Los puntos emparejados se pasan a METROS sobre el suelo con la calibración.
   Como el celular va a altura y ángulo fijos, la calibración es la misma en
   todos los cuadros: el suelo visto "desde arriba" solo se traslada y rota.
4. RANSAC busca el movimiento rígido 2D (giro + traslación) que explica la
   mayoría de puntos; los que no encajan (autos o personas que se mueven, puntos
   de paredes) quedan fuera. Luego se refina con Kabsch (SVD) sobre los inliers.
5. Encadenando esos movimientos se obtiene la pose de la cámara en un sistema
   de coordenadas fijo al MUNDO (el del primer cuadro): trayectoria + velocidad.
6. Cualquier punto del suelo visto en el cuadro k se lleva al mundo con esa pose.
   Un auto estacionado queda siempre en el mismo lugar del mundo -> 0 km/h.

¿Y la escala? Con una sola cámara, la matriz esencial (5 puntos) da la
dirección del movimiento pero NO cuántos metros. Aquí la escala sale del plano
del suelo calibrado: sabemos cuántos metros mide el rectángulo marcado.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .calibracion import Calibracion


def rigido_2d(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch 2D: R (2x2) y t (2,) que minimizan |R·src + t - dst|²."""
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    D = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0])
    R = Vt.T @ D @ U.T
    return R, cd - R @ cs


def matriz(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(3)
    M[:2, :2], M[:2, 2] = R, t
    return M


def aplicar(M: np.ndarray, pts) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return pts @ M[:2, :2].T + M[:2, 2]


@dataclass
class ResultadoVO:
    estado: str                       # "inicio" | "ok" | "perdida"
    t: float
    pose: np.ndarray                  # 3x3: coordenadas locales del cuadro -> mundo
    v_camara_kmh: float = 0.0         # velocidad (suavizada) de la cámara
    v_camara_bruta_kmh: float = 0.0
    giro_deg: float = 0.0             # orientación acumulada (yaw)
    v_suelo_aparente_kmh: float = 0.0     # lo que "ve" el radar del Nivel 1
    v_suelo_compensada_kmh: float = 0.0   # tras restar el movimiento (debe ser ~0)
    n_puntos: int = 0
    n_parejas: int = 0
    n_inliers: int = 0
    px_prev: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    px_actual: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    inliers: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))


class OdometriaVisual:
    """
    Parámetros
    ----------
    n_puntos      : máximo de puntos ORB por cuadro.
    celda_px      : tamaño de la rejilla para repartir los puntos.
    ratio         : test de Lowe (mejor pareja / segunda mejor).
    ransac_m      : tolerancia de RANSAC en METROS sobre el suelo.
    min_inliers   : menos inliers que esto = estimación no confiable.
    alpha         : suavizado de la velocidad de la cámara.
    """

    def __init__(self, calib: Calibracion, n_puntos: int = 1500, celda_px: int = 40,
                 ratio: float = 0.8, ransac_m: float = 0.12, min_inliers: int = 20,
                 alpha: float = 0.15, max_perdidos: int = 8):
        self.calib = calib
        self.n_puntos = n_puntos
        self.celda = celda_px
        self.ratio = ratio
        self.ransac_m = ransac_m
        self.min_inliers = min_inliers
        self.alpha = alpha
        self.max_perdidos = max_perdidos
        # FAST (detector) + BRIEF orientado (descriptor) = ORB
        self.orb = cv2.ORB_create(nfeatures=n_puntos * 3, scaleFactor=1.2, nlevels=8,
                                  edgeThreshold=15, patchSize=31, fastThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        W, H = calib.resolucion
        # Punto de referencia de la cámara: el suelo en el centro inferior de la imagen
        self.ref_local = calib.a_metros([(W / 2, H * 0.97)])[0]
        self._mascara_suelo = self._crear_mascara_suelo(W, H)
        self.pose = np.eye(3)
        self._prev = None                     # (puntos_px, descriptores, t)
        self._M_ultimo = np.eye(3)
        self._perdidos = 0
        self._v_suave = None
        self.trayectoria: list[tuple[float, float, float]] = []   # (t, x, y) en el mundo
        self.historial: list[ResultadoVO] = []

    # ------------------------------------------------------------------ #
    def _crear_mascara_suelo(self, W, H) -> np.ndarray:
        """Zona de la calibración = región del suelo. Sin zona: mitad inferior."""
        m = np.zeros((H, W), np.uint8)
        if self.calib.zona_px:
            cv2.fillPoly(m, [np.array(self.calib.zona_px, np.int32)], 255)
        else:
            m[H // 2:] = 255
        return m

    def mascara(self, cajas_excluir=()) -> np.ndarray:
        """Suelo menos las cajas de objetos que pueden moverse (autos, personas)."""
        m = self._mascara_suelo.copy()
        H, W = m.shape
        for x1, y1, x2, y2 in cajas_excluir:
            dx, dy = 0.08 * (x2 - x1), 0.08 * (y2 - y1)
            cv2.rectangle(m, (int(max(0, x1 - dx)), int(max(0, y1 - dy))),
                          (int(min(W - 1, x2 + dx)), int(min(H - 1, y2 + dy))), 0, -1)
        return m

    def _detectar(self, gris, mascara):
        kps = self.orb.detect(gris, mascara)
        if not kps:
            return np.zeros((0, 2), np.float32), None
        # Rejilla: como máximo k puntos (los más fuertes) por celda -> bien repartidos
        H, W = gris.shape
        por_celda = max(2, int(self.n_puntos / max(1, (W // self.celda) * (H // self.celda)) * 3))
        cuenta: dict[tuple[int, int], int] = {}
        elegidos = []
        for kp in sorted(kps, key=lambda k: -k.response):
            c = (int(kp.pt[0]) // self.celda, int(kp.pt[1]) // self.celda)
            if cuenta.get(c, 0) < por_celda:
                cuenta[c] = cuenta.get(c, 0) + 1
                elegidos.append(kp)
                if len(elegidos) >= self.n_puntos:
                    break
        elegidos, desc = self.orb.compute(gris, elegidos)
        if desc is None or not elegidos:
            return np.zeros((0, 2), np.float32), None
        return np.array([k.pt for k in elegidos], np.float32), desc

    def _emparejar(self, d_actual, d_prev):
        if d_actual is None or d_prev is None or len(d_actual) < 2 or len(d_prev) < 2:
            return []
        buenos = []
        for par in self.matcher.knnMatch(d_actual, d_prev, k=2):
            if len(par) == 2 and par[0].distance < self.ratio * par[1].distance:
                buenos.append(par[0])
        return buenos

    # ------------------------------------------------------------------ #
    def procesar(self, gris: np.ndarray, t: float, cajas_excluir=()) -> ResultadoVO:
        pts, desc = self._detectar(gris, self.mascara(cajas_excluir))

        if self._prev is None:
            self._prev = (pts, desc, t)
            self._registrar_pose(t)
            r = ResultadoVO("inicio", t, self.pose.copy(), n_puntos=len(pts))
            self.historial.append(r)
            return r

        pts_p, desc_p, t_p = self._prev
        dt = max(1e-6, t - t_p)
        parejas = self._emparejar(desc, desc_p)
        estado = "perdida"
        M = self._M_ultimo                          # modelo de velocidad constante
        px_a = np.array([pts[m.queryIdx] for m in parejas], np.float64).reshape(-1, 2)
        px_p = np.array([pts_p[m.trainIdx] for m in parejas], np.float64).reshape(-1, 2)
        inl = np.zeros(len(parejas), bool)
        v_ap = v_comp = 0.0

        if len(parejas) >= self.min_inliers:
            b = self.calib.a_metros(px_a)           # suelo en metros, cuadro actual
            a = self.calib.a_metros(px_p)           # suelo en metros, cuadro anterior
            razonable = (np.abs(b).max(axis=1) < 200) & (np.abs(a).max(axis=1) < 200)
            if razonable.sum() >= self.min_inliers:
                A, mask = cv2.estimateAffinePartial2D(
                    b[razonable], a[razonable], method=cv2.RANSAC,
                    ransacReprojThreshold=self.ransac_m, maxIters=3000, confidence=0.995)
                if A is not None and mask is not None:
                    idx = np.flatnonzero(razonable)[mask.ravel().astype(bool)]
                    if len(idx) >= self.min_inliers:
                        R, tr = rigido_2d(b[idx], a[idx])
                        M = matriz(R, tr)            # actual -> anterior
                        res = np.linalg.norm(aplicar(M, b[idx]) - a[idx], axis=1)
                        idx = idx[res < self.ransac_m * 1.5]
                        if len(idx) >= self.min_inliers:
                            R, tr = rigido_2d(b[idx], a[idx])
                            M = matriz(R, tr)
                            inl[idx] = True
                            estado = "ok"
                            # "Radar" del suelo: sin compensar vs compensado
                            v_ap = float(np.median(np.linalg.norm(b[idx] - a[idx], axis=1))) / dt * 3.6
                            v_comp = float(np.median(np.linalg.norm(aplicar(M, b[idx]) - a[idx],
                                                                    axis=1))) / dt * 3.6

        if estado == "ok":
            self._perdidos = 0
            self._M_ultimo = M
        else:
            self._perdidos += 1
            if self._perdidos > self.max_perdidos:
                M = np.eye(3)                        # sin información: asumir quieto

        self.pose = self.pose @ M
        self._prev = (pts, desc, t)
        ref_prev = self.trayectoria[-1][1:] if self.trayectoria else (0.0, 0.0)
        x, y = self._registrar_pose(t)
        v_bruta = math.hypot(x - ref_prev[0], y - ref_prev[1]) / dt * 3.6
        self._v_suave = v_bruta if self._v_suave is None else (
            self.alpha * v_bruta + (1 - self.alpha) * self._v_suave)

        r = ResultadoVO(estado, t, self.pose.copy(), v_camara_kmh=self._v_suave,
                        v_camara_bruta_kmh=v_bruta, giro_deg=self.giro_deg(),
                        v_suelo_aparente_kmh=v_ap, v_suelo_compensada_kmh=v_comp,
                        n_puntos=len(pts), n_parejas=len(parejas), n_inliers=int(inl.sum()),
                        px_prev=px_p, px_actual=px_a, inliers=inl)
        self.historial.append(r)
        return r

    def _registrar_pose(self, t):
        x, y = aplicar(self.pose, self.ref_local)[0]
        self.trayectoria.append((t, float(x), float(y)))
        return x, y

    # ------------------------------------------------------------------ #
    def a_mundo(self, pts_local) -> np.ndarray:
        """Metros en el sistema de la cámara actual -> metros en el MUNDO (fijo)."""
        return aplicar(self.pose, pts_local)

    def giro_deg(self) -> float:
        return math.degrees(math.atan2(self.pose[1, 0], self.pose[0, 0]))

    def distancia_recorrida(self) -> float:
        tr = np.array([p[1:] for p in self.trayectoria])
        return float(np.linalg.norm(np.diff(tr, axis=0), axis=1).sum()) if len(tr) > 1 else 0.0
