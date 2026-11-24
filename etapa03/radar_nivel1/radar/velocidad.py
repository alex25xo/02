"""
Estimación de velocidad por vehículo y control del límite de velocidad.

Idea principal
--------------
Para cada vehículo (track_id) guardamos sus últimas posiciones en METROS junto
con el instante en SEGUNDOS DEL VIDEO (frame / fps, no el reloj de la PC).

En lugar de usar solo dos posiciones (muy sensible al "temblor" de la caja de
YOLO), ajustamos una recta posición-vs-tiempo por mínimos cuadrados dentro de
una ventana deslizante (por defecto 0.6 s). La pendiente de esa recta es la
velocidad. Luego se suaviza con una media móvil exponencial (EMA).

    v [m/s] = pendiente de x(t) y y(t)      ->   km/h = v * 3.6
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from .calibracion import Calibracion


@dataclass
class EstadisticaVehiculo:
    clases: Counter = field(default_factory=Counter)
    medidas: list = field(default_factory=list)   # velocidades suavizadas (km/h)
    max_kmh: float = 0.0
    t_primera: float | None = None
    t_ultima: float | None = None
    exceso: bool = False
    kmh_exceso: float = 0.0

    @property
    def n_medidas(self) -> int:
        return len(self.medidas)

    @property
    def media_kmh(self) -> float:
        return float(np.mean(self.medidas)) if self.medidas else 0.0

    @property
    def mediana_kmh(self) -> float:
        """Velocidad representativa del vehículo (robusta a picos y a la entrada en escena)."""
        return float(np.median(self.medidas)) if self.medidas else 0.0

    @property
    def clase(self) -> str:
        return self.clases.most_common(1)[0][0] if self.clases else "?"


class EstimadorVelocidad:
    """
    Parámetros
    ----------
    ventana_s    : segundos de historia usados para la regresión.
    min_lapso_s  : tiempo mínimo observado antes de dar la primera velocidad.
    alpha        : suavizado EMA (0-1). Más alto = reacciona más rápido, más ruido.
    vmax_kmh     : velocidades por encima se consideran error (cambio de ID, etc.).
    olvido_s     : se borra la historia de un vehículo no visto en este tiempo.
    """

    def __init__(self, calib: Calibracion, ventana_s: float = 0.6,
                 min_lapso_s: float = 0.3, alpha: float = 0.35,
                 vmax_kmh: float = 250.0, olvido_s: float = 2.0):
        self.calib = calib
        self.ventana_s = ventana_s
        self.min_lapso_s = min(min_lapso_s, ventana_s)
        self.alpha = alpha
        self.vmax_kmh = vmax_kmh
        self.olvido_s = olvido_s

        self._hist: dict[int, deque] = defaultdict(deque)  # id -> (t, X, Y)
        self._visto: dict[int, float] = {}
        self.velocidad: dict[int, float] = {}               # id -> km/h suavizada
        self.stats: dict[int, EstadisticaVehiculo] = defaultdict(EstadisticaVehiculo)

    # ------------------------------------------------------------------ #
    def actualizar(self, tid: int, t: float, punto_px, clase: str = "",
                   medir: bool = True) -> float | None:
        """
        Registra la posición del vehículo `tid` en el instante `t` (s).
        `medir=False` (ej. caja cortada por el borde) solo marca que se vio.
        Devuelve la velocidad suavizada en km/h, o None si aún no hay medida.
        """
        self._visto[tid] = t
        st = self.stats[tid]
        if clase:
            st.clases[clase] += 1

        if not medir:
            return self.velocidad.get(tid)

        if not self.calib.en_zona(punto_px):
            # Fuera de la zona de medición: se reinicia la historia para que la
            # próxima medida use solo posiciones dentro de la zona.
            self._hist.pop(tid, None)
            return self.velocidad.get(tid)

        X, Y = self.calib.a_metros(punto_px)[0]
        h = self._hist[tid]
        if h:
            t_ant, X_ant, Y_ant = h[-1]
            if t <= t_ant:           # tiempo repetido (no debería ocurrir)
                return self.velocidad.get(tid)
            # Salto entre dos cuadros seguidos físicamente imposible: casi siempre
            # es un cambio de ID del tracker. Se reinicia la historia desde aquí.
            paso_kmh = math.hypot(X - X_ant, Y - Y_ant) / (t - t_ant) * 3.6
            if paso_kmh > 2 * self.vmax_kmh:
                h.clear()
        h.append((t, X, Y))
        while h and t - h[0][0] > self.ventana_s:
            h.popleft()

        if len(h) < 3 or (h[-1][0] - h[0][0]) < self.min_lapso_s:
            return self.velocidad.get(tid)

        kmh = self._regresion_kmh(h)
        if kmh > self.vmax_kmh:
            # Salto imposible: casi siempre es un cambio de ID del tracker o una
            # caja mal detectada. Se descarta y se empieza de nuevo.
            h.clear()
            return self.velocidad.get(tid)

        previa = self.velocidad.get(tid)
        suave = kmh if previa is None else self.alpha * kmh + (1 - self.alpha) * previa
        self.velocidad[tid] = suave

        st.medidas.append(suave)
        st.max_kmh = max(st.max_kmh, suave)
        st.t_primera = t if st.t_primera is None else st.t_primera
        st.t_ultima = t
        return suave

    @staticmethod
    def _regresion_kmh(h: deque) -> float:
        arr = np.asarray(h, dtype=np.float64)
        t = arr[:, 0] - arr[:, 0].mean()
        den = float(np.dot(t, t))
        if den <= 0:
            return 0.0
        vx = float(np.dot(t, arr[:, 1] - arr[:, 1].mean())) / den
        vy = float(np.dot(t, arr[:, 2] - arr[:, 2].mean())) / den
        return math.hypot(vx, vy) * 3.6

    def purgar(self, t_actual: float) -> None:
        """Libera memoria de vehículos que ya salieron de escena."""
        for tid in [k for k, tv in self._visto.items() if t_actual - tv > self.olvido_s]:
            self._hist.pop(tid, None)
            self._visto.pop(tid, None)

    def vehiculos_medidos(self) -> dict[int, EstadisticaVehiculo]:
        return {k: v for k, v in self.stats.items() if v.n_medidas > 0}


class ControlLimite:
    """
    Decide el color/estado de cada vehículo según el límite de velocidad.

      "sin_medida" : aún no hay velocidad (gris)
      "normal"     : por debajo del 90 % del límite (verde)
      "precaucion" : entre el 90 % y el límite, o sobre el límite sin confirmar (ámbar)
      "exceso"     : supera el límite durante `confirmar` medidas seguidas (ROJO)

    Pedir varias medidas seguidas evita falsas alarmas por un pico de ruido.
    """

    def __init__(self, limite_kmh: float, confirmar: int = 3, umbral_precaucion: float = 0.9):
        self.limite = limite_kmh
        self.confirmar = max(1, confirmar)
        self.umbral_precaucion = umbral_precaucion
        self._seguidas: dict[int, int] = defaultdict(int)
        self._ultima: dict[int, float] = {}
        self.infractores: set[int] = set()

    def evaluar(self, tid: int, kmh: float | None) -> tuple[str, bool]:
        """Devuelve (estado, es_nuevo_infractor)."""
        if kmh is None:
            return "sin_medida", False
        nuevo = False
        if kmh != self._ultima.get(tid):       # solo cuenta medidas nuevas
            self._ultima[tid] = kmh
            self._seguidas[tid] = self._seguidas[tid] + 1 if kmh > self.limite else 0
            if self._seguidas[tid] >= self.confirmar and tid not in self.infractores:
                self.infractores.add(tid)
                nuevo = True
        if kmh > self.limite:
            return ("exceso" if self._seguidas[tid] >= self.confirmar else "precaucion"), nuevo
        if kmh >= self.umbral_precaucion * self.limite:
            return "precaucion", nuevo
        return "normal", nuevo
