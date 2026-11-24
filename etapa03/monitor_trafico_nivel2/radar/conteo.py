"""
Conteo de vehículos con una LÍNEA VIRTUAL (Nivel 2).

Un vehículo se cuenta UNA sola vez, en el instante en que su punto de contacto
con el suelo cruza la línea de conteo (perpendicular a la calle, en el centro
de la zona de medición). En ese instante se guarda:
  - la hora del cruce        -> gráfica de vehículos por hora
  - la posición a lo ancho d -> carril que usó (mapa de calor / carril más usado)
  - el sentido de circulación

¿Por qué una línea y no "contar IDs"? Si el tracker pierde un auto y le da un
ID nuevo, contar IDs lo contaría dos veces; con la línea solo cuenta el
fragmento que realmente la cruza.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .calibracion import Calibracion


@dataclass
class EventoConteo:
    id: int
    t_video: float          # segundos desde el inicio del video
    clase: str
    sentido: int            # +1 = avanza hacia +s, -1 = hacia -s
    s_m: float
    d_m: float              # posición a lo ancho de la vía (define el carril)
    px: tuple[float, float]
    vel_kmh: float | None


class ContadorLinea:
    def __init__(self, calib: Calibracion, resolucion: tuple[int, int],
                 fraccion: float = 0.5, olvido_s: float = 2.0):
        self.calib = calib
        self.olvido_s = olvido_s
        W, H = resolucion
        zona = calib.zona_px or [(0, 0), (W, 0), (W, H), (0, H)]
        sd = calib.a_via(zona)
        s_min, s_max = float(sd[:, 0].min()), float(sd[:, 0].max())
        self.fraccion = fraccion
        self.s_linea = s_min + fraccion * (s_max - s_min)

        # Tramo visible de la línea (para dibujarla): puntos dentro de la zona
        ds = np.linspace(sd[:, 1].min(), sd[:, 1].max(), 300)
        px = calib.desde_via(np.c_[np.full_like(ds, self.s_linea), ds])
        dentro = [tuple(p) for p in px
                  if 0 <= p[0] < W and 0 <= p[1] < H and calib.en_zona(p)]
        self.segmento_px = (dentro[0], dentro[-1]) if len(dentro) >= 2 else None

        # ¿Hacia dónde apunta +s en la imagen? (para dibujar flechas de sentido)
        d_mid = float(np.median(ds))
        a, b = calib.desde_via([[self.s_linea, d_mid], [self.s_linea + 1.0, d_mid]])
        v = b - a
        self.via_horizontal = bool(abs(v[0]) >= abs(v[1]))
        self._signo_img = float(np.sign(v[0] if self.via_horizontal else v[1])) or 1.0

        self._prev: dict[int, tuple] = {}
        self._contados: set[int] = set()
        self.eventos: list[EventoConteo] = []
        self.por_sentido: Counter = Counter()
        self.t_ultimo_cruce = -1e9

    # ------------------------------------------------------------------ #
    def flecha(self, sentido: int, unicode: bool = False) -> str:
        """Flecha del sentido tal como se ve en la imagen."""
        positivo = sentido * self._signo_img > 0
        if self.via_horizontal:
            return ("→" if positivo else "←") if unicode else ("->" if positivo else "<-")
        return ("↓" if positivo else "↑") if unicode else ("v" if positivo else "^")

    def actualizar(self, tid: int, t: float, punto_px, clase: str,
                   vel_kmh: float | None) -> EventoConteo | None:
        s, d = self.calib.a_via([punto_px])[0]
        p = np.asarray(punto_px, dtype=np.float64)
        previo = self._prev.get(tid)
        self._prev[tid] = (t, s, d, p)
        if previo is None or tid in self._contados:
            return None
        t0, s0, d0, p0 = previo
        L = self.s_linea
        if not ((s0 < L <= s) or (s0 > L >= s)):
            return None
        f = (L - s0) / (s - s0)                       # interpolación al cruce exacto
        p_cruce = p0 + f * (p - p0)
        if not self.calib.en_zona(p_cruce):           # cruzó fuera de la pista
            return None
        ev = EventoConteo(id=tid, t_video=t0 + f * (t - t0), clase=clase,
                          sentido=1 if s > s0 else -1, s_m=L, d_m=d0 + f * (d - d0),
                          px=(float(p_cruce[0]), float(p_cruce[1])), vel_kmh=vel_kmh)
        self._contados.add(tid)
        self.eventos.append(ev)
        self.por_sentido[ev.sentido] += 1
        self.t_ultimo_cruce = t
        return ev

    def purgar(self, t_actual: float) -> None:
        for tid in [k for k, v in self._prev.items() if t_actual - v[0] > self.olvido_s]:
            del self._prev[tid]

    @property
    def total(self) -> int:
        return len(self.eventos)
