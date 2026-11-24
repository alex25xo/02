"""
Registro del historial de vehículos en CSV (Nivel 2).

Archivos (en salidas/<video>/):
  historial.csv  posición de cada vehículo a lo largo del tiempo (por defecto 5 veces/s
                 por vehículo; registrar los 30 cuadros/s generaría archivos enormes)
  conteo.csv     una fila por vehículo que cruzó la línea de conteo
  sesion.json    datos del video: hora de inicio, fps, calibración, línea de conteo...
  fondo.jpg      la calle vacía (mediana de varios cuadros), base del mapa de calor
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from .conteo import EventoConteo

CAMPOS_HISTORIAL = ["frame", "t_video_s", "fecha_hora", "id", "clase", "conf",
                    "x1", "y1", "x2", "y2", "px_x", "px_y", "s_m", "d_m",
                    "en_zona", "vel_kmh", "estado"]
CAMPOS_CONTEO = ["id", "fecha_hora", "t_video_s", "clase", "sentido", "flecha",
                 "s_m", "d_m", "px_x", "px_y", "vel_kmh"]


def _f(v, dec=2):
    return "" if v is None else f"{v:.{dec}f}"


class RegistroHistorial:
    def __init__(self, carpeta: Path, hora_inicio: datetime | None, hz: float = 5.0):
        self.carpeta = Path(carpeta)
        self.hora_inicio = hora_inicio
        self.periodo = 1.0 / hz if hz and hz > 0 else 0.0
        self._ultimo: dict[int, float] = {}
        self.filas = 0
        self._fh = open(self.carpeta / "historial.csv", "w", newline="", encoding="utf-8-sig")
        self._fc = open(self.carpeta / "conteo.csv", "w", newline="", encoding="utf-8-sig")
        self._wh = csv.writer(self._fh)
        self._wc = csv.writer(self._fc)
        self._wh.writerow(CAMPOS_HISTORIAL)
        self._wc.writerow(CAMPOS_CONTEO)

    def fecha(self, t: float) -> str:
        if self.hora_inicio is None:
            return ""
        return (self.hora_inicio + timedelta(seconds=t)).isoformat(sep=" ", timespec="milliseconds")

    def registrar(self, frame: int, t: float, tid: int, clase: str, conf: float, caja,
                  punto, sd, en_zona: bool, kmh: float | None, estado: str) -> None:
        if self.periodo and tid in self._ultimo and t - self._ultimo[tid] < self.periodo - 1e-6:
            return
        self._ultimo[tid] = t
        x1, y1, x2, y2 = caja
        self._wh.writerow([frame, f"{t:.3f}", self.fecha(t), tid, clase, f"{conf:.2f}",
                           f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                           f"{punto[0]:.1f}", f"{punto[1]:.1f}", f"{sd[0]:.2f}", f"{sd[1]:.2f}",
                           int(en_zona), _f(kmh, 1), estado])
        self.filas += 1
        if self.filas % 500 == 0:
            self._fh.flush()

    def registrar_conteo(self, ev: EventoConteo, flecha: str) -> None:
        self._wc.writerow([ev.id, self.fecha(ev.t_video), f"{ev.t_video:.3f}", ev.clase,
                           ev.sentido, flecha, f"{ev.s_m:.2f}", f"{ev.d_m:.2f}",
                           f"{ev.px[0]:.1f}", f"{ev.px[1]:.1f}", _f(ev.vel_kmh, 1)])
        self._fc.flush()

    def cerrar(self) -> None:
        for f in (self._fh, self._fc):
            if not f.closed:
                f.close()


class MuestreadorFondo:
    """Guarda algunos cuadros y calcula su mediana: la calle sin autos."""

    def __init__(self, cada_n_frames: int, ancho_max: int = 960, max_muestras: int = 40):
        self.cada = max(1, int(cada_n_frames))
        self.ancho_max = ancho_max
        self.max_muestras = max_muestras
        self.muestras: list[np.ndarray] = []

    def agregar(self, idx: int, frame: np.ndarray) -> None:
        if idx % self.cada:
            return
        h, w = frame.shape[:2]
        s = min(1.0, self.ancho_max / w)
        img = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame.copy()
        self.muestras.append(img)
        if len(self.muestras) > self.max_muestras:     # conserva muestras repartidas en el tiempo
            self.muestras = self.muestras[::2]
            self.cada *= 2

    def guardar(self, ruta: Path, tam_original: tuple[int, int]) -> bool:
        if not self.muestras:
            return False
        fondo = np.median(np.stack(self.muestras), axis=0).astype(np.uint8)
        fondo = cv2.resize(fondo, tam_original, interpolation=cv2.INTER_LINEAR)
        return bool(cv2.imwrite(str(ruta), fondo))


def guardar_sesion(ruta: Path, datos: dict) -> None:
    Path(ruta).write_text(json.dumps(datos, indent=2, ensure_ascii=False, default=str),
                          encoding="utf-8")
