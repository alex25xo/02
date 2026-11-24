"""Utilidades para abrir el video (archivo subido o webcam)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

EXTENSIONES = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.m4v", "*.wmv", "*.webm")


def elegir_video_dialogo() -> str | None:
    """Abre un explorador de archivos para 'subir' el video."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None
    raiz = tk.Tk()
    raiz.withdraw()
    raiz.attributes("-topmost", True)
    ruta = filedialog.askopenfilename(
        title="Selecciona el video de trafico",
        filetypes=[("Videos", " ".join(EXTENSIONES)), ("Todos", "*.*")])
    raiz.destroy()
    return ruta or None


def resolver_fuente(video: str | None):
    """
    Devuelve la fuente para cv2.VideoCapture:
      - None  -> abre el diálogo para elegir archivo
      - "0"   -> webcam 0 (entero)
      - ruta  -> archivo de video
    """
    if video is None:
        video = elegir_video_dialogo()
        if not video:
            sys.exit("No se seleccionó ningún video. Usa --video ruta/al/video.mp4")
    if str(video).isdigit():
        return int(video), True
    if not Path(video).exists():
        sys.exit(f"No existe el archivo de video: {video}")
    return str(video), False


def abrir(fuente) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        sys.exit(f"No se pudo abrir la fuente de video: {fuente}")
    return cap


def fps_confiable(cap: cv2.VideoCapture, fps_forzado: float | None = None) -> tuple[float, bool]:
    """
    FPS del video. Devuelve (fps, es_confiable).
    Algunos archivos (sobre todo de celular o editados) reportan FPS absurdos.
    """
    if fps_forzado:
        return float(fps_forzado), True
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if 1.0 <= fps <= 240.0:
        return float(fps), True
    return 30.0, False


def info_video(cap: cv2.VideoCapture) -> dict:
    return {
        "ancho": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "alto": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
