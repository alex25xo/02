"""
Funciones de dibujo sobre los cuadros del video.

Nota: las fuentes de OpenCV (Hershey) NO tienen tildes ni ñ, por eso todo el
texto que se dibuja en pantalla va sin acentos.
"""
from __future__ import annotations

import cv2
import numpy as np

from .calibracion import Calibracion

COLORES = {                      # BGR
    "sin_medida": (170, 170, 170),
    "normal": (60, 200, 60),
    "precaucion": (0, 170, 255),
    "exceso": (0, 0, 255),
}
FUENTE = cv2.FONT_HERSHEY_SIMPLEX


def etiqueta(img, texto, x, y, color_fondo, escala=0.55, grosor=1, color_texto=(255, 255, 255)):
    """Texto con fondo sólido. (x, y) = esquina inferior izquierda."""
    (tw, th), base = cv2.getTextSize(texto, FUENTE, escala, grosor)
    x = int(max(0, min(x, img.shape[1] - tw - 6)))
    y = int(max(th + 6, y))
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + base - 2), color_fondo, -1)
    cv2.putText(img, texto, (x + 3, y - 3), FUENTE, escala, color_texto, grosor, cv2.LINE_AA)


def dibujar_zona(img, calib: Calibracion, alpha: float = 0.07):
    """Zona de medición semitransparente y referencias de calibración."""
    if calib.zona_px:
        pts = np.array(calib.zona_px, dtype=np.int32).reshape(-1, 1, 2)
        capa = img.copy()
        cv2.fillPoly(capa, [pts], (255, 200, 0))
        cv2.addWeighted(capa, alpha, img, 1 - alpha, 0, dst=img)
        cv2.polylines(img, [pts], True, (255, 200, 0), 1, cv2.LINE_AA)

    if calib.modo == "ppm" and len(calib.puntos_px) == 2:
        p1, p2 = [tuple(map(int, p)) for p in calib.puntos_px]
        cv2.line(img, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
        for p in (p1, p2):
            cv2.circle(img, p, 5, (0, 255, 255), -1, cv2.LINE_AA)
        d = calib.medidas_m.get("distancia", 0)
        cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        etiqueta(img, f"{d:.1f} m", cx - 20, cy - 8, (0, 120, 120), 0.5)
    elif calib.modo == "homografia" and len(calib.puntos_px) == 4:
        pts = np.array(calib.puntos_px, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)


def dibujar_vehiculo(img, caja, tid, clase, kmh, estado, trayectoria=None, parpadeo=False):
    x1, y1, x2, y2 = map(int, caja)
    color = COLORES[estado]
    grosor = 3 if estado == "exceso" else 2

    if estado == "exceso":
        # Relleno rojo intermitente = alerta visual del reto
        if parpadeo:
            capa = img.copy()
            cv2.rectangle(capa, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(capa, 0.25, img, 0.75, 0, dst=img)

    if trayectoria is not None and len(trayectoria) > 1:
        pts = np.array(trayectoria, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, grosor, cv2.LINE_AA)
    # punto de medición: centro inferior de la caja (contacto con el suelo)
    cv2.circle(img, ((x1 + x2) // 2, y2), 4, color, -1, cv2.LINE_AA)

    vel = f"{kmh:.0f} km/h" if kmh is not None else "-- km/h"
    etiqueta(img, f"#{tid} {clase} {vel}", x1, y1 - 2, color, 0.55, 1,
             (0, 0, 0) if estado in ("precaucion", "sin_medida") else (255, 255, 255))
    if estado == "exceso":
        etiqueta(img, "EXCESO DE VELOCIDAD", x1, y2 + 22, (0, 0, 200), 0.55, 2)


def dibujar_panel(img, lineas: list[str], alerta: bool = False):
    """Panel de información (HUD) en la esquina superior izquierda."""
    escala, grosor, sep = 0.6, 1, 26
    ancho = max(cv2.getTextSize(l, FUENTE, escala, grosor)[0][0] for l in lineas) + 20
    alto = sep * len(lineas) + 12
    capa = img.copy()
    cv2.rectangle(capa, (8, 8), (8 + ancho, 8 + alto), (20, 20, 20), -1)
    cv2.addWeighted(capa, 0.65, img, 0.35, 0, dst=img)
    if alerta:
        cv2.rectangle(img, (8, 8), (8 + ancho, 8 + alto), (0, 0, 255), 2)
    for i, l in enumerate(lineas):
        cv2.putText(img, l, (18, 8 + sep * (i + 1)), FUENTE, escala,
                    (255, 255, 255), grosor, cv2.LINE_AA)


def ajustar_para_pantalla(img, ancho_max=1280, alto_max=720, ampliar_max=1.0):
    """
    Ajusta la imagen al tamano de pantalla; devuelve (imagen, escala).
    Con ampliar_max > 1 tambien agranda los videos pequenos.
    """
    h, w = img.shape[:2]
    s = min(ampliar_max, ancho_max / w, alto_max / h)
    if abs(s - 1.0) > 1e-3:
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=interp)
    else:
        s = 1.0
    return img, s


def ventana_cerrada(nombre: str) -> bool:
    """True si el usuario cerro la ventana de OpenCV con la X."""
    try:
        return cv2.getWindowProperty(nombre, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return False


def dibujar_linea_conteo(img, contador, t: float):
    """Línea virtual de conteo (Nivel 2). Destella en verde cuando alguien la cruza."""
    if contador.segmento_px is None:
        return
    (xa, ya), (xb, yb) = contador.segmento_px
    recien = t - contador.t_ultimo_cruce < 0.4
    color = (60, 230, 60) if recien else (255, 255, 255)
    cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (0, 0, 0), 4, cv2.LINE_AA)
    cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), color, 2, cv2.LINE_AA)
    etiqueta(img, "LINEA DE CONTEO", int(xa) + 4, int(ya) - 6, (40, 40, 40), 0.45)
