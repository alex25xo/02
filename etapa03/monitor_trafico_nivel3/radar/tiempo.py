"""
Hora REAL de inicio del video (Nivel 2: gráfica de vehículos por hora).

Orden de prioridad:
  1. --hora-inicio escrita por el usuario       (lo más confiable)
  2. Nombre del archivo del celular              VID_20261120_073015.mp4, PXL_..., 20261120_073015...
  3. Metadato interno del MP4/MOV (átomo 'mvhd', fecha de creación)
  4. Sin hora real -> se usa el tiempo relativo desde el inicio del video
"""
from __future__ import annotations

import re
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path

_PATRON_NOMBRE = re.compile(
    r"(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})"           # fecha  AAAA MM DD
    r"(?:[ _T-]|\s+at\s+)?"                           # separador
    r"(\d{2})[-_.:h]?(\d{2})[-_.:m]?(\d{2})")         # hora   HH MM SS


def hora_desde_nombre(nombre: str) -> datetime | None:
    for m in _PATRON_NOMBRE.finditer(Path(nombre).stem):
        try:
            return datetime(*map(int, m.groups()))
        except ValueError:
            continue
    return None


def _buscar_atomo(f, fin: int, nombre: bytes):
    """Recorre átomos MP4 hermanos entre la posición actual y `fin`."""
    while f.tell() + 8 <= fin:
        inicio = f.tell()
        tam, tipo = struct.unpack(">I4s", f.read(8))
        cabecera = 8
        if tam == 1:
            tam = struct.unpack(">Q", f.read(8))[0]
            cabecera = 16
        elif tam == 0:
            tam = fin - inicio
        if tam < cabecera:
            return None
        if tipo == nombre:
            return inicio + cabecera, inicio + tam
        f.seek(inicio + tam)
    return None


def hora_desde_mp4(ruta: str | Path) -> datetime | None:
    """Fecha de creación guardada en el MP4/MOV (UTC) convertida a hora local."""
    try:
        with open(ruta, "rb") as f:
            f.seek(0, 2)
            fin = f.tell()
            f.seek(0)
            moov = _buscar_atomo(f, fin, b"moov")
            if not moov:
                return None
            f.seek(moov[0])
            mvhd = _buscar_atomo(f, moov[1], b"mvhd")
            if not mvhd:
                return None
            f.seek(mvhd[0])
            version = f.read(4)[0]
            segundos = (struct.unpack(">Q", f.read(8))[0] if version == 1
                        else struct.unpack(">I", f.read(4))[0])
    except (OSError, struct.error, IndexError):
        return None
    if segundos == 0:
        return None
    utc = datetime(1904, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=segundos)
    local = utc.astimezone().replace(tzinfo=None)
    if not (2005 <= local.year <= datetime.now().year + 1):
        return None
    return local


def parsear_hora_usuario(texto: str) -> datetime:
    """Acepta 'AAAA-MM-DD HH:MM[:SS]' o solo 'HH:MM[:SS]' (fecha de hoy)."""
    texto = texto.strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(texto, fmt)
        except ValueError:
            pass
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            h = datetime.strptime(texto, fmt)
            return datetime.combine(datetime.now().date(), h.time())
        except ValueError:
            pass
    raise ValueError(f"No entiendo la hora '{texto}'. Usa por ejemplo \"2026-11-20 07:30\" o \"07:30\"")


def determinar_hora_inicio(fuente, hora_usuario: str | None, es_webcam: bool):
    """Devuelve (datetime | None, descripción de la fuente)."""
    if hora_usuario:
        return parsear_hora_usuario(hora_usuario), "--hora-inicio"
    if es_webcam:
        return datetime.now(), "reloj de la PC (webcam en vivo)"
    h = hora_desde_nombre(str(fuente))
    if h:
        return h, "nombre del archivo"
    h = hora_desde_mp4(fuente)
    if h:
        return h, "metadatos del MP4 (verifica que sea correcta)"
    return None, "sin hora real (usa --hora-inicio)"
