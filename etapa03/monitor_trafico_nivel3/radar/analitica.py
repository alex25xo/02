"""
NIVEL 2 - Analítica de tránsito.

A partir de lo que guardó el radar (historial.csv, conteo.csv, sesion.json,
fondo.jpg) genera:
  - mapa_calor.png         por dónde pasan los vehículos + carril más usado
  - vehiculos_por_hora.png a qué hora pasaron más vehículos
  - vehiculos_contados.csv cada vehículo contado con su carril asignado
  - carriles.csv           tabla por carril
  - resumen_nivel2.txt     resumen en texto (hora pico, carril más usado...)

Puede combinar varias sesiones (clips grabados a distintas horas desde el mismo
punto): la gráfica por hora las junta, y el mapa de calor y los carriles se
combinan si la calibración es la misma.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.ticker import MaxNLocator                       # noqa: E402

from .calibracion import Calibracion                             # noqa: E402
from .conteo import ContadorLinea                                # noqa: E402

# ---------------------------------------------------------------------- #
# Estilo (paleta de referencia validada: orden fijo, el color sigue a la clase)
# ---------------------------------------------------------------------- #
ORDEN_CLASES = ["auto", "moto", "bus", "camion"]
COLOR_CLASE = {"auto": "#2a78d6", "moto": "#eb6834", "bus": "#1baf7a", "camion": "#eda100",
               "vehiculo": "#6250d6"}
TINTA, TINTA2, APAGADO = "#0b0b0b", "#52514e", "#898781"
REJILLA, EJE, SUPERFICIE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
RAMPA_AZUL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
              "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP_CALOR = LinearSegmentedColormap.from_list("calor_azul", RAMPA_AZUL)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE,
    "savefig.facecolor": SUPERFICIE, "text.color": TINTA,
    "axes.labelcolor": TINTA2, "xtick.color": APAGADO, "ytick.color": APAGADO,
    "axes.edgecolor": EJE, "font.size": 10,
})


# ---------------------------------------------------------------------- #
# Carga de datos
# ---------------------------------------------------------------------- #
@dataclass
class Sesion:
    carpeta: Path
    meta: dict
    calib: Calibracion
    hora_inicio: datetime | None
    duracion_s: float
    eventos: list[dict]
    trayectorias: dict[int, np.ndarray]          # id -> [[t, px_x, px_y, s, d, en_zona], ...]
    fondo: np.ndarray | None = None
    contador: ContadorLinea | None = field(default=None, repr=False)

    @property
    def nombre(self) -> str:
        return self.meta.get("nombre", self.carpeta.name)

    @property
    def intervalo(self):
        if self.hora_inicio is None:
            return None
        return self.hora_inicio, self.hora_inicio + timedelta(seconds=self.duracion_s)


def _num(v, defecto=None):
    try:
        return float(v) if v not in ("", None) else defecto
    except ValueError:
        return defecto


def cargar_sesion(carpeta: str | Path) -> Sesion:
    carpeta = Path(carpeta)
    meta_ruta = carpeta / "sesion.json"
    if not meta_ruta.exists():
        raise FileNotFoundError(f"No hay sesion.json en {carpeta}. ¿Ejecutaste el radar con ese video?")
    meta = json.loads(meta_ruta.read_text(encoding="utf-8"))
    c = meta["calibracion"]
    calib = Calibracion(c["modo"], tuple(c["resolucion"]), ppm=c.get("ppm"), H=c.get("H"),
                        puntos_px=c.get("puntos_px"), medidas_m=c.get("medidas_m"),
                        zona_px=c.get("zona_px"), origen=c.get("origen", ""))
    hora = datetime.fromisoformat(meta["hora_inicio"]) if meta.get("hora_inicio") else None

    eventos = []
    with open(carpeta / "conteo.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            t = float(r["t_video_s"])
            eventos.append({
                "sesion": meta.get("nombre", carpeta.name), "id": int(r["id"]), "t_video": t,
                "fecha": hora + timedelta(seconds=t) if hora else None,
                "clase": r["clase"], "sentido": int(r["sentido"]),
                "d_m": float(r["d_m"]), "vel_kmh": _num(r["vel_kmh"]),
            })

    filas = defaultdict(list)
    ruta_h = carpeta / "historial.csv"
    if ruta_h.exists():
        with open(ruta_h, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                filas[int(r["id"])].append((float(r["t_video_s"]), float(r["px_x"]), float(r["px_y"]),
                                            float(r["s_m"]), float(r["d_m"]), int(r["en_zona"])))
    trayectorias = {k: np.array(sorted(v)) for k, v in filas.items()}

    fondo = None
    if (carpeta / meta.get("fondo", "fondo.jpg")).exists():
        fondo = cv2.imread(str(carpeta / meta.get("fondo", "fondo.jpg")))
    ses = Sesion(carpeta, meta, calib, hora, float(meta.get("duracion_s", 0.0)),
                 eventos, trayectorias, fondo)
    ses.contador = ContadorLinea(calib, calib.resolucion,
                                 fraccion=meta.get("linea_conteo", {}).get("fraccion", 0.5))
    return ses


def misma_camara(a: Sesion, b: Sesion) -> bool:
    ca, cb = a.calib, b.calib
    if ca.modo != cb.modo or ca.resolucion != cb.resolucion:
        return False
    if ca.modo == "ppm":
        return abs(ca.ppm - cb.ppm) < 1e-6 and np.allclose(ca.puntos_px, cb.puntos_px)
    return np.allclose(ca.H, cb.H, rtol=1e-5, atol=1e-8)


# ---------------------------------------------------------------------- #
# Carriles
# ---------------------------------------------------------------------- #
def _kmeans_1d(x: np.ndarray, centros: np.ndarray, iter_max: int = 100) -> np.ndarray:
    c = np.sort(np.asarray(centros, dtype=float))
    for _ in range(iter_max):
        etiqueta = np.argmin(np.abs(x[:, None] - c[None, :]), axis=1)
        nuevo = np.array([x[etiqueta == k].mean() if np.any(etiqueta == k) else c[k]
                          for k in range(len(c))])
        if np.allclose(nuevo, c, atol=1e-4):
            break
        c = nuevo
    return np.sort(c)


def detectar_carriles(d, n: int | None = None, sep_min: float = 2.0) -> np.ndarray:
    """
    Centros de carril (posición a lo ancho, en metros) a partir de dónde cruzó
    cada vehículo la línea de conteo.
      - n dado: agrupa en n carriles (k-means 1D).
      - n None: busca los picos del histograma (carriles separados >= sep_min m).
    """
    d = np.asarray(d, dtype=float)
    if len(d) == 0:
        return np.array([])
    if n:
        n = max(1, min(int(n), len(np.unique(np.round(d, 2)))))
        return _kmeans_1d(d, np.quantile(d, (np.arange(n) + 0.5) / n))
    paso, sigma = 0.1, 0.45
    bordes = np.arange(d.min() - 3, d.max() + 3 + paso, paso)
    hist, _ = np.histogram(d, bordes)
    k = np.arange(-int(4 * sigma / paso), int(4 * sigma / paso) + 1) * paso
    suave = np.convolve(hist, np.exp(-0.5 * (k / sigma) ** 2), mode="same")
    x = (bordes[:-1] + bordes[1:]) / 2
    picos = [i for i in range(1, len(suave) - 1)
             if suave[i] >= suave[i - 1] and suave[i] > suave[i + 1] and suave[i] >= 0.12 * suave.max()]
    elegidos: list[float] = []
    for i in sorted(picos, key=lambda i: -suave[i]):
        if all(abs(x[i] - e) >= sep_min for e in elegidos):
            elegidos.append(x[i])
    if not elegidos:
        elegidos = [float(np.median(d))]
    return _kmeans_1d(d, np.array(elegidos))


def asignar_carril(d, centros: np.ndarray) -> np.ndarray:
    return np.argmin(np.abs(np.asarray(d, float)[:, None] - centros[None, :]), axis=1)


def numerar_carriles(centros: np.ndarray, ses: Sesion) -> dict[int, int]:
    """Carril 1 = el de más arriba (vía horizontal en la imagen) o el de más a la izquierda."""
    s = ses.contador.s_linea
    px = ses.calib.desde_via(np.c_[np.full(len(centros), s), centros])
    clave = px[:, 1] if ses.contador.via_horizontal else px[:, 0]
    return {int(idx): k + 1 for k, idx in enumerate(np.argsort(clave))}


def linea_carril_px(ses: Sesion, d_c: float, n: int = 80) -> np.ndarray:
    """Puntos (px) de la línea central de un carril dentro de la zona de medición."""
    W, H = ses.calib.resolucion
    zona = ses.calib.zona_px or [(0, 0), (W, 0), (W, H), (0, H)]
    sd = ses.calib.a_via(zona)
    ss = np.linspace(sd[:, 0].min(), sd[:, 0].max(), n)
    px = ses.calib.desde_via(np.c_[ss, np.full_like(ss, d_c)])
    ok = [(0 <= p[0] < W and 0 <= p[1] < H and ses.calib.en_zona(p)) for p in px]
    return px[np.array(ok, dtype=bool)]


# ---------------------------------------------------------------------- #
# Gráfica 1: mapa de calor
# ---------------------------------------------------------------------- #
def calcular_calor(sesiones: list[Sesion], ancho_trabajo: int = 960):
    """Acumulador: cuántos vehículos pasaron por cada píxel (cada vehículo suma 1)."""
    W, H = sesiones[0].calib.resolucion
    esc = min(1.0, ancho_trabajo / W)
    w2, h2 = int(W * esc), int(H * esc)
    acc = np.zeros((h2, w2), np.float32)
    grosor = max(6, int(0.02 * w2))         # ancho de la "huella" de cada trayectoria
    n_tray = 0
    for ses in sesiones:
        for tray in ses.trayectorias.values():
            if len(tray) < 3:
                continue
            pts = np.round(tray[:, 1:3] * esc).astype(np.int32)
            x0, y0 = np.maximum(pts.min(axis=0) - grosor, 0)
            x1, y1 = np.minimum(pts.max(axis=0) + grosor + 1, [w2, h2])
            if x1 <= x0 or y1 <= y0:
                continue
            m = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.polylines(m, [pts - [x0, y0]], False, 1, grosor, cv2.LINE_8)
            acc[y0:y1, x0:x1] += m
            n_tray += 1
    # Desenfoque leve: suaviza bordes pero conserva el valor del centro de la banda
    acc = cv2.GaussianBlur(acc, (0, 0), grosor * 0.25)
    return acc, esc, n_tray


def figura_mapa_calor(sesiones, carriles, ruta: Path, titulo_extra: str = "") -> None:
    ses0 = sesiones[0]
    W, H = ses0.calib.resolucion
    acc, esc, n_tray = calcular_calor(sesiones)
    h2, w2 = acc.shape

    if ses0.fondo is not None:
        fondo = cv2.resize(ses0.fondo, (w2, h2), interpolation=cv2.INTER_AREA)
        gris = cv2.cvtColor(fondo, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    else:
        gris = np.full((h2, w2), 0.5, np.float32)
    gris = 0.45 + 0.5 * gris                         # foto aclarada: el color resalta

    vmax = max(1.0, float(acc.max()))
    norm = Normalize(0, vmax)
    rgba = CMAP_CALOR(norm(acc))
    rgba[..., 3] = np.clip(acc / vmax * 1.4, 0, 0.88) * (acc > 0.02 * vmax)

    tiene_carriles = carriles is not None and len(carriles["filas"]) > 0
    ancho_fig = 13.0 if tiene_carriles else 10.0
    alto_img = 9.0 * h2 / w2
    fig = plt.figure(figsize=(ancho_fig, alto_img + 1.6), dpi=150)
    gs = fig.add_gridspec(1, 2 if tiene_carriles else 1,
                          width_ratios=[3, 1.15] if tiene_carriles else [1], wspace=0.2,
                          left=0.02, right=0.98, top=1 - 1.1 / (alto_img + 1.6), bottom=0.1)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(gris, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    ax.imshow(rgba, interpolation="bilinear")
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Línea de conteo
    seg = ses0.contador.segmento_px
    if seg:
        (xa, ya), (xb, yb) = seg
        ax.plot([xa * esc, xb * esc], [ya * esc, yb * esc], color=TINTA2, lw=1.2, alpha=0.8)
        ax.annotate("línea de conteo", (xb * esc, yb * esc), xytext=(4, -4), textcoords="offset points",
                    fontsize=8, color=TINTA2)

    # Carriles: línea central + etiqueta en el inicio
    if tiene_carriles:
        for fila in carriles["filas"]:
            pts = linea_carril_px(ses0, fila["d_centro"]) * esc
            if len(pts) < 2:
                continue
            ax.plot(pts[:, 0], pts[:, 1], color=TINTA, lw=0.8, alpha=0.55)
            i0 = int(np.argmin(pts[:, 0] if ses0.contador.via_horizontal else pts[:, 1]))
            x, y = pts[i0]
            y -= 0.035 * h2
            texto = f"Carril {fila['carril']} {fila['flecha']}  {fila['vehiculos']} ({fila['pct']:.0f}%)"
            mas = fila["carril"] == carriles["mas_usado"]
            ax.text(x + 4, y, texto + ("  · más usado" if mas else ""), fontsize=8.5,
                    fontweight="bold" if mas else "normal", color=TINTA, va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=TINTA if mas else "none",
                              lw=0.8, alpha=0.9))

    # Barra de color (escala secuencial de un solo tono)
    cax = fig.add_axes([0.02, 0.05, 0.35 if tiene_carriles else 0.45, 0.018])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP_CALOR), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=0)
    cb.ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    cax.set_title("vehículos que pasaron por ese punto", fontsize=8, color=TINTA2, loc="left", pad=3)

    # Panel derecho: vehículos por carril
    if tiene_carriles:
        axb = fig.add_subplot(gs[0, 1])
        filas = carriles["filas"]
        y = np.arange(len(filas))
        valores = [f["vehiculos"] for f in filas]
        axb.barh(y, valores, height=min(0.55, 0.15 + 0.1 * len(filas)), color=COLOR_CLASE["auto"],
                 edgecolor=SUPERFICIE, lw=1.0)
        axb.set_yticks(y, [f"Carril {f['carril']} {f['flecha']}" for f in filas], fontsize=9.5, color=TINTA)
        axb.invert_yaxis()
        top = max(valores) if valores else 1
        for yi, f in zip(y, filas):
            mas = f["carril"] == carriles["mas_usado"]
            axb.text(f["vehiculos"] + top * 0.03, yi, f"{f['vehiculos']}  ({f['pct']:.0f}%)"
                     + ("\nmás usado" if mas else ""), va="center", fontsize=9,
                     color=TINTA, fontweight="bold" if mas else "normal")
        axb.set_xlim(0, top * 1.55)
        axb.set_xlabel("vehículos contados", fontsize=9)
        axb.tick_params(axis="y", length=0)
        axb.grid(axis="x", color=REJILLA, lw=0.6)
        axb.set_axisbelow(True)
        for lado in ("top", "right", "left"):
            axb.spines[lado].set_visible(False)
        axb.set_title("Vehículos por carril", fontsize=10.5, loc="left", color=TINTA)

    total = carriles["total"] if tiene_carriles else 0
    sub = (f"{total} vehículos contados · {n_tray} trayectorias"
           + (f" · carril más usado: Carril {carriles['mas_usado']}" if tiene_carriles else ""))
    fig.text(0.02, 1 - 0.35 / (alto_img + 1.6), "Mapa de calor: ¿por dónde circulan los vehículos?",
             fontsize=14, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.72 / (alto_img + 1.6), sub + titulo_extra, fontsize=10, color=TINTA2, va="top")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------- #
# Gráfica 2: vehículos por hora
# ---------------------------------------------------------------------- #
def _dur(minutos: float) -> str:
    return f"{minutos * 60:.0f} s" if minutos < 1 else f"{minutos:.0f} min"


def cobertura_por_hora(sesiones: list[Sesion]) -> dict[datetime, float]:
    """Minutos grabados dentro de cada hora (une los intervalos de todas las sesiones)."""
    intervalos = sorted(s.intervalo for s in sesiones if s.intervalo)
    unidos: list[list[datetime]] = []
    for a, b in intervalos:
        if unidos and a <= unidos[-1][1]:
            unidos[-1][1] = max(unidos[-1][1], b)
        else:
            unidos.append([a, b])
    cob: dict[datetime, float] = defaultdict(float)
    for a, b in unidos:
        t = a
        while t < b:
            hora = t.replace(minute=0, second=0, microsecond=0)
            fin = min(b, hora + timedelta(hours=1))
            cob[hora] += (fin - t).total_seconds() / 60.0
            t = fin
    return dict(cob)


def _barras_apiladas(ax, etiquetas, conteos_por_clase, clases, etiquetar_totales=True, resaltar=None,
                     notas=None):
    x = np.arange(len(etiquetas))
    base = np.zeros(len(etiquetas))
    for c in clases:
        v = np.array(conteos_por_clase[c], dtype=float)
        ax.bar(x, v, bottom=base, width=0.62 if len(x) >= 8 else 0.25 + 0.045 * len(x), color=COLOR_CLASE.get(c, "#6250d6"),
               edgecolor=SUPERFICIE, linewidth=1.0, label=c)
        base += v
    top = base.max() if len(base) else 1
    for i, tot in enumerate(base):
        if etiquetar_totales or i == resaltar:
            nota = f"\n{notas[i]}" if notas else ""
            ax.text(i, tot + top * 0.02, f"{int(tot)}{nota}", ha="center", va="bottom", fontsize=9,
                    color=TINTA, fontweight="bold" if i == resaltar else "normal", linespacing=1.3)
    ax.set_xticks(x, etiquetas)
    relleno = max(0.0, (4 - len(x)) / 2)          # con pocas barras, que no se vuelvan bloques
    ax.set_xlim(-0.6 - relleno, len(x) - 0.4 + relleno)
    ax.set_ylim(0, top * (1.32 if notas else 1.18) if top else 1)
    ax.grid(axis="y", color=REJILLA, lw=0.6)
    ax.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(axis="both", length=0, labelsize=9)
    ax.set_ylabel("vehículos", fontsize=9)
    return base


def _detalle(eventos, sesiones, clases):
    """Barras de minutos adaptativas (1, 2, 5, 10, 15 o 30 min)."""
    reales = (bool(eventos) and all(e["fecha"] is not None for e in eventos)
              and any(s.intervalo for s in sesiones))
    if reales:
        ini = min(s.intervalo[0] for s in sesiones if s.intervalo)
        fin = max(s.intervalo[1] for s in sesiones if s.intervalo)
        span = (fin - ini).total_seconds() / 60.0
    else:
        span = max([s.duracion_s for s in sesiones] + [1.0]) / 60.0
    # Tamaño de grupo que deja ~12 barras (10 s ... 30 min)
    pasos_s = (10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600)
    paso_s = min(pasos_s, key=lambda p: (abs(span * 60 / p - 12), p))
    con_seg = paso_s < 60
    if reales:
        origen = ini.replace(microsecond=0)
        seg_dia = origen.hour * 3600 + origen.minute * 60 + origen.second
        origen -= timedelta(seconds=seg_dia % paso_s)
        n = int(math.ceil((fin - origen).total_seconds() / paso_s)) or 1
        idx = [int((e["fecha"] - origen).total_seconds() // paso_s) for e in eventos]
        fmt = "%H:%M:%S" if con_seg else "%H:%M"
        etiquetas = [(origen + timedelta(seconds=paso_s * i)).strftime(fmt) for i in range(n)]
    else:
        n = int(math.ceil(span * 60 / paso_s)) or 1
        idx = [int(e["t_video"] // paso_s) for e in eventos]
        etiquetas = [f"{(paso_s * i) // 60}:{(paso_s * i) % 60:02d}" for i in range(n)]
    paso = paso_s / 60.0
    conteos = {c: [0] * n for c in clases}
    for e, i in zip(eventos, idx):
        if 0 <= i < n:
            conteos[e["clase"]][i] += 1
    return etiquetas, conteos, paso


def figura_por_hora(eventos: list[dict], sesiones: list[Sesion], ruta: Path) -> dict:
    """Devuelve un dict con la hora pico y la tabla por hora (para el resumen)."""
    clases = [c for c in ORDEN_CLASES if any(e["clase"] == c for e in eventos)]
    clases += sorted({e["clase"] for e in eventos} - set(clases))
    reales = bool(eventos) and all(e["fecha"] is not None for e in eventos)
    info = {"reales": reales, "tabla": [], "pico": None, "paso_detalle": None}

    paneles = []
    if reales:
        cob = cobertura_por_hora(sesiones)
        horas = sorted(cob)
        por_hora = {h: Counter() for h in horas}
        for e in eventos:
            h = e["fecha"].replace(minute=0, second=0, microsecond=0)
            por_hora.setdefault(h, Counter())[e["clase"]] += 1
            cob.setdefault(h, 0.0)
        horas = sorted(por_hora)
        for h in horas:
            tot = sum(por_hora[h].values())
            minutos = cob.get(h, 0.0)
            tasa = tot / minutos * 60 if minutos > 0 else float("nan")
            info["tabla"].append({"hora": h, "vehiculos": tot, "minutos": minutos, "tasa": tasa,
                                  "por_clase": dict(por_hora[h])})
        validas = [r for r in info["tabla"] if r["minutos"] > 0]
        if validas:
            info["pico"] = max(validas, key=lambda r: (r["tasa"], r["vehiculos"]))
        if len(horas) >= 2:
            paneles.append("horas")
        ini = min(s.intervalo[0] for s in sesiones if s.intervalo)
        fin = max(s.intervalo[1] for s in sesiones if s.intervalo)
        if (fin - ini) <= timedelta(hours=3):
            paneles.append("detalle")
    else:
        paneles.append("detalle")
    if not paneles:
        paneles.append("detalle")

    alto = 3.6 * len(paneles) + 1.4
    fig, axes = plt.subplots(len(paneles), 1, figsize=(11, alto), dpi=150, squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.98, top=1 - 1.25 / alto, bottom=0.55 / alto, hspace=0.55)
    total = len(eventos)

    for ax, panel in zip(axes[:, 0], paneles):
        if panel == "horas":
            tabla = info["tabla"]
            etiquetas = []
            for r in tabla:
                et = r["hora"].strftime("%H:%M")
                if len({x["hora"].date() for x in tabla}) > 1:
                    et = r["hora"].strftime("%d/%m\n%H:%M")
                if 0 < r["minutos"] < 55:
                    et += f"\n({_dur(r['minutos'])} grabados)"
                etiquetas.append(et)
            conteos = {c: [r["por_clase"].get(c, 0) for r in tabla] for c in clases}
            pico_i = tabla.index(info["pico"]) if info["pico"] in tabla else None
            mins = [r["minutos"] for r in tabla if r["minutos"] > 0]
            desigual = bool(mins) and (max(mins) - min(mins)) > 0.1 * max(mins)
            notas = ([f"≈ {r['tasa']:.0f} veh/h" if r["minutos"] > 0 else "" for r in tabla]
                     if desigual else None)
            _barras_apiladas(ax, etiquetas, conteos, clases, len(tabla) <= 14, pico_i, notas)
            ax.set_title("Vehículos por hora del día", loc="left", fontsize=11, color=TINTA)
        else:
            etiquetas, conteos, paso = _detalle(eventos, sesiones, clases)
            info["paso_detalle"] = paso
            totales = np.sum([conteos[c] for c in clases], axis=0) if clases else np.zeros(len(etiquetas))
            pico_i = int(np.argmax(totales)) if len(totales) else None
            _barras_apiladas(ax, etiquetas, conteos, clases, len(etiquetas) <= 14, pico_i)
            if len(etiquetas) > 14:
                for lab in ax.get_xticklabels():
                    lab.set_rotation(90)
            unidad = (f"{paso * 60:.0f} s" if paso < 1 else "minuto" if paso == 1
                      else f"{paso:.0f} min")
            ax.set_title(f"Detalle: vehículos cada {unidad}"
                         + ("" if reales else " (min:seg desde el inicio del video)"),
                         loc="left", fontsize=11, color=TINTA)
    if len(clases) >= 2:          # una sola clase no necesita leyenda: lo dice el título
        axes[0, 0].legend(ncols=len(clases), frameon=False, loc="upper right",
                          bbox_to_anchor=(1.0, 1.16), fontsize=9, handlelength=1.0)

    if info["pico"]:
        p = info["pico"]
        h0 = p["hora"]
        sub = (f"Hora con más vehículos: {h0:%H:%M}–{h0 + timedelta(hours=1):%H:%M} · "
               f"{p['vehiculos']} vehículos en {_dur(p['minutos'])} grabados "
               f"(≈ {p['tasa']:.0f} veh/h) · total {total}")
    elif not reales:
        sub = f"{total} vehículos · sin hora real de grabación: usa --hora-inicio para ver la hora del día"
    else:
        sub = f"{total} vehiculos"
    fig.text(0.07, 1 - 0.3 / alto, "¿A qué hora pasaron más vehículos?", fontsize=14,
             fontweight="bold", va="top")
    fig.text(0.07, 1 - 0.68 / alto, sub, fontsize=10, color=TINTA2, va="top")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return info


# ---------------------------------------------------------------------- #
# Comprobación: ¿la calibración dice bien hacia dónde va la calle?
# ---------------------------------------------------------------------- #
def chequear_direccion(sesiones: list[Sesion]) -> str | None:
    largo = ancho = 0.0
    for ses in sesiones:
        for tray in ses.trayectorias.values():
            z = tray[tray[:, 5] > 0]
            if len(z) < 5:
                continue
            ds, dd = z[-1, 3] - z[0, 3], z[-1, 4] - z[0, 4]
            if math.hypot(ds, dd) > 3:
                largo += abs(ds)
                ancho += abs(dd)
    if largo + ancho == 0:
        return None
    if largo / (largo + ancho) < 0.75:
        return ("AVISO: los vehículos no se mueven a lo largo del eje de la calle de tu calibración. "
                "Revisa el orden de los puntos (PPM: P1->P2 en el sentido del tráfico; "
                "homografia: P2->P3 a lo largo de la calle). Los carriles pueden salir mal.")
    return None


# ---------------------------------------------------------------------- #
# Orquestador
# ---------------------------------------------------------------------- #
def generar_reportes(carpetas, salida=None, n_carriles: int | None = None,
                     imprimir: bool = True) -> dict:
    sesiones = [cargar_sesion(c) for c in carpetas]
    salida = Path(salida or sesiones[0].carpeta)
    salida.mkdir(parents=True, exist_ok=True)
    avisos = []

    # Sesiones de la misma cámara que la primera -> se combinan en carriles y calor
    compatibles = [s for s in sesiones if misma_camara(sesiones[0], s)]
    if len(compatibles) < len(sesiones):
        avisos.append("Algunas sesiones tienen otra calibración/encuadre: el mapa de calor y los "
                      "carriles usan solo las que coinciden con la primera.")
    av = chequear_direccion(compatibles)
    if av:
        avisos.append(av)

    eventos = [e for s in sesiones for e in s.eventos]
    eventos.sort(key=lambda e: (e["fecha"] or datetime.min, e["t_video"]))
    nombres_comp = {s.nombre for s in compatibles}
    ev_comp = [e for e in eventos if e["sesion"] in nombres_comp]

    # ---- Carriles ----
    carriles = None
    if ev_comp:
        centros = detectar_carriles([e["d_m"] for e in ev_comp], n_carriles)
        etiqueta = asignar_carril([e["d_m"] for e in ev_comp], centros)
        numero = numerar_carriles(centros, compatibles[0])
        filas = []
        for idx, centro in enumerate(centros):
            del_carril = [e for e, k in zip(ev_comp, etiqueta) if k == idx]
            sentidos = Counter(e["sentido"] for e in del_carril)
            sentido = sentidos.most_common(1)[0][0] if sentidos else 1
            vels = [e["vel_kmh"] for e in del_carril if e["vel_kmh"] is not None]
            filas.append({"carril": numero[idx], "d_centro": float(centro),
                          "vehiculos": len(del_carril), "pct": 100 * len(del_carril) / len(ev_comp),
                          "sentido": sentido,
                          "flecha": compatibles[0].contador.flecha(sentido, unicode=True),
                          "vel_media": float(np.mean(vels)) if vels else None,
                          "por_clase": dict(Counter(e["clase"] for e in del_carril))})
        filas.sort(key=lambda f: f["carril"])
        mas = max(filas, key=lambda f: f["vehiculos"])
        carriles = {"filas": filas, "mas_usado": mas["carril"], "total": len(ev_comp),
                    "auto": n_carriles is None}
        for e, k in zip(ev_comp, etiqueta):
            e["carril"] = numero[int(k)]

    # ---- Figuras ----
    ruta_calor = salida / "mapa_calor.png"
    figura_mapa_calor(compatibles, carriles, ruta_calor)
    ruta_horas = salida / "vehiculos_por_hora.png"
    info_horas = figura_por_hora(eventos, sesiones, ruta_horas)

    # ---- Tablas ----
    with open(salida / "vehiculos_contados.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["sesion", "id", "fecha_hora", "t_video_s", "clase", "sentido", "carril", "d_m", "vel_kmh"])
        for e in eventos:
            w.writerow([e["sesion"], e["id"], e["fecha"].isoformat(sep=" ", timespec="seconds") if e["fecha"] else "",
                        f"{e['t_video']:.2f}", e["clase"],
                        compatibles[0].contador.flecha(e["sentido"], unicode=True),
                        e.get("carril", ""), f"{e['d_m']:.2f}",
                        "" if e["vel_kmh"] is None else f"{e['vel_kmh']:.1f}"])
    if carriles:
        with open(salida / "carriles.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["carril", "sentido", "vehiculos", "porcentaje", "vel_media_kmh", "d_centro_m"])
            for r in carriles["filas"]:
                w.writerow([r["carril"], r["flecha"], r["vehiculos"], f"{r['pct']:.1f}",
                            "" if r["vel_media"] is None else f"{r['vel_media']:.1f}", f"{r['d_centro']:.2f}"])

    # ---- Resumen en texto ----
    L = ["RESUMEN - ANALITICA DE TRANSITO (Nivel 2)", "=" * 60]
    for s in sesiones:
        if s.intervalo:
            a, b = s.intervalo
            L.append(f"Sesión {s.nombre}: {a:%Y-%m-%d %H:%M:%S} -> {b:%H:%M:%S} "
                     f"({_dur(s.duracion_s / 60)}) | hora: {s.meta.get('fuente_hora', '?')}")
        else:
            L.append(f"Sesión {s.nombre}: {_dur(s.duracion_s / 60)} | SIN hora real "
                     "(ejecuta con --hora-inicio \"AAAA-MM-DD HH:MM\")")
    por_clase = Counter(e["clase"] for e in eventos)
    L += ["", f"Vehículos contados (cruzaron la línea de conteo): {len(eventos)}",
          "  " + " | ".join(f"{c}: {por_clase[c]}" for c in ORDEN_CLASES if por_clase[c])]
    if carriles:
        L += ["", f"CARRILES ({len(carriles['filas'])}, "
                  f"{'detectados automáticamente' if carriles['auto'] else 'número indicado'}; "
                  "carril 1 = el de más arriba o más a la izquierda en la imagen)"]
        for r in carriles["filas"]:
            v = f"{r['vel_media']:.1f} km/h" if r["vel_media"] is not None else "-"
            L.append(f"  Carril {r['carril']} {r['flecha']}  {r['vehiculos']:>5} veh  {r['pct']:5.1f}%"
                     f"  vel. media {v}" + ("   <-- MÁS USADO" if r["carril"] == carriles["mas_usado"] else ""))
    if info_horas["tabla"]:
        L += ["", "VEHÍCULOS POR HORA"]
        for r in info_horas["tabla"]:
            L.append(f"  {r['hora']:%Y-%m-%d %H:00}  {r['vehiculos']:>5} veh  "
                     f"(grabado {_dur(r['minutos'])} -> ≈ {r['tasa']:.0f} veh/h)")
        if info_horas["pico"]:
            p = info_horas["pico"]
            L.append(f"  Hora con más vehículos: {p['hora']:%H:%M}-{p['hora'] + timedelta(hours=1):%H:%M}"
                     " (comparando el flujo por hora, porque cada hora puede tener distinto tiempo grabado)")
    if avisos:
        L += [""] + avisos
    L += ["", "Archivos: mapa_calor.png, vehiculos_por_hora.png, vehiculos_contados.csv, carriles.csv"]
    texto = "\n".join(L)
    (salida / "resumen_nivel2.txt").write_text(texto + "\n", encoding="utf-8")
    if imprimir:
        print("\n" + texto)
    return {"carriles": carriles, "horas": info_horas, "eventos": eventos,
            "archivos": [ruta_calor, ruta_horas], "avisos": avisos}
