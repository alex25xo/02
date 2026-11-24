"""Gráficos del NIVEL 3: trayectoria de la cámara y comparación de velocidades."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .analitica import APAGADO, COLOR_CLASE, REJILLA, TINTA, TINTA2, plt  # estilo común

AZUL, NARANJA, AGUA = COLOR_CLASE["auto"], COLOR_CLASE["moto"], COLOR_CLASE["bus"]


def _ejes_limpios(ax):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.grid(color=REJILLA, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=9)


def figura_odometria(trayectoria, historial, vehiculos: dict, ruta: Path, titulo_sub: str) -> None:
    """
    trayectoria: [(t, x, y)] de la cámara en el mundo
    historial:   [ResultadoVO]
    vehiculos:   id -> {"estado": "estacionado"/"en movimiento", "mundo": Nx2, "clase": str}
    """
    tr = np.array(trayectoria)
    fig, (ax, axv) = plt.subplots(1, 2, figsize=(13, 6.2), dpi=150,
                                  gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.22})
    fig.subplots_adjust(left=0.06, right=0.98, top=0.8, bottom=0.11)

    # ---- Mapa (vista desde arriba, metros) ----
    ax.plot(tr[:, 1], tr[:, 2], color=AZUL, lw=2, label="cámara", zorder=3)
    ax.scatter(tr[0, 1], tr[0, 2], s=60, color=AZUL, edgecolor="white", lw=1.5, zorder=4)
    ax.annotate("inicio", (tr[0, 1], tr[0, 2]), xytext=(8, -4), textcoords="offset points",
                fontsize=9, color=TINTA2)
    ax.scatter(tr[-1, 1], tr[-1, 2], s=90, marker="^", color=AZUL, edgecolor="white", lw=1.5, zorder=4)
    ax.annotate("fin", (tr[-1, 1], tr[-1, 2]), xytext=(8, -4), textcoords="offset points",
                fontsize=9, color=TINTA2)
    est = [v for v in vehiculos.values() if v["estado"] == "estacionado"]
    mov = [v for v in vehiculos.values() if v["estado"] == "en movimiento"]
    if est:
        p = np.array([np.median(v["mundo"], axis=0) for v in est])
        ax.scatter(p[:, 0], p[:, 1], s=70, marker="s", color=NARANJA, edgecolor="white", lw=1.5,
                   label="vehículo estacionado", zorder=3)
    for i, v in enumerate(mov):
        m = v["mundo"]
        ax.plot(m[:, 0], m[:, 1], color=AGUA, lw=2, label="vehículo en movimiento" if i == 0 else None,
                zorder=2)
        ax.scatter(m[-1, 0], m[-1, 1], s=50, marker="o", color=AGUA, edgecolor="white", lw=1.5, zorder=3)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("metros (lateral)", fontsize=9)
    ax.set_ylabel("metros (hacia adelante)", fontsize=9)
    ax.set_title("Mapa: trayectoria de la cámara", loc="left", fontsize=11, color=TINTA)
    _ejes_limpios(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="best")

    # ---- Velocidad en el tiempo ----
    historial = [h for h in historial if h.estado != "inicio"]    # el 1.er cuadro no tiene velocidad
    t = np.array([h.t for h in historial])
    v_cam = np.array([h.v_camara_kmh for h in historial])
    v_res = np.array([h.v_suelo_compensada_kmh if h.estado == "ok" else np.nan for h in historial])
    axv.plot(t, v_cam, color=AZUL, lw=2, label="cámara (= lo que el radar del Nivel 1 cree que se mueve el suelo)")
    axv.plot(t, v_res, color=NARANJA, lw=1.2, label="suelo después de compensar (ruido de medición)")
    perdidos = [h.t for h in historial if h.estado == "perdida"]
    if perdidos:
        axv.scatter(perdidos, np.zeros(len(perdidos)), marker="|", s=80, color=APAGADO,
                    label="odometría perdida")
    axv.set_xlabel("tiempo del video (s)", fontsize=9)
    axv.set_ylabel("km/h", fontsize=9)
    axv.set_ylim(bottom=0)
    axv.set_title("Velocidad estimada", loc="left", fontsize=11, color=TINTA)
    _ejes_limpios(axv)
    axv.legend(frameon=False, fontsize=8.5, loc="lower right")

    fig.text(0.06, 0.95, "Odometría visual: ¿cuánto se movió la cámara?", fontsize=14,
             fontweight="bold", va="top")
    fig.text(0.06, 0.895, titulo_sub, fontsize=10, color=TINTA2, va="top")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_comparacion(filas: list[dict], umbral: float, ruta: Path) -> None:
    """filas: [{"id", "clase", "aparente", "real", "estado"}]"""
    n = len(filas)
    fig, ax = plt.subplots(figsize=(10.5, 1.6 + 0.55 * n), dpi=150)
    fig.subplots_adjust(left=0.24, right=0.97, top=1 - 1.25 / (1.6 + 0.55 * n),
                        bottom=0.55 / (1.6 + 0.55 * n))
    y = np.arange(n)
    alto = 0.36
    ap = [f["aparente"] for f in filas]
    re = [f["real"] for f in filas]
    ax.barh(y - alto / 2, ap, height=alto, color=NARANJA, edgecolor="white", lw=1,
            label="aparente: radar del Nivel 1 (sin compensar)")
    ax.barh(y + alto / 2, re, height=alto, color=AZUL, edgecolor="white", lw=1,
            label="real: con odometría visual (Nivel 3)")
    top = max(ap + re + [1.0])
    if n <= 15:
        for yi, a, r in zip(y, ap, re):
            ax.text(a + top * 0.01, yi - alto / 2, f"{a:.1f}", va="center", fontsize=8.5, color=TINTA2)
            ax.text(r + top * 0.01, yi + alto / 2, f"{r:.1f}", va="center", fontsize=8.5, color=TINTA,
                    fontweight="bold")
    ax.axvline(umbral, color=APAGADO, lw=0.8, ls=(0, (3, 3)),
               label=f"umbral de estacionado ({umbral:g} km/h)")
    ax.set_yticks(y, [f"#{f['id']} {f['clase']} · {f['estado']}" for f in filas], fontsize=9, color=TINTA)
    ax.invert_yaxis()
    ax.set_xlim(0, top * 1.15)
    ax.set_xlabel("velocidad típica (mediana, km/h)", fontsize=9)
    _ejes_limpios(ax)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=3)
    h = 1.6 + 0.55 * n
    fig.text(0.02, 1 - 0.25 / h, "Velocidad de cada vehículo: antes y después de compensar",
             fontsize=13, fontweight="bold", va="top")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
