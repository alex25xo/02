"""
PARTE 3 - Analisis de las trayectorias guardadas (ID, frame, centro X, Y)
=======================================================================
Lee el trayectorias.csv que produce la Parte 2 (o la Parte 4) y responde preguntas:
  * por donde se movio cada objeto            -> trayectorias.png
  * como evolucionan en el tiempo             -> cubo_espacio_tiempo.png  (x, y, t en 3D)
  * hacia donde va la gente / los autos       -> direcciones.png          (rosa de rumbos)
  * de donde vienen y a donde van             -> origen_destino.png       (matriz O-D por zonas)
  * cuales son los flujos principales         -> patrones.png             (k-means de trayectorias)
  * quien se comporta distinto                -> anomalias.png            (contraflujo, permanencia, rapidez)
  * cuanta actividad hay en el tiempo         -> actividad.png
  * tabla por objeto                          -> estadisticas_por_id.csv

Ejemplo:
  python parte3_analisis_trayectorias.py --carpeta salidas/parte2/peatones_vtest_bytetrack
  python parte3_analisis_trayectorias.py --csv mi_tracking.csv --fondo foto.jpg --k 3
"""
import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold", "figure.dpi": 110})


def color_id(tid):
    return hsv_to_rgb([(tid * 0.61803398875) % 1.0, 0.75, 0.95])


def suavizar(xy, k=5):
    if len(xy) < k:
        return xy
    ker = np.ones(k) / k
    pad = np.pad(xy, ((k // 2, k // 2), (0, 0)), mode="edge")
    return np.column_stack([np.convolve(pad[:, 0], ker, "valid"), np.convolve(pad[:, 1], ker, "valid")])


def remuestrear(xy, n=16):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    if d[-1] < 1e-6:
        return np.repeat(xy[:1], n, axis=0)
    s = np.linspace(0, d[-1], n)
    return np.column_stack([np.interp(s, d, xy[:, 0]), np.interp(s, d, xy[:, 1])])


def zona(p, w, h, filas=3, cols=3):
    i = min(int(p[1] / h * filas), filas - 1)
    j = min(int(p[0] / w * cols), cols - 1)
    return "ABC"[i] + str(j + 1)


def rumbo_texto(ang):
    # angulo en grados con eje x a la derecha y eje y hacia ARRIBA de la imagen
    nombres = ["derecha", "arriba-der", "arriba", "arriba-izq", "izquierda", "abajo-izq", "abajo", "abajo-der"]
    return nombres[int(((ang + 22.5) % 360) // 45)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--carpeta", help="carpeta de salida de la Parte 2 (contiene trayectorias.csv y fondo.jpg)")
    ap.add_argument("--csv")
    ap.add_argument("--fondo", help="imagen de fondo (por defecto fondo.jpg de la carpeta)")
    ap.add_argument("--min-frames", type=int, default=10, help="descarta IDs muy cortos (ruido)")
    ap.add_argument("--k", type=int, default=4, help="numero de patrones (flujos) a buscar")
    ap.add_argument("--salida", default=None)
    a = ap.parse_args()

    ruta_csv = a.csv or os.path.join(a.carpeta, "trayectorias.csv")
    salida = a.salida or os.path.join(os.path.dirname(ruta_csv), "analisis")
    os.makedirs(salida, exist_ok=True)
    df = pd.read_csv(ruta_csv)
    ruta_fondo = a.fondo or os.path.join(os.path.dirname(ruta_csv), "fondo.jpg")
    fondo = cv2.imread(ruta_fondo) if os.path.exists(ruta_fondo) else None
    if fondo is not None:
        fondo = cv2.cvtColor(fondo, cv2.COLOR_BGR2RGB)
        h, w = fondo.shape[:2]
    else:
        w, h = int(df["x2"].max()) + 10, int(df["y2"].max()) + 10
        fondo = np.full((h, w, 3), 235, np.uint8)
    eventos = None
    ruta_ev = os.path.join(os.path.dirname(ruta_csv), "conteo_eventos.csv")
    if os.path.exists(ruta_ev):
        eventos = pd.read_csv(ruta_ev)
    fps = 1.0 / np.median(np.diff(np.sort(df["t_s"].unique()))) if df["t_s"].nunique() > 1 else 25

    # ------------------------------------------------------------------ estadisticas por objeto
    tray, filas = {}, []
    for tid, g in df.groupby("id"):
        g = g.sort_values("frame")
        if len(g) < a.min_frames:
            continue
        xy = suavizar(g[["cx", "cy"]].to_numpy(float))
        tray[tid] = (g["t_s"].to_numpy(), xy, g["frame"].to_numpy())
        pasos = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        largo = float(pasos.sum())
        neto = xy[-1] - xy[0]
        dur = float(g["t_s"].iloc[-1] - g["t_s"].iloc[0])
        ang = float(np.degrees(np.arctan2(-neto[1], neto[0])) % 360)
        filas.append({"id": tid, "clase": g["clase"].mode()[0], "frames": len(g),
                      "t_inicio_s": round(g["t_s"].iloc[0], 2), "t_fin_s": round(g["t_s"].iloc[-1], 2),
                      "duracion_s": round(dur, 2), "longitud_px": round(largo, 1),
                      "desplazamiento_neto_px": round(float(np.linalg.norm(neto)), 1),
                      "rectitud": round(float(np.linalg.norm(neto) / largo), 2) if largo > 0 else 0,
                      "vel_media_px_s": round(largo / dur, 1) if dur > 0 else 0,
                      "rumbo_deg": round(ang, 1), "direccion": rumbo_texto(ang),
                      "zona_entrada": zona(xy[0], w, h), "zona_salida": zona(xy[-1], w, h),
                      "cruzo_linea": (eventos["sentido"][eventos["id"] == tid].iloc[0]
                                      if eventos is not None and (eventos["id"] == tid).any() else "")})
    est = pd.DataFrame(filas)
    est.to_csv(os.path.join(salida, "estadisticas_por_id.csv"), index=False)
    print(f"Objetos analizados: {len(est)} (de {df['id'].nunique()} IDs; min {a.min_frames} frames)")

    def base(ax, titulo):
        ax.imshow(fondo)
        ax.set_title(titulo)
        ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")

    # ------------------------------------------------------------------ 1. trayectorias
    fig, ax = plt.subplots(figsize=(9, 9 * h / w + 0.4))
    base(ax, f"Trayectorias de {len(tray)} objetos")
    for tid, (t, xy, _) in tray.items():
        c = color_id(tid)
        ax.plot(xy[:, 0], xy[:, 1], "-", color=c, lw=1.8, alpha=0.9)
        ax.plot(*xy[0], "o", color=c, ms=5, mec="k", mew=0.5)
        if len(xy) > 3:
            ax.annotate("", xy=xy[-1], xytext=xy[-4], arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8))
        ax.text(xy[0, 0] + 3, xy[0, 1] - 3, str(tid), fontsize=7, color="white",
                bbox=dict(boxstyle="round,pad=0.15", fc=c, ec="none", alpha=0.85))
    fig.tight_layout(); fig.savefig(os.path.join(salida, "trayectorias.png")); plt.close(fig)

    # ------------------------------------------------------------------ 2. cubo espacio-tiempo
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    small = cv2.resize(fondo, (80, int(80 * h / w)))
    X, Y = np.meshgrid(np.linspace(0, w, small.shape[1]), np.linspace(0, h, small.shape[0]))
    ax.plot_surface(X, Y, np.zeros_like(X), facecolors=small / 255.0, rstride=1, cstride=1, shade=False, alpha=0.6)
    for tid, (t, xy, _) in tray.items():
        ax.plot(xy[:, 0], xy[:, 1], t, color=color_id(tid), lw=1.6)
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)"); ax.set_zlabel("tiempo (s)")
    ax.set_ylim(h, 0); ax.view_init(elev=24, azim=-58)
    ax.set_title("Cubo espacio-tiempo: cada linea es un ID")
    fig.tight_layout(); fig.savefig(os.path.join(salida, "cubo_espacio_tiempo.png")); plt.close(fig)

    # ------------------------------------------------------------------ 3. rosa de direcciones
    mov = est[est["desplazamiento_neto_px"] > 0.15 * min(w, h)]
    fig = plt.figure(figsize=(5.5, 5.5))
    ax = fig.add_subplot(111, projection="polar")
    bins = np.radians(np.arange(0, 361, 30))
    hist, _ = np.histogram(np.radians(mov["rumbo_deg"]), bins=bins)
    ax.bar(bins[:-1] + np.radians(15), hist, width=np.radians(28), color="#2563EB", alpha=0.85, edgecolor="white")
    ax.set_title(f"Rumbo de {len(mov)} objetos que se desplazan", pad=14)
    ax.set_xticks(np.radians([0, 90, 180, 270])); ax.set_xticklabels(["derecha", "arriba", "izquierda", "abajo"])
    fig.tight_layout(); fig.savefig(os.path.join(salida, "direcciones.png")); plt.close(fig)

    # ------------------------------------------------------------------ 4. matriz origen-destino
    zonas_n = [f + str(c) for f in "ABC" for c in (1, 2, 3)]
    od = pd.crosstab(est["zona_entrada"], est["zona_salida"]).reindex(index=zonas_n, columns=zonas_n, fill_value=0)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    base(axs[0], "Zonas (rejilla 3x3)")
    for i in range(1, 3):
        axs[0].axhline(i * h / 3, color="w", lw=1); axs[0].axvline(i * w / 3, color="w", lw=1)
    for z in zonas_n:
        i, j = "ABC".index(z[0]), int(z[1]) - 1
        axs[0].text((j + 0.5) * w / 3, (i + 0.5) * h / 3, z, color="w", ha="center", va="center", fontsize=16,
                    fontweight="bold", bbox=dict(fc="k", alpha=0.35, ec="none"))
    im = axs[1].imshow(od.values, cmap="Blues")
    axs[1].set_xticks(range(9)); axs[1].set_xticklabels(zonas_n); axs[1].set_yticks(range(9)); axs[1].set_yticklabels(zonas_n)
    axs[1].set_xlabel("zona de SALIDA"); axs[1].set_ylabel("zona de ENTRADA"); axs[1].set_title("Matriz origen-destino")
    for i in range(9):
        for j in range(9):
            if od.values[i, j]:
                axs[1].text(j, i, od.values[i, j], ha="center", va="center",
                            color="white" if od.values[i, j] > od.values.max() / 2 else "black", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(salida, "origen_destino.png")); plt.close(fig)

    # ------------------------------------------------------------------ 5. patrones (k-means de trayectorias)
    ids_mov = [tid for tid in mov["id"] if tid in tray]
    etiquetas = {}
    k = min(a.k, len(ids_mov))
    if k >= 2:
        datos = np.float32([remuestrear(tray[tid][1]).ravel() / np.tile([w, h], 16) for tid in ids_mov])
        _, lab, centros = cv2.kmeans(datos, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4),
                                     10, cv2.KMEANS_PP_CENTERS)
        etiquetas = dict(zip(ids_mov, lab.ravel()))
        paleta = ["#2563EB", "#F97316", "#16A34A", "#7C3AED", "#DC2626", "#0EA5A4", "#CA8A04", "#DB2777"]
        fig, ax = plt.subplots(figsize=(9, 9 * h / w + 0.4))
        base(ax, f"{k} flujos principales (k-means sobre trayectorias remuestreadas)")
        for tid, c in etiquetas.items():
            xy = tray[tid][1]
            ax.plot(xy[:, 0], xy[:, 1], color=paleta[c % 8], lw=1, alpha=0.55)
        for c in range(k):
            m = centros[c].reshape(16, 2) * [w, h]
            n = int((lab.ravel() == c).sum())
            ax.plot(m[:, 0], m[:, 1], color="k", lw=7, alpha=0.5)
            ax.plot(m[:, 0], m[:, 1], color=paleta[c % 8], lw=4.5)
            ax.annotate("", xy=m[-1], xytext=m[-3], arrowprops=dict(arrowstyle="-|>,head_width=0.6,head_length=1",
                                                                     color=paleta[c % 8], lw=4.5))
            ax.text(*m[int(len(m) * (0.25 + 0.5 * c / max(k - 1, 1)))], f" flujo {c + 1}: {n} ", color="white", fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round", fc=paleta[c % 8], ec="none"))
        fig.tight_layout(); fig.savefig(os.path.join(salida, "patrones.png")); plt.close(fig)
    est["flujo"] = est["id"].map(lambda i: int(etiquetas[i]) + 1 if i in etiquetas else "")

    # ------------------------------------------------------------------ 6. anomalias
    anom = {}
    if len(est) > 3:
        p90 = est["duracion_s"].quantile(0.9)
        for _, r in est.iterrows():
            if r["duracion_s"] >= p90 and r["rectitud"] < 0.5:
                anom[r["id"]] = "permanencia / merodeo"
        v = est["vel_media_px_s"]
        z = (v - v.median()) / (1.4826 * (v - v.median()).abs().median() + 1e-6)
        for tid, zz in zip(est["id"], z):
            if zz > 3:
                anom.setdefault(tid, "mucho mas rapido que el resto")
        if etiquetas:
            for c in set(etiquetas.values()):
                miembros = [t for t, l in etiquetas.items() if l == c]
                if len(miembros) < 3:
                    for t in miembros:
                        anom.setdefault(t, "trayectoria poco comun")
    est["anomalia"] = est["id"].map(lambda i: anom.get(i, ""))
    fig, ax = plt.subplots(figsize=(9, 9 * h / w + 0.4))
    base(ax, f"Comportamientos a revisar: {len(anom)}")
    for tid, (t, xy, _) in tray.items():
        ax.plot(xy[:, 0], xy[:, 1], color="white", lw=1, alpha=0.35)
    colores = {"permanencia / merodeo": "#F97316", "mucho mas rapido que el resto": "#DC2626",
               "trayectoria poco comun": "#7C3AED"}
    for tid, motivo in anom.items():
        xy = tray[tid][1]
        ax.plot(xy[:, 0], xy[:, 1], color=colores[motivo], lw=3)
        ax.text(xy[-1, 0], xy[-1, 1], f" #{tid}", color="white", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc=colores[motivo], ec="none"))
    for m, c in colores.items():
        ax.plot([], [], color=c, lw=3, label=m)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(os.path.join(salida, "anomalias.png")); plt.close(fig)

    # ------------------------------------------------------------------ 7. actividad en el tiempo
    fig, ax = plt.subplots(figsize=(9, 3.4))
    seg = df.groupby(df["t_s"].astype(int))["id"].nunique()
    ax.fill_between(seg.index, seg.values, step="mid", color="#93C5FD", alpha=0.8, label="objetos visibles")
    ax.set_xlabel("tiempo (s)"); ax.set_ylabel("objetos visibles")
    if eventos is not None and len(eventos):
        ax2 = ax.twinx()
        ev = eventos.sort_values("t_s")
        ax2.step(ev["t_s"], np.arange(1, len(ev) + 1), where="post", color="#F97316", lw=2.2, label="cruces acumulados")
        ax2.set_ylabel("cruces acumulados", color="#F97316")
    ax.set_title("Actividad en el tiempo")
    fig.tight_layout(); fig.savefig(os.path.join(salida, "actividad.png")); plt.close(fig)

    est.to_csv(os.path.join(salida, "estadisticas_por_id.csv"), index=False)
    resumen = {
        "objetos_analizados": int(len(est)),
        "duracion_media_s": round(float(est["duracion_s"].mean()), 2),
        "longitud_media_px": round(float(est["longitud_px"].mean()), 1),
        "vel_media_px_s": round(float(est["vel_media_px_s"].median()), 1),
        "direccion_mas_comun": est["direccion"].mode()[0] if len(est) else "",
        "ruta_mas_comun": (est["zona_entrada"] + "->" + est["zona_salida"]).mode()[0] if len(est) else "",
        "anomalias": {int(k): v for k, v in anom.items()},
    }
    json.dump(resumen, open(os.path.join(salida, "resumen_analisis.json"), "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"Figuras y tablas en: {salida}")


if __name__ == "__main__":
    main()
