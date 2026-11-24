"""
PARTE 1 - ¿Por que la odometria visual (VO) / Pose Tracking 3D es mas dificil que el tracking 2D?
================================================================================================
Tres experimentos numericos (no necesitan video) + una demostracion en vivo:

  python parte1_dificultad_vo.py --modo simulacion
     A) Mismo ruido de pixeles, distinto destino: en tracking 2D el error se queda acotado; en VO
        el error de cada frame se ACUMULA (deriva) y la trayectoria no se cierra.
     B) Ambiguedad de escala: una escena 2x mas grande y 2x mas lejos produce EXACTAMENTE la misma imagen.
     C) Mundo dinamico: si muchos puntos pertenecen a objetos que se mueven, la VO estima mal SU propio
        movimiento (el tracking 2D no sufre esto: justamente sigue a los objetos que se mueven).

  python parte1_dificultad_vo.py --modo vivo --fuente 0
     VO monocular en vivo con la camara web: mueva la camara y vuelva al punto de partida:
     la trayectoria estimada NO regresa a cero (deriva) y no tiene unidades (escala desconocida).
     Con --yolo se muestra a la vez el tracking 2D, que sigue funcionando sin problemas.
"""
import argparse
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold", "figure.dpi": 110})
W, H_IMG, F = 640, 480, 600.0
K = np.array([[F, 0, W / 2], [0, F, H_IMG / 2], [0, 0, 1]])


def proyectar(Pw, R, C):
    """Pw: puntos del mundo (N,3); R: rotacion mundo->camara; C: centro de la camara."""
    Pc = (Pw - C) @ R.T
    uv = Pc @ K.T
    uv = uv[:, :2] / uv[:, 2:3]
    vis = (Pc[:, 2] > 0.5) & (uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H_IMG)
    return uv, vis


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])      # camara mirando a lo largo de su eje z


def trayectoria_circular(n=360, radio=20.0):
    ang = np.linspace(0, 2 * np.pi, n)
    C = np.column_stack([radio * (1 - np.cos(ang)), np.zeros(n), radio * np.sin(ang)])
    yaw = ang                                                    # mira tangente al circulo
    return C, yaw


def vo_entre(Pw, R0, C0, R1, C1, sigma, rng, extra=None):
    uv0, v0 = proyectar(Pw, R0, C0)
    uv1, v1 = proyectar(Pw, R1, C1)
    m = v0 & v1
    a = uv0[m] + rng.normal(0, sigma, (m.sum(), 2))
    b = uv1[m] + rng.normal(0, sigma, (m.sum(), 2))
    if extra is not None:
        a, b = np.vstack([a, extra[0]]), np.vstack([b, extra[1]])
    E, msk = cv2.findEssentialMat(a, b, K, cv2.RANSAC, 0.999, 1.0)
    if E is None or E.shape != (3, 3):
        return None, None
    _, R, t, _ = cv2.recoverPose(E, a, b, K, mask=msk)
    return R, t.ravel()


def experimento_deriva(salida, rng):
    Pw = np.column_stack([rng.uniform(-25, 65, 4000), rng.uniform(-6, 6, 4000), rng.uniform(-45, 45, 4000)])
    dist = np.linalg.norm(Pw[:, [0, 2]] - [20, 0], axis=1)
    Pw = Pw[(dist > 26) | (dist < 14)]                          # deja libre el "camino"
    C, yaw = trayectoria_circular()
    resultados = {}
    for sigma in (0.5, 1.0, 2.0):
        R_est, C_est = rot_y(yaw[0]), C[0].copy()
        est = [C_est.copy()]
        for k in range(1, len(C)):
            R0, R1 = rot_y(yaw[k - 1]), rot_y(yaw[k])
            Rrel, t = vo_entre(Pw, R0, C[k - 1], R1, C[k], sigma, rng)
            if Rrel is None:
                est.append(C_est.copy()); continue
            paso = np.linalg.norm(C[k] - C[k - 1])               # escala "oraculo": aun asi hay deriva
            # x1 = Rrel x0 + t  ->  centro de la camara 1 en el sistema de la camara 0 = -Rrel^T t
            C_est = C_est + R_est.T @ (-Rrel.T @ t) * paso
            R_est = Rrel @ R_est
            est.append(C_est.copy())
        resultados[sigma] = np.array(est)
        print(f"  sigma={sigma} px -> error final {np.linalg.norm(resultados[sigma][-1] - C[-1]):.2f} m "
              f"(recorrido {2 * np.pi * 20:.0f} m)")

    # tracking 2D con el mismo ruido: filtro de Kalman de velocidad constante sobre el centro de una caja
    n = len(C)
    t = np.arange(n)
    verdad = np.column_stack([320 + 200 * np.sin(t / 60), 240 + 120 * np.sin(t / 37)])
    err2d = {}
    for sigma in (0.5, 1.0, 2.0):
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * sigma ** 2
        kf.statePost = np.float32([[verdad[0, 0]], [verdad[0, 1]], [0], [0]])
        e = []
        for k in range(n):
            kf.predict()
            z = (verdad[k] + rng.normal(0, sigma, 2)).astype(np.float32).reshape(2, 1)
            s = kf.correct(z)
            e.append(np.linalg.norm(s[:2, 0] - verdad[k]))
        err2d[sigma] = np.array(e)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    ax = axs[0]
    ax.plot(C[:, 0], C[:, 2], "k-", lw=3, label="trayectoria real (circulo de 20 m)")
    colores = {0.5: "#16A34A", 1.0: "#F97316", 2.0: "#DC2626"}
    for s, e in resultados.items():
        ax.plot(e[:, 0], e[:, 2], "-", color=colores[s], lw=2, label=f"VO con ruido {s} px")
        ax.plot(*e[-1, [0, 2]], "o", color=colores[s], ms=8)
    ax.plot(*C[0, [0, 2]], "k*", ms=16, label="inicio = fin real")
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ax.set_title("VO: la vuelta no se cierra")
    ax.legend(fontsize=7.5, loc="best")
    for s, e in resultados.items():
        axs[1].plot(np.arange(n), np.linalg.norm(e - C, axis=1), "-", color=colores[s], lw=2, label=f"ruido {s} px")
        axs[2].plot(np.arange(n), err2d[s], "-", color=colores[s], lw=1.2, alpha=0.9, label=f"ruido {s} px")
    axs[1].set_title("VO: error de posicion (m) CRECE")
    axs[1].set_xlabel("frame"); axs[1].set_ylabel("error (m)"); axs[1].legend(fontsize=8)
    axs[2].set_title("Tracking 2D: error (px) ACOTADO")
    axs[2].set_xlabel("frame"); axs[2].set_ylabel("error (px)"); axs[2].legend(fontsize=8)
    axs[2].set_ylim(0, max(axs[1].get_ylim()[1], 4))
    fig.tight_layout(); fig.savefig(os.path.join(salida, "A_deriva_vo_vs_2d.png")); plt.close(fig)
    return resultados, err2d


def experimento_escala(salida, rng):
    Pw = np.column_stack([rng.uniform(-3, 3, 60), rng.uniform(-2, 2, 60), rng.uniform(8, 14, 60)])
    R0, C0 = np.eye(3), np.zeros(3)
    R1, C1 = rot_y(0.05), np.array([0.8, 0, 0.3])
    fig, axs = plt.subplots(1, 4, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
    for i, s in enumerate((0.5, 1.0, 2.0)):
        uv0, _ = proyectar(Pw * s, R0, C0 * s)
        uv1, _ = proyectar(Pw * s, R1, C1 * s)
        if i == 0:
            ref0, ref1 = uv0, uv1
        dif = max(np.abs(uv0 - ref0).max(), np.abs(uv1 - ref1).max())
        ax = axs[i]
        ax.scatter(Pw[:, 0] * s, Pw[:, 2] * s, s=10, c="#2563EB", label="puntos 3D")
        for Cc, col in ((C0 * s, "#0F2547"), (C1 * s, "#F97316")):
            ax.plot([Cc[0], Cc[0] - 3], [Cc[2], Cc[2] + 5], color=col, lw=0.8)
            ax.plot([Cc[0], Cc[0] + 3], [Cc[2], Cc[2] + 5], color=col, lw=0.8)
            ax.plot(Cc[0], Cc[2], "s", color=col, ms=7)
        ax.annotate("", xy=(C1[0] * s, C1[2] * s), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="r", lw=2))
        ax.set_xlim(-8, 8); ax.set_ylim(-1, 30); ax.set_aspect("equal")
        ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)" if i == 0 else "")
        ax.set_title(f"mundo x{s}: camara se mueve {np.linalg.norm(C1) * s:.2f} m\ndiferencia en la imagen: {dif:.0e} px",
                     fontsize=9)
    ax = axs[3]
    ax.scatter(ref0[:, 0], ref0[:, 1], s=9, c="#0F2547", label="frame 1")
    ax.scatter(ref1[:, 0], ref1[:, 1], s=9, c="#F97316", label="frame 2")
    ax.set_xlim(0, W); ax.set_ylim(H_IMG, 0); ax.set_aspect("equal"); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("lo que ve la camara\n(identico en los 3 mundos)", fontsize=9)
    fig.suptitle("B) Ambiguedad de escala: con UNA camara, mundo pequeno y cercano = mundo grande y lejano",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(salida, "B_ambiguedad_escala.png")); plt.close(fig)


def experimento_dinamico(salida, rng):
    Pw = np.column_stack([rng.uniform(-10, 10, 600), rng.uniform(-3, 3, 600), rng.uniform(6, 30, 600)])
    R0, C0 = np.eye(3), np.zeros(3)
    R1, C1 = np.eye(3), np.array([0, 0, 0.5])                  # la camara avanza 0.5 m
    fracciones = np.linspace(0, 0.8, 17)
    err = []
    for f in fracciones:
        e = []
        for _ in range(20):
            n_mov = int(f / (1 - f + 1e-9) * 400) if f < 1 else 0
            Pm = np.column_stack([rng.uniform(-4, 4, n_mov), rng.uniform(-2, 1, n_mov), rng.uniform(8, 14, n_mov)])
            # un objeto rigido grande (bus) que se mueve 1.2 m a la derecha entre los dos frames
            uv0, v0 = proyectar(Pm, R0, C0)
            uv1, v1 = proyectar(Pm + [1.2, 0, 0], R1, C1)
            m = v0 & v1
            extra = (uv0[m], uv1[m])
            idx = rng.choice(len(Pw), 400, replace=False)
            _, t = vo_entre(Pw[idx], R0, C0, R1, C1, 0.5, rng, extra)
            if t is None:
                continue
            real = -(C1 - C0) / np.linalg.norm(C1 - C0)       # traslacion esperada en el sistema de la camara
            e.append(np.degrees(np.arccos(np.clip(abs(t @ real), -1, 1))))
        err.append(np.median(e))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(100 * fracciones, err, "o-", color="#DC2626", lw=2.2)
    corte = next((100 * f for f, e in zip(fracciones, err) if e > 20), None)
    if corte is not None:
        ax.axvline(corte, color="k", ls="--", lw=1)
        ax.text(corte + 1.5, max(err) * 0.45, f"desde ~{corte:.0f}% de puntos sobre un objeto\ngrande en movimiento, RANSAC\nelige el movimiento EQUIVOCADO", fontsize=9)
    ax.set_xlabel("% de puntos sobre objetos en movimiento")
    ax.set_ylabel("error en la direccion de\nmovimiento de la camara (grados)")
    ax.set_title("C) La VO supone un mundo estatico")
    fig.tight_layout(); fig.savefig(os.path.join(salida, "C_mundo_dinamico.png")); plt.close(fig)
    for f, e in zip(fracciones, err):
        print(f"  {100 * f:4.0f}% puntos moviles -> error {e:6.2f} grados")


def modo_vivo(a):
    import time
    cap = cv2.VideoCapture(int(a.fuente), cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY) if a.fuente.isdigit() \
        else cv2.VideoCapture(a.fuente)
    ok, f = cap.read()
    if not ok:
        raise SystemExit("No se pudo abrir la fuente")
    h, w = f.shape[:2]
    foco = a.foco or 0.9 * w
    Kv = np.array([[foco, 0, w / 2], [0, foco, h / 2], [0, 0, 1]])
    yolo = None
    if a.yolo:
        from comun import TrackerYOLO, dibujar_objeto
        yolo = TrackerYOLO("yolo11n.pt", "bytetrack", ["person"], 0.3, 480)
    prev = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    R, C = np.eye(3), np.zeros(3)
    tray = [C.copy()]
    pts = cv2.goodFeaturesToTrack(prev, 1000, 0.01, 8)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        vis = f.copy()
        n_inl = 0
        if pts is not None and len(pts) > 30:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None, winSize=(21, 21), maxLevel=3)
            a0, a1 = pts[st[:, 0] == 1].reshape(-1, 2), p1[st[:, 0] == 1].reshape(-1, 2)
            flujo = np.median(np.linalg.norm(a1 - a0, axis=1)) if len(a0) else 0
            if len(a0) > 30 and flujo > 1.5:                    # solo si hubo movimiento suficiente
                E, m = cv2.findEssentialMat(a0, a1, Kv, cv2.RANSAC, 0.999, 1.0)
                if E is not None and E.shape == (3, 3):
                    n_inl, Rr, t, m2 = cv2.recoverPose(E, a0, a1, Kv, mask=m)
                    if n_inl > 20:
                        C = C + R.T @ (-Rr.T @ t.ravel())       # paso de longitud 1 (escala desconocida)
                        R = Rr @ R
                        tray.append(C.copy())
                        for (x0, y0), (x1, y1) in zip(a0[m2.ravel() > 0], a1[m2.ravel() > 0]):
                            cv2.line(vis, (int(x0), int(y0)), (int(x1), int(y1)), (60, 220, 60), 1, cv2.LINE_AA)
                pts = a1.reshape(-1, 1, 2)
                prev = g
            if len(pts) < 300:
                pts = cv2.goodFeaturesToTrack(g, 1000, 0.01, 8)
                prev = g
        else:
            pts = cv2.goodFeaturesToTrack(g, 1000, 0.01, 8); prev = g
        if yolo is not None:
            for o in yolo.procesar(f):
                dibujar_objeto(vis, o)
        # mapa de la trayectoria (vista superior x-z)
        mapa = np.full((260, 260, 3), 30, np.uint8)
        T = np.array(tray)
        esc = 100 / max(5, np.abs(T[:, [0, 2]]).max())
        pts_m = np.int32(np.column_stack([130 + T[:, 0] * esc, 130 - T[:, 2] * esc]))
        cv2.polylines(mapa, [pts_m.reshape(-1, 1, 2)], False, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(mapa, (130, 130), (255, 255, 255), cv2.MARKER_STAR, 14, 2)
        cv2.circle(mapa, tuple(pts_m[-1]), 5, (0, 0, 255), -1)
        cv2.putText(mapa, f"dist. al inicio: {np.linalg.norm(C):.1f} (sin unidades)", (6, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        vis[10:270, w - 270:w - 10] = mapa
        cv2.putText(vis, f"VO monocular | inliers {n_inl} | vuelva al inicio y mire la deriva | r: reiniciar",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Parte 1 - VO en vivo", vis)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        if k == ord("r"):
            R, C, tray = np.eye(3), np.zeros(3), [np.zeros(3)]
    cap.release(); cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modo", default="simulacion", choices=["simulacion", "vivo"])
    ap.add_argument("--fuente", default="0")
    ap.add_argument("--foco", type=float, default=None, help="distancia focal en px (si se calibro la camara)")
    ap.add_argument("--yolo", action="store_true", help="en modo vivo, mostrar tambien el tracking 2D")
    ap.add_argument("--salida", default=os.path.join("salidas", "parte1"))
    a = ap.parse_args()
    if a.modo == "vivo":
        return modo_vivo(a)
    os.makedirs(a.salida, exist_ok=True)
    rng = np.random.default_rng(7)
    print("A) deriva acumulada (VO) vs error acotado (tracking 2D)")
    experimento_deriva(a.salida, rng)
    print("B) ambiguedad de escala")
    experimento_escala(a.salida, rng)
    print("C) mundo dinamico")
    experimento_dinamico(a.salida, rng)
    print(f"Figuras en {a.salida}")


if __name__ == "__main__":
    main()
