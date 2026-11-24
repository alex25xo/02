"""Pasada final: odometria visual monocular con escala metrica sobre el video real de carretera.

- Rotacion R: matriz esencial (RANSAC) con puntos de todo el entorno, excluyendo los vehiculos (YOLO).
- Traslacion T en metros: puntos del pavimento (10-45 m) cuya posicion 3D se conoce gracias al plano de la
  carretera y la altura de la camara (1,23 m, deducida del ancho del carril de 3,7 m). Cada punto da dos
  ecuaciones LINEALES en T; se resuelve con RANSAC (2 puntos) y se refina con todos los inliers.
"""
import json
import pickle
import numpy as np
import cv2
from scipy.ndimage import median_filter
from ultralytics import YOLO

import vo_carretera as V
import os
AQUI = os.path.dirname(os.path.abspath(__file__))

d1 = json.load(open(os.path.join(AQUI, 'pasada1.json')))
F1 = d1['filas']
h = V.ANCHO_CARRIL / np.median([f['ancho_u'] for f in F1 if 'ancho_u' in f])
N = np.array([f['n_plano'] for f in F1 if 'n_plano' in f and f.get('inl_plano', 0) > 40])
n = np.median(N, axis=0)
n /= np.linalg.norm(n)
K, Kinv = V.K, V.Kinv
ADELANTE = np.array([0, 0, 1.0]) - n * n[2]
ADELANTE /= np.linalg.norm(ADELANTE)
LATERAL = np.cross(n, ADELANTE)

_u, _v = np.meshgrid(np.arange(V.W), np.arange(V.H))
_r = Kinv @ np.vstack([_u.ravel(), _v.ravel(), np.ones(_u.size)])
_den = n @ _r
_lam = np.where(_den > 1e-6, h / np.maximum(_den, 1e-6), np.inf)
_lat = (LATERAL @ _r) * _lam
MASCARA_PAV = (((_lam > 10) & (_lam < 45) & (_lat > -2.6) & (_lat < 6.5)).reshape(V.H, V.W)
               & (V.mascara > 0)).astype(np.uint8) * 255
RNG = np.random.default_rng(1)


def mascara_vehiculos(modelo, img):
    r = modelo.predict(img, verbose=False, conf=0.25, classes=[2, 3, 5, 7], imgsz=960)[0]
    m = np.zeros((V.H, V.W), np.uint8)
    cajas = r.boxes.xyxy.cpu().numpy().astype(int) if len(r.boxes) else np.zeros((0, 4), int)
    for x1, y1, x2, y2 in cajas:
        hh = y2 - y1
        cv2.rectangle(m, (x1 - 10, y1 - 10), (x2 + 10, y2 + int(0.25 * hh) + 10), 255, -1)   # incluye la sombra
    return m, cajas


def traslacion_ransac(R, a, b):
    """T (m) tal que X_prev = R X_curr + T, usando puntos del pavimento a (cuadro anterior) y b (actual)."""
    r = (Kinv @ np.vstack([a.T, np.ones(len(a))])).T
    lam = h / (r @ n)
    Xp = r * lam[:, None]
    A = (R.T @ Xp.T).T                                      # X_curr = A - R^T T
    S = R.T
    xb = (Kinv @ np.vstack([b.T, np.ones(len(b))])).T       # coordenadas normalizadas observadas
    xs, ys = xb[:, 0], xb[:, 1]
    M = np.vstack([xs[:, None] * S[2] - S[0], ys[:, None] * S[2] - S[1]])     # (2N, 3)
    y = np.concatenate([xs * A[:, 2] - A[:, 0], ys * A[:, 2] - A[:, 1]])
    Np = len(a)

    def error_px(T):
        Xc = A - (S @ T)
        p = (K @ Xc.T).T
        return np.linalg.norm(p[:, :2] / p[:, 2:3] - b, axis=1)

    def plausible(T):
        return 0.3 < T @ ADELANTE < 2.6 and abs(T @ LATERAL) < 0.35 and abs(T @ n) < 0.25

    mejor, mejor_k = None, 0
    for _ in range(150):
        i, j = RNG.choice(Np, 2, replace=False)
        idx = [i, j, i + Np, j + Np]
        T, *_ = np.linalg.lstsq(M[idx], y[idx], rcond=None)
        if not plausible(T):
            continue
        k = (error_px(T) < 1.0).sum()
        if k > mejor_k:
            mejor_k, mejor = k, T
    if mejor is None or mejor_k < 12:
        return None, 0
    inl = error_px(mejor) < 1.0
    idx = np.concatenate([np.nonzero(inl)[0], np.nonzero(inl)[0] + Np])
    T, *_ = np.linalg.lstsq(M[idx], y[idx], rcond=None)
    if not plausible(T):
        T = mejor
    return T, int((error_px(T) < 1.0).sum())


def klt(prev, g, p0):
    p1, s1, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None, winSize=(31, 31), maxLevel=4)
    pb, s2, _ = cv2.calcOpticalFlowPyrLK(g, prev, p1, None, winSize=(31, 31), maxLevel=4)
    ok = (s1.ravel() > 0) & (s2.ravel() > 0) & (np.linalg.norm((p0 - pb).reshape(-1, 2), axis=1) < 0.7)
    return p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]


def main():
    modelo = YOLO(os.path.join(AQUI, '..', 'yolo11n.pt'))
    cap = cv2.VideoCapture(V.VIDEO)
    ok, f = cap.read()
    und_prev = V.deshacer(f)
    prev = cv2.cvtColor(und_prev, cv2.COLOR_BGR2GRAY)
    autos_prev, cajas_prev = mascara_vehiculos(modelo, und_prev)
    filas, dibujo = [], []
    Rprev = np.eye(3)
    k = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        k += 1
        und = V.deshacer(f)
        g = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
        # rotacion: puntos de todo el entorno salvo vehiculos
        m_gen = cv2.bitwise_and(V.mascara, cv2.bitwise_not(autos_prev))
        p0 = cv2.goodFeaturesToTrack(prev, 2500, 0.005, 6, mask=m_gen)
        a, b = klt(prev, g, p0)
        R, valido, inl = Rprev, False, np.zeros(len(a), bool)
        if len(a) > 60:
            E, msk = cv2.findEssentialMat(b, a, K, cv2.RANSAC, 0.999, 0.6)
            if E is not None and E.shape == (3, 3):
                _, Rc, t, _ = cv2.recoverPose(E, b, a, K, mask=msk.copy())
                inl = msk.ravel() > 0
                if np.degrees(np.linalg.norm(cv2.Rodrigues(Rc)[0])) < 2.5:
                    R, valido = Rc, True
        # traslacion metrica: pavimento sin vehiculos ni sus sombras
        m_pav = cv2.bitwise_and(MASCARA_PAV, cv2.bitwise_not(autos_prev))
        q0 = cv2.goodFeaturesToTrack(prev, 900, 0.001, 4, mask=m_pav)
        T, n_pav = None, 0
        if q0 is not None and len(q0) > 15:
            qa, qb = klt(prev, g, q0)
            if len(qa) > 15:
                T, n_pav = traslacion_ransac(R, qa, qb)
        filas.append({'frame': k, 'rv': cv2.Rodrigues(R)[0].ravel().tolist(), 'rot_ok': valido,
                      'T': [float(x) for x in T] if T is not None else None, 'n_pav': n_pav,
                      'n': len(a), 'inl': int(inl.sum())})
        sel = RNG.permutation(len(a))[:600]
        dibujo.append({'a': a[sel].astype(np.float16), 'b': b[sel].astype(np.float16), 'inl': inl[sel],
                       'cajas': cajas_prev})
        autos_prev, cajas_prev = mascara_vehiculos(modelo, und)
        prev, Rprev = g, R
        if k % 100 == 0:
            vv = [np.linalg.norm(x['T']) * 90 for x in filas[-100:] if x['T'] is not None]
            print(k, 'escala en %d/100 cuadros, v mediana %.0f km/h' % (len(vv), np.median(vv) if vv else -1), flush=True)
    # ---- suavizado e integracion
    Tc = np.array([x['T'] if x['T'] is not None else [np.nan] * 3 for x in filas], float)
    val = np.all(np.isfinite(Tc), axis=1)
    idx = np.arange(len(Tc))
    for j in range(3):
        Tc[~val, j] = np.interp(idx[~val], idx[val], Tc[val, j])
    v_cruda = np.linalg.norm(Tc, axis=1)
    rapidez = median_filter(v_cruda, size=25, mode='nearest')
    direc = median_filter(Tc / v_cruda[:, None], size=(9, 1), mode='nearest')
    direc /= np.linalg.norm(direc, axis=1)[:, None]
    TT = direc * rapidez[:, None]
    RV = median_filter(np.array([x['rv'] for x in filas]), size=(7, 1), mode='nearest')
    Rw, C = np.eye(3), np.zeros(3)
    tray = [(0.0, 0.0, 0.0)]
    for i in range(len(filas)):
        C = C + Rw @ TT[i]
        Rw = Rw @ cv2.Rodrigues(RV[i])[0]
        fwd = Rw @ ADELANTE
        tray.append((float(C @ LATERAL), float(C @ ADELANTE), float(np.degrees(np.arctan2(fwd @ LATERAL, fwd @ ADELANTE)))))
    vel = rapidez * V.FPS * 3.6
    json.dump({'h': h, 'n': n.tolist(), 'tray': tray, 'vel_kmh': vel.tolist(), 'vel_cruda_kmh': (v_cruda * 90).tolist(),
               'escala_propia': val.tolist(), 'n_pav': [x['n_pav'] for x in filas], 'inl': [x['inl'] for x in filas],
               'n_track': [x['n'] for x in filas], 'K': K.tolist()},
              open(os.path.join(AQUI, 'resultado.json'), 'w'))
    pickle.dump(dibujo, open(os.path.join(AQUI, 'dibujo.pkl'), 'wb'))
    tray = np.array(tray)
    dist = np.sum(np.linalg.norm(np.diff(tray[:, :2], axis=0), axis=1))
    print('h=%.3f m | escala propia en %d de %d cuadros' % (h, val.sum(), len(val)))
    print('distancia %.0f m en %.1f s -> velocidad media %.1f km/h' % (dist, len(filas) / V.FPS, dist / (len(filas) / V.FPS) * 3.6))
    print('velocidad suavizada: mediana %.1f, p5 %.1f, p95 %.1f km/h' % (np.median(vel), np.percentile(vel, 5), np.percentile(vel, 95)))
    for s0 in range(0, len(filas), 125):
        print('t=%4.1f s  v=%.0f km/h  rumbo=%.1f  escala propia %d/125' % (s0 / 25, np.median(vel[s0:s0 + 125]),
              tray[min(s0 + 125, len(tray) - 1), 2], val[s0:s0 + 125].sum()))


if __name__ == '__main__':
    main()
