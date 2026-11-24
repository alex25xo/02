"""Odometria visual monocular sobre un video real de carretera (dashcam).

1. Calibracion de la camara (calib_udacity.json, hecha con sus 20 fotos de tablero 9x6).
2. Seguimiento de puntos con KLT, matriz esencial con RANSAC y recoverPose -> R, t (escala 1).
3. Escala metrica: se triangulan puntos del pavimento, se ajusta el plano de la carretera y se usa la
   altura de la camara, que a su vez se obtiene del ancho del carril (3,7 m en autopistas de EE. UU.).
"""
import json
import csv
import sys
import numpy as np
import cv2
import os
AQUI = os.path.dirname(os.path.abspath(__file__))

VIDEO = os.path.join(AQUI, 'datos', 'project_video.mp4')
ANCHO_CARRIL = 3.7     # m
cal = json.load(open(os.path.join(AQUI, 'calib_udacity.json')))
K0, D0 = np.array(cal['K']), np.array(cal['D'])
W, H = 1280, 720
K, _ = cv2.getOptimalNewCameraMatrix(K0, D0, (W, H), 0)
m1, m2 = cv2.initUndistortRectifyMap(K0, D0, None, K, (W, H), cv2.CV_16SC2)
Kinv = np.linalg.inv(K)
FPS = 25.0

mascara = np.full((H, W), 255, np.uint8)
mascara[int(0.86 * H):] = 0          # capo del auto


def deshacer(f):
    return cv2.remap(f, m1, m2, cv2.INTER_LINEAR)


def plano_ransac(X, iters=200, umbral=0.02):
    """Ajusta n.X = d con RANSAC (umbral relativo a la distancia mediana)."""
    if len(X) < 12:
        return None
    escala = np.median(np.linalg.norm(X, axis=1))
    mejor, mejor_n = None, 0
    rng = np.random.default_rng(0)
    for _ in range(iters):
        idx = rng.choice(len(X), 3, replace=False)
        n = np.cross(X[idx[1]] - X[idx[0]], X[idx[2]] - X[idx[0]])
        L = np.linalg.norm(n)
        if L < 1e-9:
            continue
        n /= L
        d = n @ X[idx[0]]
        r = np.abs(X @ n - d)
        k = (r < umbral * escala).sum()
        if k > mejor_n:
            mejor_n, mejor = k, (n, d, r < umbral * escala)
    if mejor is None or mejor_n < 10:
        return None
    n, d, inl = mejor
    P = X[inl]                                  # refinamiento por minimos cuadrados
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c)
    n = vt[2]
    d = n @ c
    if d < 0:
        n, d = -n, -d
    if n[1] < 0.6:                              # la carretera debe estar "abajo" (eje y de la camara)
        return None
    return n, d, int(inl.sum())


def ancho_carril_unidades(img, n):
    """Ancho del carril propio en 'alturas de camara' usando el plano (n, d=1) y los colores de las lineas."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    amarillo = cv2.inRange(hsv, (15, 90, 120), (35, 255, 255))
    blanco = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))
    roi = np.zeros((H, W), np.uint8)
    roi[int(0.62 * H):int(0.84 * H), int(0.08 * W):int(0.95 * W)] = 255
    adelante = np.array([0, 0, 1.0]) - n * n[2]
    adelante /= np.linalg.norm(adelante)
    lateral = np.cross(n, adelante)
    res = {}
    for nombre, m in (('amarillo', amarillo), ('blanco', blanco)):
        ys, xs = np.nonzero(m & roi)
        if len(xs) < 80:
            return None
        rays = (Kinv @ np.vstack([xs, ys, np.ones_like(xs)])).T
        X = rays / (rays @ n)[:, None]             # interseccion con el plano a distancia 1
        res[nombre] = (X @ lateral, X @ adelante)
    lat_a, fw_a = res['amarillo']
    lat_b, fw_b = res['blanco']
    la = np.median(lat_a)
    # linea blanca del carril propio: el grupo de blancos mas cercano a la derecha del amarillo
    cand = lat_b[(lat_b - la) * np.sign(np.median(lat_b) - la) > 1.2]
    if len(cand) < 40:
        return None
    hist, bordes = np.histogram(np.abs(cand - la), bins=np.arange(1.2, 6.0, 0.1))
    k = np.argmax(hist)
    sel = np.abs(np.abs(cand - la) - (bordes[k] + 0.05)) < 0.35
    if sel.sum() < 30:
        return None
    return float(np.median(np.abs(cand[sel] - la)))


def main():
    cap = cv2.VideoCapture(VIDEO)
    ok, f = cap.read()
    prev = cv2.cvtColor(deshacer(f), cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(prev, 3000, 0.01, 7, mask=mascara)
    filas = []
    k = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        k += 1
        und = deshacer(f)
        g = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None, winSize=(21, 21), maxLevel=3)
        st = st.ravel().astype(bool)
        a, b = p0[st].reshape(-1, 2), p1[st].reshape(-1, 2)
        dentro = (b[:, 0] > 0) & (b[:, 0] < W) & (b[:, 1] > 0) & (b[:, 1] < 0.86 * H)
        a, b = a[dentro], b[dentro]
        fila = {'frame': k, 'n': len(a)}
        if len(a) > 50:
            E, inl = cv2.findEssentialMat(b, a, K, cv2.RANSAC, 0.999, 0.6)
            if E is not None and E.shape == (3, 3):
                _, R, t, inl2 = cv2.recoverPose(E, b, a, K, mask=inl.copy())
                m = inl2.ravel() > 0
                fila.update({'inl': int(m.sum()), 'R': R.ravel().tolist(), 't': t.ravel().tolist()})
                # puntos del pavimento: parte baja y central de la imagen
                road = m & (a[:, 1] > 0.6 * H) & (np.abs(a[:, 0] - W / 2) < 0.35 * W)
                if road.sum() > 15:
                    # X_prev = R X_curr + t  ->  P_curr = K [R^T | -R^T t]
                    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
                    P1 = K @ np.hstack([R.T, -R.T @ t])
                    Xh = cv2.triangulatePoints(P0, P1, a[road].T, b[road].T)
                    X = (Xh[:3] / Xh[3]).T
                    X = X[(X[:, 2] > 0) & (X[:, 2] < 200)]
                    pl = plano_ransac(X)
                    if pl is not None:
                        fila.update({'n_plano': [float(v) for v in pl[0]], 'd': float(pl[1]), 'inl_plano': int(pl[2])})
                        if k % 5 == 0:
                            w = ancho_carril_unidades(und, pl[0])
                            if w is not None:
                                fila['ancho_u'] = w
        filas.append(fila)
        if len(b) < 1500 or k % 5 == 0:
            nuevos = cv2.goodFeaturesToTrack(g, 3000, 0.01, 7, mask=mascara)
            p0 = nuevos if nuevos is not None else b.reshape(-1, 1, 2)
        else:
            p0 = b.reshape(-1, 1, 2)
        prev = g
        if k % 100 == 0:
            print(k, fila.get('inl'), fila.get('d'), fila.get('ancho_u'), file=sys.stderr)
    json.dump({'K': K.tolist(), 'filas': filas}, open(os.path.join(AQUI, 'pasada1.json'), 'w'))
    anchos = [f['ancho_u'] for f in filas if 'ancho_u' in f]
    print('frames', k, 'anchos medidos', len(anchos), 'mediana', np.median(anchos), 'p25-p75', np.percentile(anchos, [25, 75]))


if __name__ == '__main__':
    main()
