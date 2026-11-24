"""Etapa 02 - item 4: tracking con la camara EN MOVIMIENTO (video real de autopista).

Sin odometria visual solo se mide el movimiento RELATIVO a nuestro auto (un auto que va a nuestra misma
velocidad parece quieto). Con la odometria visual conocemos el movimiento de la camara y cada vehiculo se
ubica en coordenadas del MUNDO (metros sobre la carretera): asi se obtiene su velocidad REAL.

  posicion en la camara:   X_c = h * r / (n . r)      (punto de apoyo del vehiculo sobre el plano de la pista)
  posicion en el mundo:    X_w = C_k + R(rumbo_k) X_c  (pose de la camara estimada por la VO)
  velocidad real = derivada de X_w ; velocidad relativa = derivada de X_c
"""
import json
import csv
import numpy as np
import cv2
from ultralytics import YOLO

import vo_carretera as V
import os
AQUI = os.path.dirname(os.path.abspath(__file__))

R = json.load(open(os.path.join(AQUI, 'resultado.json')))
h = R['h']
n = np.array(R['n'])
K = np.array(R['K'])
Kinv = np.linalg.inv(K)
ADELANTE = np.array([0, 0, 1.0]) - n * n[2]
ADELANTE /= np.linalg.norm(ADELANTE)
LATERAL = np.cross(n, ADELANTE)
TRAY = np.array(R['tray'])          # por cuadro: (lateral, adelante, rumbo en grados) de la camara
VEL_EGO = np.array(R['vel_kmh'])
FPS = V.FPS


def punto_en_pista(u, v):
    r = Kinv @ np.array([u, v, 1.0])
    den = n @ r
    if den <= 1e-6:
        return None
    X = r * (h / den)
    return float(X @ LATERAL), float(X @ ADELANTE)


def a_mundo(k, lat_c, fwd_c):
    cl, cf, rumbo = TRAY[min(k, len(TRAY) - 1)]
    s, c = np.sin(np.radians(rumbo)), np.cos(np.radians(rumbo))
    return cl + fwd_c * s + lat_c * c, cf + fwd_c * c - lat_c * s


def velocidad(ts, xs, ys, ventana=1.0):
    """Velocidad (m/s) por regresion lineal en la ultima 'ventana' de segundos."""
    ts, xs, ys = np.array(ts), np.array(xs), np.array(ys)
    sel = ts >= ts[-1] - ventana
    if sel.sum() < 6 or ts[sel][-1] - ts[sel][0] < 0.5:
        return None
    vx = np.polyfit(ts[sel], xs[sel], 1)[0]
    vy = np.polyfit(ts[sel], ys[sel], 1)[0]
    return float(np.hypot(vx, vy))


def main(guardar_video=False):
    modelo = YOLO(os.path.join(AQUI, '..', 'yolo11n.pt'))
    cap = cv2.VideoCapture(V.VIDEO)
    filas = []
    hist = {}
    salida = None
    k = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        und = V.deshacer(f)
        res = modelo.track(und, persist=True, tracker='botsort.yaml', classes=[2, 5, 7], conf=0.3, imgsz=960,
                           verbose=False)[0]
        t = k / FPS
        vis = und.copy()
        if res.boxes.id is not None:
            for (x1, y1, x2, y2), tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.cpu().numpy().astype(int)):
                u, v = (x1 + x2) / 2, y2 - 0.03 * (y2 - y1)      # punto de apoyo sobre la pista
                pc = punto_en_pista(u, v)
                if pc is None or not (6 < pc[1] < 70) or pc[0] < -6:   # solo nuestra calzada (tras la barrera no se ven las ruedas)
                    continue
                pw = a_mundo(k, *pc)
                H_ = hist.setdefault(tid, {'t': [], 'lc': [], 'fc': [], 'lw': [], 'fw': []})
                H_['t'].append(t); H_['lc'].append(pc[0]); H_['fc'].append(pc[1]); H_['lw'].append(pw[0]); H_['fw'].append(pw[1])
                v_rel = velocidad(H_['t'], H_['lc'], H_['fc'])
                v_abs = velocidad(H_['t'], H_['lw'], H_['fw'])
                filas.append({'frame': k, 't': round(t, 3), 'id': tid, 'x1': round(float(x1), 1), 'y1': round(float(y1), 1), 'x2': round(float(x2), 1), 'y2': round(float(y2), 1), 'u': round(u, 1), 'v': round(v, 1),
                              'lat_cam': round(pc[0], 2), 'adel_cam': round(pc[1], 2),
                              'lat_mundo': round(pw[0], 2), 'adel_mundo': round(pw[1], 2),
                              'v_rel_kmh': round(v_rel * 3.6, 1) if v_rel is not None else '',
                              'v_real_kmh': round(v_abs * 3.6, 1) if v_abs is not None else ''})
                if guardar_video:
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 180, 40), 2, cv2.LINE_AA)
                    txt1 = 'ID %d  a %.0f m' % (tid, pc[1])
                    txt2 = ('relativa %+.0f km/h' % (v_rel * 3.6 * np.sign(np.diff(H_['fc'][-8:]).sum() or 1))) if v_rel is not None else 'relativa ...'
                    txt3 = ('REAL %.0f km/h' % (v_abs * 3.6)) if v_abs is not None else 'real ...'
                    y0 = int(y1) - 56
                    cv2.rectangle(vis, (int(x1), y0), (int(x1) + 190, int(y1)), (40, 30, 15), -1)
                    cv2.putText(vis, txt1, (int(x1) + 6, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(vis, txt2, (int(x1) + 6, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 200, 255), 1, cv2.LINE_AA)
                    cv2.putText(vis, txt3, (int(x1) + 6, y0 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 120), 2, cv2.LINE_AA)
        if guardar_video:
            cv2.rectangle(vis, (14, 14), (520, 70), (40, 30, 15), -1)
            cv2.putText(vis, 'Nuestro auto (odometria visual): %.0f km/h' % VEL_EGO[min(k, len(VEL_EGO) - 1)], (26, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, 't = %.1f s' % t, (26, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            if salida is None:
                salida = cv2.VideoWriter(os.path.join(AQUI, 'tracking_absoluto_tmp.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                         FPS, (vis.shape[1], vis.shape[0]))
            salida.write(vis)
        k += 1
        if k % 200 == 0:
            print(k, 'ids', len(hist), flush=True)
    if salida is not None:
        salida.release()
    with open(os.path.join(AQUI, 'trayectorias_autopista.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader(); w.writerows(filas)
    # resumen por vehiculo
    resumen = []
    for tid, H_ in hist.items():
        if len(H_['t']) < 40:
            continue
        T = np.array(H_['t'])
        dur = T[-1] - T[0]
        # velocidad media robusta: desplazamiento total / tiempo (mundo y camara)
        vr = np.hypot(H_['lc'][-1] - H_['lc'][0], H_['fc'][-1] - H_['fc'][0]) / dur * 3.6
        signo = np.sign(H_['fc'][-1] - H_['fc'][0])
        va = np.hypot(H_['lw'][-1] - H_['lw'][0], H_['fw'][-1] - H_['fw'][0]) / dur * 3.6
        ego = float(np.mean(VEL_EGO[int(T[0] * FPS):int(T[-1] * FPS) + 1]))
        resumen.append({'id': int(tid), 'desde_s': round(T[0], 1), 'hasta_s': round(T[-1], 1), 'muestras': len(T),
                        'distancia_media_m': round(float(np.mean(H_['fc'])), 1),
                        'v_relativa_kmh': round(float(vr * signo), 1), 'v_real_kmh': round(float(va), 1),
                        'v_nuestro_auto_kmh': round(ego, 1)})
    resumen.sort(key=lambda r: r['desde_s'])
    json.dump(resumen, open(os.path.join(AQUI, 'resumen_vehiculos.json'), 'w'), indent=1)
    for r in resumen:
        print(r)


if __name__ == '__main__':
    main()
