"""Video de presentacion: tracking con camara en movimiento (autopista real) + vista superior en el mundo."""
import json
import subprocess
import numpy as np
import pandas as pd
import cv2
import vo_carretera as V
import os
AQUI = os.path.dirname(os.path.abspath(__file__))

R = json.load(open(os.path.join(AQUI, 'resultado.json')))
TRAY = np.array(R['tray'])
VEL = np.array(R['vel_kmh'])
df = pd.read_csv(os.path.join(AQUI, 'trayectorias_autopista.csv'))
df['v_real_kmh'] = pd.to_numeric(df['v_real_kmh'], errors='coerce')
df['v_rel_kmh'] = pd.to_numeric(df['v_rel_kmh'], errors='coerce')
# solo vehiculos seguidos al menos 1.5 s
largos = df.groupby('id').size()
df = df[df['id'].isin(largos[largos >= 38].index)]
por_frame = {k: g for k, g in df.groupby('frame')}
FPS = V.FPS
ESC = 0.75
MW, MH = 330, 540
PXM = 6.6                     # px por metro en el mapa
BASE_Y = MH - 70              # posicion de nuestro auto en el mapa

PALETA = [(235, 99, 37), (22, 115, 249), (74, 163, 22), (237, 58, 124), (164, 165, 14), (4, 138, 202)]


ORDEN = {}


def col(tid):
    if int(tid) not in ORDEN:
        ORDEN[int(tid)] = len(ORDEN)
    return PALETA[ORDEN[int(tid)] % len(PALETA)]


def ego(k, lat, fwd):
    cl, cf, rumbo = TRAY[min(k, len(TRAY) - 1)]
    d = np.array([lat - cl, fwd - cf])
    s, c = np.sin(np.radians(rumbo)), np.cos(np.radians(rumbo))
    # coordenadas del mundo -> marco de nuestro auto (adelante = arriba)
    adel = d[0] * s + d[1] * c
    lat_e = d[0] * c - d[1] * s
    return lat_e, adel


PXM_LAT = 15.0               # el ancho se exagera para que se vean los carriles


def a_mapa(lat_e, adel):
    return int(100 + lat_e * PXM_LAT), int(BASE_Y - adel * PXM)


def texto(img, t, org, esc=0.5, color=(255, 255, 255), grosor=1):
    cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, esc, color, grosor, cv2.LINE_AA)


def mapa(k):
    m = np.full((MH, MW, 3), (38, 32, 26), np.uint8)
    # asfalto
    x0, x1 = a_mapa(-3.4, 0)[0], a_mapa(11.0, 0)[0]
    cv2.rectangle(m, (x0, 0), (x1, MH), (70, 66, 62), -1)
    cv2.line(m, (x0 - 3, 0), (x0 - 3, MH), (170, 170, 170), 4)          # barrera
    # marcas del suelo fijas al MUNDO (se desplazan hacia atras porque avanzamos)
    cf = TRAY[min(k, len(TRAY) - 1)][1]
    for lat in (1.83, 5.49, 9.15):
        x = a_mapa(lat, 0)[0]
        inicio = -((cf % 12.2))
        for j in range(-3, 14):
            a = inicio + j * 12.2
            y0, y1 = int(BASE_Y - a * PXM), int(BASE_Y - (a + 3.05) * PXM)
            cv2.line(m, (x, y0), (x, y1), (230, 230, 230), 2)
    xa = a_mapa(-1.83, 0)[0]
    cv2.line(m, (xa, 0), (xa, MH), (40, 200, 240), 2)                  # linea amarilla
    # vehiculos con su estela en el MUNDO (ultimos 2 s)
    act = por_frame.get(k)
    if act is not None:
        for _, r in act.iterrows():
            tid = r['id']
            h = df[(df['id'] == tid) & (df['frame'] <= k) & (df['frame'] > k - 2 * FPS)]
            pts = [a_mapa(*ego(k, a, b)) for a, b in zip(h['lat_mundo'], h['adel_mundo'])]
            if len(pts) > 1:
                cv2.polylines(m, [np.int32(pts).reshape(-1, 1, 2)], False, col(tid), 2, cv2.LINE_AA)
            p = a_mapa(*ego(k, r['lat_mundo'], r['adel_mundo']))
            cv2.rectangle(m, (p[0] - 6, p[1] - 14), (p[0] + 6, p[1] + 14), col(tid), -1)
            cv2.rectangle(m, (p[0] - 6, p[1] - 14), (p[0] + 6, p[1] + 14), (255, 255, 255), 1)
            v = r['v_real_kmh']
            if not np.isnan(v):
                texto(m, f"{v:.0f} km/h", (p[0] + 10, p[1] + 5), 0.45, (255, 255, 255), 1)
    # nuestra trayectoria (ultimos 2 s) y nuestro auto
    hs = [a_mapa(*ego(k, TRAY[j][0], TRAY[j][1])) for j in range(max(0, k - 2 * int(FPS)), k + 1)]
    if len(hs) > 1:
        cv2.polylines(m, [np.int32(hs).reshape(-1, 1, 2)], False, (255, 255, 255), 2, cv2.LINE_AA)
    p = a_mapa(0, 0)
    tri = np.int32([[p[0], p[1] - 16], [p[0] - 8, p[1] + 12], [p[0] + 8, p[1] + 12]])
    cv2.fillPoly(m, [tri], (60, 220, 255))
    cv2.rectangle(m, (p[0] - 50, p[1] + 18), (p[0] + 50, p[1] + 38), (25, 20, 15), -1)
    texto(m, "nuestro auto", (p[0] - 44, p[1] + 33), 0.45, (60, 220, 255), 1)
    cv2.rectangle(m, (0, MH - 22), (200, MH), (25, 20, 15), -1)
    texto(m, "ancho de carriles exagerado", (8, MH - 8), 0.38, (150, 150, 150), 1)
    cv2.rectangle(m, (0, 0), (MW, 54), (25, 20, 15), -1)
    texto(m, "VISTA SUPERIOR EN EL MUNDO", (12, 22), 0.55, (255, 255, 255), 1)
    texto(m, "(posiciones de la odometria visual)", (12, 43), 0.45, (190, 190, 190), 1)
    for d in (10, 20, 30, 40, 50, 60):
        y = int(BASE_Y - d * PXM)
        if y > 60:
            texto(m, f"{d} m", (MW - 40, y + 4), 0.38, (170, 170, 170), 1)
    return m


def main():
    cap = cv2.VideoCapture(V.VIDEO)
    tmp = os.path.join(AQUI, 'tmp_tracking.mp4')
    out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (960 + MW, 540))
    k = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        und = cv2.resize(V.deshacer(f), (960, 540))
        act = por_frame.get(k)
        if act is not None:
            for _, r in act.iterrows():
                x1, y1, x2, y2 = (np.array([r['x1'], r['y1'], r['x2'], r['y2']]) * ESC).astype(int)
                c = col(r['id'])
                cv2.rectangle(und, (x1, y1), (x2, y2), c, 2, cv2.LINE_AA)
                y0 = max(0, y1 - 58)
                cv2.rectangle(und, (x1, y0), (x1 + 178, y0 + 54), (25, 20, 15), -1)
                cv2.rectangle(und, (x1, y0), (x1 + 4, y0 + 54), c, -1)
                texto(und, f"ID {int(r['id'])}  a {r['adel_cam']:.0f} m", (x1 + 10, y0 + 16), 0.45)
                vr, va = r['v_rel_kmh'], r['v_real_kmh']
                texto(und, "sin VO: " + (f"{vr:.0f} km/h" if not np.isnan(vr) else "..."), (x1 + 10, y0 + 33), 0.45,
                      (120, 190, 255))
                texto(und, "con VO: " + (f"{va:.0f} km/h" if not np.isnan(va) else "..."), (x1 + 10, y0 + 50), 0.5,
                      (120, 255, 140), 1)
        cv2.rectangle(und, (12, 12), (430, 66), (25, 20, 15), -1)
        texto(und, f"Nuestro auto (odometria visual): {VEL[min(k, len(VEL) - 1)]:.0f} km/h", (22, 36), 0.6,
              (255, 255, 255), 1)
        texto(und, "sin VO = velocidad relativa | con VO = velocidad real", (22, 57), 0.45, (200, 200, 200), 1)
        out.write(np.hstack([und, mapa(k)]))
        k += 1
        if k % 300 == 0:
            print(k, flush=True)
    out.release()
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp, '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '26',
                    '-pix_fmt', 'yuv420p', os.path.join(AQUI, 'tracking_camara_movil_autopista.mp4')], check=True)
    print('ok')


if __name__ == '__main__':
    main()
