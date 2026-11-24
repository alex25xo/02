"""
comun.py - Piezas compartidas por las 5 partes de la Etapa 02
=============================================================
  * abrir_fuente ......... webcam (0, 1, ...) o archivo de video
  * TrackerYOLO .......... YOLO + ByteTrack / BoT-SORT  (tracking por deteccion)
  * LineaConteo .......... linea virtual: cuenta cruces y su sentido
  * RegistroTrayectorias . CSV con (ID, frame, centro X, Y, ...) para analizar despues
  * Calibracion .......... pixeles -> metros (factor px/m, homografia o altura de persona)
  * Velocimetro .......... velocidad por regresion lineal en una ventana de tiempo
  * MapaCalor / Zonas .... donde estuvieron los objetos
  * OdometriaVisual2D .... movimiento de la camara entre frames (parte 5)
  * dibujo ............... colores por ID, etiquetas, paneles, estelas
"""
import csv
import json
import os
import time
from collections import defaultdict, deque

import cv2
import numpy as np

# ----------------------------------------------------------------------------- fuente de video

def abrir_fuente(fuente, ancho=None):
    """'0', '1'... -> webcam ; cualquier otra cosa -> archivo. Devuelve (cap, fps, es_webcam)."""
    es_webcam = str(fuente).isdigit()
    if es_webcam:
        api = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY      # en Windows abre mas rapido
        cap = cv2.VideoCapture(int(fuente), api)
        if ancho:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
    else:
        if not os.path.exists(fuente):
            raise SystemExit(f"No existe el video: {fuente}")
        cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la fuente: {fuente}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or fps > 240:
        fps = 30.0
    return cap, fps, es_webcam


class Reloj:
    """Tiempo de cada frame: en un video = frame/fps ; en la webcam = tiempo real."""

    def __init__(self, fps, es_webcam):
        self.fps, self.es_webcam, self.t0 = fps, es_webcam, time.time()

    def t(self, frame_idx):
        return time.time() - self.t0 if self.es_webcam else frame_idx / self.fps


# ----------------------------------------------------------------------------- tracking 2D

class TrackerYOLO:
    """Tracking por deteccion: YOLO detecta en cada frame y ByteTrack/BoT-SORT asigna IDs.

    ByteTrack: Kalman + IoU + Hungaro, en 2 rondas (primero cajas de confianza alta, luego las bajas).
    BoT-SORT : lo mismo + compensacion del movimiento de camara (GMC) + (opcional) apariencia ReID.
    """

    def __init__(self, modelo="yolo11n.pt", tracker="bytetrack", clases=None, conf=0.3, imgsz=640):
        from ultralytics import YOLO
        self.modelo = YOLO(modelo)
        self.nombres = self.modelo.names
        self.tracker = tracker if tracker.endswith(".yaml") else f"{tracker}.yaml"
        self.conf, self.imgsz = conf, imgsz
        self.clases = None
        if clases:
            inv = {v: k for k, v in self.nombres.items()}
            self.clases = [inv[c] if c in inv else int(c) for c in clases]

    def procesar(self, frame):
        """Devuelve una lista de dicts: id, clase, conf, caja (x1,y1,x2,y2), centro (cx,cy), pie (cx,y2)."""
        r = self.modelo.track(frame, persist=True, tracker=self.tracker, classes=self.clases,
                              conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        objetos = []
        if r.boxes is None or r.boxes.id is None:
            return objetos
        cajas = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        cls = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), tid, c, cf in zip(cajas, ids, cls, confs):
            objetos.append({"id": int(tid), "clase": self.nombres[int(c)], "conf": float(cf),
                            "caja": (float(x1), float(y1), float(x2), float(y2)),
                            "centro": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                            "pie": ((x1 + x2) / 2.0, float(y2))})
        return objetos


# ----------------------------------------------------------------------------- linea virtual

def parse_linea(texto, ancho, alto):
    """'x1,y1,x2,y2' en fracciones (0-1) o en pixeles."""
    v = [float(x) for x in texto.split(",")]
    if all(0 <= x <= 1 for x in v):
        v = [v[0] * ancho, v[1] * alto, v[2] * ancho, v[3] * alto]
    return (v[0], v[1]), (v[2], v[3])


def _lado(p1, p2, p):
    """Signo del producto cruz: de que lado de la recta p1->p2 esta p."""
    return np.sign((p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0]))


def segmentos_se_cruzan(a, b, c, d):
    """True si el segmento a-b (movimiento del objeto) corta al segmento c-d (linea virtual)."""
    return (_lado(c, d, a) * _lado(c, d, b) < 0) and (_lado(a, b, c) * _lado(a, b, d) < 0)


class LineaConteo:
    """Cuenta un objeto cuando su centro pasa de un lado al otro de la linea (una sola vez por ID)."""

    def __init__(self, p1, p2):
        self.p1, self.p2 = tuple(map(float, p1)), tuple(map(float, p2))
        self.ultimo = {}            # id -> ultima posicion
        self.contados = {}          # id -> sentido
        self.por_clase = defaultdict(lambda: {"A->B": 0, "B->A": 0})
        self.eventos = []           # (frame, t, id, clase, sentido)
        self.destello = 0

    def actualizar(self, tid, p, clase="obj", frame=0, t=0.0):
        cruzo = None
        q = self.ultimo.get(tid)
        if q is not None and tid not in self.contados and segmentos_se_cruzan(q, p, self.p1, self.p2):
            sentido = "A->B" if _lado(self.p1, self.p2, p) > 0 else "B->A"
            self.contados[tid] = sentido
            self.por_clase[clase][sentido] += 1
            self.eventos.append((frame, round(t, 3), tid, clase, sentido))
            self.destello = 6
            cruzo = sentido
        self.ultimo[tid] = p
        return cruzo

    @property
    def total(self):
        return len(self.contados)

    def totales(self):
        ab = sum(v["A->B"] for v in self.por_clase.values())
        return ab, self.total - ab

    def dibujar(self, img, p1=None, p2=None):
        p1 = tuple(map(int, p1 or self.p1)); p2 = tuple(map(int, p2 or self.p2))
        color = (0, 255, 255) if self.destello > 0 else (255, 255, 255)
        cv2.line(img, p1, p2, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)
        cv2.putText(img, "A", (p1[0] + 6, p1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.putText(img, "B", (p2[0] - 20, p2[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        self.destello = max(0, self.destello - 1)


class SelectorLinea:
    """Permite redibujar la linea con el mouse (2 clics) durante la ejecucion."""

    def __init__(self, ventana):
        self.puntos, self.activo = [], False
        cv2.setMouseCallback(ventana, self._cb)

    def iniciar(self):
        self.puntos, self.activo = [], True

    def _cb(self, ev, x, y, *_):
        if self.activo and ev == cv2.EVENT_LBUTTONDOWN:
            self.puntos.append((x, y))
            if len(self.puntos) == 2:
                self.activo = False

    def nueva_linea(self):
        if not self.activo and len(self.puntos) == 2:
            p = self.puntos
            self.puntos = []
            return p
        return None


# ----------------------------------------------------------------------------- registro de trayectorias

class RegistroTrayectorias:
    """Guarda una fila por objeto y por frame. Columnas minimas pedidas: id, frame, cx, cy."""

    COLUMNAS = ["frame", "t_s", "id", "clase", "conf", "cx", "cy", "x1", "y1", "x2", "y2"]

    def __init__(self, ruta, extra=()):
        os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
        self.columnas = self.COLUMNAS + list(extra)
        self.f = open(ruta, "w", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=self.columnas, extrasaction="ignore")
        self.w.writeheader()
        self.n = 0

    def guardar(self, frame, t, o, **extra):
        x1, y1, x2, y2 = o["caja"]
        fila = {"frame": frame, "t_s": round(t, 3), "id": o["id"], "clase": o["clase"], "conf": round(o["conf"], 3),
                "cx": round(o["centro"][0], 1), "cy": round(o["centro"][1], 1),
                "x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)}
        fila.update({k: (round(v, 3) if isinstance(v, float) else v) for k, v in extra.items()})
        self.w.writerow(fila)
        self.n += 1

    def cerrar(self):
        self.f.close()


# ----------------------------------------------------------------------------- calibracion px -> m

class Calibracion:
    """Convierte puntos de la imagen a metros sobre el suelo.

    tipo 'px'        : sin calibrar (resultado en px)
    tipo 'ppm'       : un solo factor pixeles/metro (vale solo si la camara mira de frente al plano)
    tipo 'homografia': H 3x3 imagen->suelo con 4 puntos de medidas conocidas (corrige la perspectiva)
    tipo 'persona'   : escala local = 1.70 m / altura de la caja de cada persona (sin calibrar nada)
    """

    ALTURA_PERSONA = 1.70

    def __init__(self, tipo="px", ppm=None, H=None):
        self.tipo, self.ppm, self.H = tipo, ppm, H

    @staticmethod
    def cargar(ruta):
        d = json.load(open(ruta, encoding="utf-8"))
        if d["tipo"] == "ppm":
            return Calibracion("ppm", ppm=float(d["ppm"]))
        H = cv2.getPerspectiveTransform(np.float32(d["puntos_imagen"]), np.float32(d["puntos_mundo_m"]))
        return Calibracion("homografia", H=H)

    @property
    def metrica(self):
        return self.tipo != "px"

    def a_mundo(self, p, caja=None, H_extra=None):
        """p (x,y) en px -> (X,Y) en m (o px si no hay calibracion).
        H_extra: homografia adicional imagen_actual -> imagen_referencia (parte 5, camara movil)."""
        x, y = p
        if H_extra is not None:
            x, y = aplicar_H(H_extra, (x, y))
        if self.tipo == "homografia":
            return aplicar_H(self.H, (x, y))
        if self.tipo == "ppm":
            return x / self.ppm, y / self.ppm
        if self.tipo == "persona" and caja is not None:
            alto = max(caja[3] - caja[1], 1.0)
            s = self.ALTURA_PERSONA / alto            # metros por pixel EN ESA POSICION
            return x * s, y * s
        return x, y


def aplicar_H(H, p):
    v = H @ np.array([p[0], p[1], 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


# ----------------------------------------------------------------------------- velocidad

class Velocimetro:
    """Velocidad = pendiente de la recta posicion(t) en los ultimos 'ventana' segundos (menos ruido que
    restar dos frames consecutivos) + suavizado exponencial."""

    def __init__(self, ventana=1.0, alfa=0.4, min_muestras=5):
        self.ventana, self.alfa, self.min_muestras = ventana, alfa, min_muestras
        self.hist = defaultdict(lambda: deque(maxlen=400))
        self.vel = {}

    def actualizar(self, tid, t, X, Y, escala_local=None):
        h = self.hist[tid]
        h.append((t, X, Y))
        while h and h[-1][0] - h[0][0] > self.ventana:
            h.popleft()
        if len(h) < self.min_muestras or h[-1][0] - h[0][0] < 0.3 * self.ventana:
            return self.vel.get(tid)
        a = np.array(h)
        vx = np.polyfit(a[:, 0], a[:, 1], 1)[0]
        vy = np.polyfit(a[:, 0], a[:, 2], 1)[0]
        v = float(np.hypot(vx, vy))
        self.vel[tid] = v if tid not in self.vel else self.alfa * v + (1 - self.alfa) * self.vel[tid]
        return self.vel[tid]


# ----------------------------------------------------------------------------- mapa de calor y zonas

class MapaCalor:
    def __init__(self, ancho, alto, sigma=12):
        self.acum = np.zeros((alto, ancho), np.float32)
        self.sigma = sigma
        r = int(3 * sigma)
        g = cv2.getGaussianKernel(2 * r + 1, sigma)
        self.kernel, self.r = (g @ g.T).astype(np.float32), r

    def agregar(self, p, peso=1.0):
        x, y = int(round(p[0])), int(round(p[1]))
        h, w = self.acum.shape
        r = self.r
        x0, x1, y0, y1 = max(x - r, 0), min(x + r + 1, w), max(y - r, 0), min(y + r + 1, h)
        if x0 >= x1 or y0 >= y1:
            return
        self.acum[y0:y1, x0:x1] += peso * self.kernel[y0 - (y - r):y1 - (y - r), x0 - (x - r):x1 - (x - r)]

    def colorear(self, fondo, alfa_max=0.7):
        if self.acum.max() <= 0:
            return fondo.copy()
        d = (self.acum / self.acum.max()) ** 0.5
        color = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET)
        a = (np.clip(d * 1.6, 0, 1) * alfa_max)[..., None]
        return (fondo * (1 - a) + color * a).astype(np.uint8)


class Zonas:
    """Rejilla de filas x columnas (o poligonos de un JSON). Por zona: IDs distintos, tiempo y velocidad."""

    def __init__(self, ancho, alto, filas=3, cols=3, ruta_json=None):
        self.poligonos = []
        if ruta_json:
            d = json.load(open(ruta_json, encoding="utf-8"))
            for z in d["zonas"]:
                self.poligonos.append((z["nombre"], np.int32(z["puntos"])))
        else:
            letras = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for i in range(filas):
                for j in range(cols):
                    x0, x1 = int(j * ancho / cols), int((j + 1) * ancho / cols)
                    y0, y1 = int(i * alto / filas), int((i + 1) * alto / filas)
                    self.poligonos.append((f"{letras[i]}{j + 1}", np.int32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])))
        self.ids = defaultdict(set)
        self.muestras = defaultdict(int)
        self.vels = defaultdict(list)

    def zona_de(self, p):
        for nombre, poly in self.poligonos:
            if cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0:
                return nombre
        return None

    def actualizar(self, tid, p, vel=None):
        z = self.zona_de(p)
        if z:
            self.ids[z].add(tid)
            self.muestras[z] += 1
            if vel is not None:
                self.vels[z].append(vel)
        return z

    def tabla(self, fps):
        filas = []
        for nombre, _ in self.poligonos:
            v = self.vels[nombre]
            filas.append({"zona": nombre, "objetos_distintos": len(self.ids[nombre]),
                          "tiempo_total_s": round(self.muestras[nombre] / fps, 1),
                          "vel_media": round(float(np.mean(v)), 2) if v else ""})
        return filas

    def dibujar(self, img, metrica=None):
        maxn = max([len(s) for s in self.ids.values()] + [1])
        capa = img.copy()
        for nombre, poly in self.poligonos:
            n = len(self.ids[nombre])
            col = cv2.applyColorMap(np.uint8([[int(255 * n / maxn)]]), cv2.COLORMAP_INFERNO)[0, 0].tolist()
            cv2.fillPoly(capa, [poly], col)
        out = cv2.addWeighted(capa, 0.28, img, 0.72, 0)
        for nombre, poly in self.poligonos:
            cv2.polylines(out, [poly], True, (255, 255, 255), 1, cv2.LINE_AA)
            x, y = poly[:, 0].min() + 6, poly[:, 1].min() + 20
            texto = f"{nombre}: {len(self.ids[nombre])}"
            if metrica and self.vels[nombre]:
                texto += f" | {np.mean(self.vels[nombre]):.1f} {metrica}"
            etiqueta(out, texto, (x, y), (30, 30, 30))
        return out


# ----------------------------------------------------------------------------- odometria visual 2D (parte 5)

class OdometriaVisual2D:
    """Estima el movimiento de la CAMARA entre frames usando solo el fondo (escena estatica).

    1) Puntos esquina (Shi-Tomasi) fuera de las cajas de los objetos moviles (mascara).
    2) Flujo optico KLT frame k-1 -> k  (+ chequeo ida y vuelta).
    3) RANSAC: homografia H_k (k-1 -> k) que explica a la mayoria (inliers = fondo).
    4) Se acumula G_k = H_k ... H_1 : (frame 0 -> frame k).  inv(G_k) lleva el frame actual
       al sistema de referencia del frame 0, que esta "anclado" al mundo.
    """

    def __init__(self, n_puntos=800, modelo="homografia"):
        self.n_puntos, self.modelo = n_puntos, modelo
        self.prev = None
        self.G = np.eye(3)
        self.info = {"puntos": 0, "inliers": 0, "flujo_px": 0.0}
        self.tray = [(0.0, 0.0)]

    def _mascara(self, gris, cajas, margen=10):
        m = np.full(gris.shape, 255, np.uint8)
        for x1, y1, x2, y2 in cajas:
            cv2.rectangle(m, (int(x1) - margen, int(y1) - margen), (int(x2) + margen, int(y2) + margen), 0, -1)
        return m

    def procesar(self, frame, cajas=()):
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        masc = self._mascara(gris, cajas)
        if self.prev is None:
            self.prev, self.prev_masc = gris, masc
            return self.G
        p0 = cv2.goodFeaturesToTrack(self.prev, self.n_puntos, 0.01, 8, mask=self.prev_masc)
        H = np.eye(3)
        if p0 is not None and len(p0) >= 12:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gris, p0, None, winSize=(21, 21), maxLevel=3)
            pr, st2, _ = cv2.calcOpticalFlowPyrLK(gris, self.prev, p1, None, winSize=(21, 21), maxLevel=3)
            ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(p0 - pr, axis=2)[:, 0] < 1.0)
            a, b = p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2)
            if len(a) >= 12:
                if self.modelo == "homografia":
                    Hn, inl = cv2.findHomography(a, b, cv2.RANSAC, 2.0)
                else:
                    A, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                    Hn = None if A is None else np.vstack([A, [0, 0, 1]])
                if Hn is not None:
                    H = Hn
                    inl = inl.ravel().astype(bool)
                    self.info = {"puntos": int(len(a)), "inliers": int(inl.sum()),
                                 "flujo_px": float(np.median(np.linalg.norm(b - a, axis=1))),
                                 "a": a, "b": b, "inl": inl}
        self.G = H @ self.G
        c = aplicar_H(np.linalg.inv(self.G), (gris.shape[1] / 2, gris.shape[0] / 2))
        self.tray.append(c)
        self.prev, self.prev_masc = gris, masc
        return self.G

    @property
    def a_referencia(self):
        """Homografia frame actual -> frame 0 (mundo)."""
        return np.linalg.inv(self.G)


# ----------------------------------------------------------------------------- dibujo

def color_id(tid):
    """Color estable por ID (proporcion aurea en el circulo de tono)."""
    h = int((tid * 0.61803398875 % 1.0) * 179)
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, 210, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


def etiqueta(img, texto, org, fondo, color=(255, 255, 255), escala=0.5, grosor=1):
    (w, h), b = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x - 2, y - h - 4), (x + w + 2, y + b), fondo, -1)
    cv2.putText(img, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)


def dibujar_objeto(img, o, texto=None, color=None):
    color = color or color_id(o["id"])
    x1, y1, x2, y2 = map(int, o["caja"])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    etiqueta(img, texto or f"#{o['id']} {o['clase']}", (x1, y1 - 3), color, (0, 0, 0))


def panel(img, lineas, org=(10, 10), escala=0.55, alfa=0.6):
    if not lineas:
        return
    alto = int(26 * escala / 0.55)
    ancho = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, escala, 1)[0][0] for l in lineas) + 20
    x, y = org
    capa = img.copy()
    cv2.rectangle(capa, (x, y), (x + ancho, y + alto * len(lineas) + 8), (25, 25, 25), -1)
    cv2.addWeighted(capa, alfa, img, 1 - alfa, 0, img)
    for i, l in enumerate(lineas):
        cv2.putText(img, l, (x + 10, y + alto * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255), 1, cv2.LINE_AA)


def dibujar_estela(img, pts, color, grosor=2):
    if len(pts) > 1:
        cv2.polylines(img, [np.int32(pts).reshape(-1, 1, 2)], False, color, grosor, cv2.LINE_AA)


class SalidaVideo:
    def __init__(self, ruta, fps):
        self.ruta, self.fps, self.w = ruta, fps, None

    def escribir(self, img):
        if self.w is None:
            os.makedirs(os.path.dirname(self.ruta) or ".", exist_ok=True)
            self.w = cv2.VideoWriter(self.ruta, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (img.shape[1], img.shape[0]))
        self.w.write(img)

    def cerrar(self):
        if self.w is not None:
            self.w.release()


def tecla(mostrar, espera=1):
    return cv2.waitKey(espera) & 0xFF if mostrar else 255
