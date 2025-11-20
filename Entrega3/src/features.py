import math
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

# Índices de landmarks (33 puntos)
LHIP  = mp_pose.PoseLandmark.LEFT_HIP.value
RHIP  = mp_pose.PoseLandmark.RIGHT_HIP.value
LSHO  = mp_pose.PoseLandmark.LEFT_SHOULDER.value
RSHO  = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
LKNE  = mp_pose.PoseLandmark.LEFT_KNEE.value
RKNE  = mp_pose.PoseLandmark.RIGHT_KNEE.value
LANK  = mp_pose.PoseLandmark.LEFT_ANKLE.value
RANK  = mp_pose.PoseLandmark.RIGHT_ANKLE.value


def safe_nanmean(values):
    """
    Hace la media ignorando NaN, pero:
      - Si la lista está vacía -> devuelve np.nan sin warning
      - Si todos los valores son NaN -> devuelve np.nan sin warning
    """
    arr = np.asarray(values, dtype=float)

    # Si no hay elementos, devolvemos NaN
    if arr.size == 0:
        return np.nan

    # Filtramos NaN
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return np.nan

    return float(np.mean(valid))



def angle_between(p_a, p_b, p_c):
    """
    Ángulo ABC (en grados) con vértice en B.
    p_* son tuplas (x, y) en píxeles.
    """
    a, b, c = map(lambda p: np.array(p, dtype=float), (p_a, p_b, p_c))
    v1, v2 = a - b, c - b
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return np.nan
    cosang = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    # Nos aseguramos de devolver siempre un escalar float
    return float(ang)


def hip_center(landmarks_px):
    """Centro de cadera (promedio de cadera izq/der) en píxeles."""
    lh, rh = landmarks_px[LHIP], landmarks_px[RHIP]
    return ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)


def shoulder_inclination(landmarks_px):
    """
    Inclinación de hombros (diferencia vertical, en píxeles).
    Positivo si el hombro derecho está más abajo que el izquierdo.
    """
    lsh, rsh = landmarks_px[LSHO], landmarks_px[RSHO]
    return rsh[1] - lsh[1]


def average_knee_angle(landmarks_px):
    """
    Ángulo promedio de rodillas (en grados) combinando izquierda y derecha.
    Devuelve NaN si no hay datos suficientes SIN usar np.isnan en arrays.
    """
    vals = []

    # Rodilla izquierda
    if all(
        landmarks_px[i] is not None
        for i in (LHIP, LKNE, LANK)
    ):
        lk = angle_between(
            landmarks_px[LHIP],
            landmarks_px[LKNE],
            landmarks_px[LANK],
        )
        # lk ya es float o np.nan
        if not np.isnan(lk):
            vals.append(float(lk))

    # Rodilla derecha
    if all(
        landmarks_px[i] is not None
        for i in (RHIP, RKNE, RANK)
    ):
        rk = angle_between(
            landmarks_px[RHIP],
            landmarks_px[RKNE],
            landmarks_px[RANK],
        )
        if not np.isnan(rk):
            vals.append(float(rk))

    if not vals:
        return np.nan

    return float(np.mean(vals))


def avg_frame_brightness(frame_bgr):
    """Brillo promedio (0-255) del frame en escala de grises."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def interframe_displacement(prev_points, curr_points, use_subset=True):
    """
    Movimiento promedio: distancia euclídea entre frames consecutivos
    (promedio sobre un subconjunto estable de landmarks).
    """
    if prev_points is None or curr_points is None:
        return np.nan

    idxs = [LSHO, RSHO, LHIP, RHIP, LKNE, RKNE, LANK, RANK] if use_subset else range(len(curr_points))
    dists = []
    for i in idxs:
        p, q = prev_points[i], curr_points[i]
        if p is None or q is None:
            continue
        dists.append(math.dist(p, q))

    return float(np.mean(dists)) if dists else np.nan


class FeatureAccumulator:
    """
    Acumula métricas por frame en una ventana deslizante y construye
    un vector de características compatible con las columnas de df_norm:

    ['frames', 'duration_s', 'brightness_avg', 'hip_speed_avg_px_per_frame',
     'shoulder_tilt_avg_px', 'knee_angle_avg_deg', 'movement_avg_px']
    """

    def __init__(self, fps: float = 30.0, window_size: int = 15):
        self.fps = fps if fps and fps > 0 else 30.0
        self.window_size = window_size

        self.hip_speeds = deque(maxlen=window_size)
        self.shoulder_tilts = deque(maxlen=window_size)
        self.knee_angles = deque(maxlen=window_size)
        self.movements = deque(maxlen=window_size)
        self.brightness_vals = deque(maxlen=window_size)

        self.prev_hip_center = None
        self.prev_pts = None

        self.total_frames_seen = 0  # solo informativo

    def update(self, landmarks_px, frame_bgr):
        """
        Actualiza las métricas con un nuevo frame.
        Debe llamarse en cada iteración de la cámara.
        """
        self.total_frames_seen += 1

        hc = None
        sh_inc = np.nan
        knee = np.nan

        if landmarks_px is not None and len(landmarks_px) > 0:
            try:
                hc = hip_center(landmarks_px)
            except Exception:
                hc = None

            try:
                sh_inc = shoulder_inclination(landmarks_px)
            except Exception:
                sh_inc = np.nan

            try:
                knee = average_knee_angle(landmarks_px)
            except Exception:
                knee = np.nan

        # Velocidad de cadera
        hip_speed = np.nan
        if hc is not None and self.prev_hip_center is not None:
            hip_speed = math.dist(hc, self.prev_hip_center)

        # Movimiento promedio entre frames
        move_avg = interframe_displacement(self.prev_pts, landmarks_px, use_subset=True)

        # Brillo
        bri = avg_frame_brightness(frame_bgr)

        # Guardamos en las ventanas
        self.hip_speeds.append(hip_speed)
        self.shoulder_tilts.append(sh_inc)
        self.knee_angles.append(knee)
        self.movements.append(move_avg)
        self.brightness_vals.append(bri)

        # Actualizamos "prev"
        self.prev_hip_center = hc
        self.prev_pts = landmarks_px

    def build_feature_vector(self):
        """
        Construye un diccionario con las mismas columnas que usaste para entrenar:
        frames, duration_s, brightness_avg, hip_speed_avg_px_per_frame,
        shoulder_tilt_avg_px, knee_angle_avg_deg, movement_avg_px
        Usando promedios en la ventana actual.
        """
        n = len(self.brightness_vals)
        if n == 0:
            return None

        duration_s = n / self.fps

        brightness_avg = safe_nanmean(self.brightness_vals)
        hip_speed_avg = safe_nanmean(self.hip_speeds)
        shoulder_tilt_avg = safe_nanmean(self.shoulder_tilts)
        knee_angle_avg = safe_nanmean(self.knee_angles)
        movement_avg = safe_nanmean(self.movements)

        features = {
            "frames": float(n),
            "duration_s": float(duration_s),
            "brightness_avg": brightness_avg,
            "hip_speed_avg_px_per_frame": hip_speed_avg,
            "shoulder_tilt_avg_px": shoulder_tilt_avg,
            "knee_angle_avg_deg": knee_angle_avg,
            "movement_avg_px": movement_avg,
            # 'action' NO se incluye: el modelo la predice
        }
        return features
