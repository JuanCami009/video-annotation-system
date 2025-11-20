import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def get_pose_tracker():
    """
    Crea y devuelve un objeto Pose de MediaPipe con la misma configuración
    que usaste en el notebook.
    """
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,          # equilibrio latencia/precisión
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return pose


def landmarks_to_pixel_list(landmarks, width, height, min_visibility=0.5):
    """
    Convierte landmarks normalizados a (x_px, y_px) o None si la visibilidad es baja.
    Es la misma lógica que en tu notebook.
    """
    pts = []
    for lm in landmarks:
        x_px = int(lm.x * width)
        y_px = int(lm.y * height)
        if lm.visibility is not None and lm.visibility < min_visibility:
            pts.append(None)
        else:
            pts.append((x_px, y_px))
    return pts


def process_frame(frame_bgr, pose):
    """
    Procesa un frame BGR:
      - corre MediaPipe Pose
      - dibuja el esqueleto
      - devuelve lista de puntos en píxeles y el frame dibujado
    """
    h, w, _ = frame_bgr.shape

    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    landmarks_px = None
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame_bgr,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
        )
        landmarks_px = landmarks_to_pixel_list(
            results.pose_landmarks.landmark, w, h, min_visibility=0.5
        )

    return landmarks_px, frame_bgr
