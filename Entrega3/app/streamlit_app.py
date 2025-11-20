import os
import sys
import time

import cv2
import numpy as np
import streamlit as st

# Añadir la raíz del proyecto al sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)  # carpeta Entrega3

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pose_tracking import get_pose_tracker, process_frame
from src.features import FeatureAccumulator
from src.inference import predict_from_features
from src.config import WINDOW_SIZE


def main():
    st.set_page_config(page_title="Sistema de anotación de video", layout="wide")

    st.title("Sistema de anotación de video")
    st.write("Detección de actividades e inclinaciones en tiempo real (RandomForest).")

    run = st.checkbox("Iniciar cámara")

    frame_placeholder = st.empty()
    activity_placeholder = st.empty()

    if not run:
        st.info("Activa la cámara para iniciar el análisis.")
        return

    # Abrimos cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("No se pudo acceder a la cámara.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0  # valor por defecto

    pose = get_pose_tracker()
    accumulator = FeatureAccumulator(fps=fps, window_size=WINDOW_SIZE)

    # Mínimo de frames para empezar a predecir
    MIN_FRAMES_FOR_PRED = max(5, WINDOW_SIZE // 2)

    st.write(f"FPS estimado: {fps:.1f} | Ventana de frames: {WINDOW_SIZE}")
    stop = st.button("Detener")

    while True:
        if stop:
            break

        ret, frame = cap.read()
        if not ret:
            st.warning("No se pudo leer frame de la cámara.")
            break

        # Espejo horizontal
        frame = cv2.flip(frame, 1)

        # Landmarks + dibujo del esqueleto
        landmarks_px, frame_draw = process_frame(frame, pose)

        # Actualizar acumulador de features
        accumulator.update(landmarks_px, frame_draw)

        # Construir vector de características
        feature_vector = accumulator.build_feature_vector()

        # Lógica para decidir si predecimos o no
        if feature_vector is None:
            action = "Sin detección"
        else:
            values = np.array(list(feature_vector.values()), dtype=float)

            # Condiciones para NO llamar al modelo:
            # - muy pocos frames acumulados
            # - hay NaNs en las características
            if (
                accumulator.total_frames_seen < MIN_FRAMES_FOR_PRED
                or np.isnan(values).any()
            ):
                action = "Recolectando datos..."
            else:
                action = predict_from_features(feature_vector)

        # Texto a mostrar en el overlay del frame
        if action == "Recolectando datos...":
            overlay_text = "Recolectando datos..."
        else:
            overlay_text = f"Actividad: {action}"

        # Overlay en la imagen
        cv2.rectangle(frame_draw, (10, 10), (10 + 320, 10 + 30), (0, 0, 0), thickness=-1)
        cv2.putText(
            frame_draw,
            overlay_text,
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Mostrar en Streamlit
        activity_placeholder.markdown(f"**Estado / Actividad:** `{action}`")
        frame_placeholder.image(frame_draw, channels="BGR")

        # Pequeña pausa para no saturar la CPU
        time.sleep(0.01)

    cap.release()
    st.write("Cámara detenida.")


if __name__ == "__main__":
    main()
