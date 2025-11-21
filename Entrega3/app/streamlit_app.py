import os
import sys
import time

import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Añadir la raíz del proyecto al sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)  # carpeta Entrega3

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pose_tracking import get_pose_tracker, process_frame
from src.features import FeatureAccumulator
from src.inference import predict_from_features
from src.config import WINDOW_SIZE

# Mínimo de frames para empezar a predecir
MIN_FRAMES_FOR_PRED = max(10, WINDOW_SIZE // 2)

# Mapeo de acciones a etiquetas amigables
ACTION_LABELS = {
    "caminar_hacia_adelante": "Caminando Adelante",
    "caminar_atras": "Caminando Atrás",
    "girar_derecha": "Girando Derecha",
    "sentarse": "Sentándose",
    "pararse": "Parándose",
    "Recolectando datos...": "Recolectando datos...",
    "Sin detección": "Sin detección",
}


class PoseActivityProcessor(VideoTransformerBase):
    """
    Procesa el video en tiempo real (WebRTC):
    - Detecta pose con MediaPipe
    - Calcula features con FeatureAccumulator
    - Predice actividad con tu modelo
    - Dibuja overlay con la actividad
    """

    def __init__(self):
        self.pose = get_pose_tracker()
        # Asumimos ~30 fps; en la nube no tenemos CAP_PROP_FPS real
        self.accumulator = FeatureAccumulator(fps=30.0, window_size=WINDOW_SIZE)
        self.last_action = "Sin detección"
        self.last_display_action = ACTION_LABELS.get(self.last_action, self.last_action)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Frame en BGR
        img = frame.to_ndarray(format="bgr24")

        # Espejo horizontal para que se vea natural tipo selfie
        img = cv2.flip(img, 1)

        # Pose + dibujo del esqueleto
        landmarks_px, frame_draw = process_frame(img, self.pose)

        # Actualizamos acumulador de features
        self.accumulator.update(landmarks_px, frame_draw)

        # Construimos vector de características
        feature_vector = self.accumulator.build_feature_vector()

        # Lógica de predicción
        if feature_vector is None:
            action = "Sin detección"
        else:
            values = np.array(list(feature_vector.values()), dtype=float)

            if (
                self.accumulator.total_frames_seen < MIN_FRAMES_FOR_PRED
                or np.isnan(values).any()
            ):
                action = "Recolectando datos..."
            else:
                action = predict_from_features(feature_vector)

        # Guardamos acción "cruda"
        self.last_action = action

        # Transformamos a etiqueta amigable
        display_action = ACTION_LABELS.get(action, str(action))
        self.last_display_action = display_action

        # Overlay en la imagen (reutilizamos tu diseño)
        overlay_height = 50
        overlay_width = 400
        cv2.rectangle(
            frame_draw,
            (10, 10),
            (10 + overlay_width, 10 + overlay_height),
            (0, 0, 0),
            thickness=-1,
        )
        cv2.putText(
            frame_draw,
            display_action,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Devolvemos el frame modificado
        return av.VideoFrame.from_ndarray(frame_draw, format="bgr24")


def main():
    # Configuración responsive para móviles
    st.set_page_config(
        page_title="Detector de Actividades",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            "About": "Sistema de detección de actividades en tiempo real",
        },
    )

    # CSS responsive y estilos
    st.markdown(
        """
        <style>
        /* Responsive para móviles */
        @media (max-width: 768px) {
            .main .block-container {
                padding: 1rem 0.5rem;
                max-width: 100%;
            }
            h1 {
                font-size: 1.5rem !important;
            }
            .stButton button {
                width: 100%;
                height: 3rem;
                font-size: 1.2rem;
            }
        }

        /* Estilo para el indicador de actividad */
        .activity-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            color: white;
            font-size: 1.5rem;
            font-weight: bold;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        /* Hacer video responsive */
        video {
            max-width: 100% !important;
            height: auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Detector de Actividades")
    st.markdown("Detección en tiempo real usando IA (MediaPipe + RandomForest)")

    # Estado para mostrar/ocultar el componente de cámara
    if "show_webrtc" not in st.session_state:
        st.session_state["show_webrtc"] = False

    col1, col2 = st.columns([2, 1])

    with col2:
        btn_label = (
            "Iniciar Cámara"
            if not st.session_state["show_webrtc"]
            else "Detener Cámara"
        )
        if st.button(btn_label, use_container_width=True):
            st.session_state["show_webrtc"] = not st.session_state["show_webrtc"]

    # Placeholder para mostrar actividad
    activity_placeholder = col2.empty()

    if not st.session_state["show_webrtc"]:
        st.info("Presiona el botón para iniciar la cámara.")
        return

    # Componente WebRTC (usa la cámara del navegador, no cv2.VideoCapture)
    webrtc_ctx = webrtc_streamer(
        key="pose-detector",
        video_processor_factory=PoseActivityProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Información lateral (estática, pero útil)
    with col2:
        st.metric("Ventana de análisis", f"{WINDOW_SIZE} frames mínimo")
        st.metric("Frames requeridos para predecir", f"{MIN_FRAMES_FOR_PRED}")

    # Mostramos la última acción conocida en el recuadro bonito.
    # NOTA: el overlay del video SIEMPRE estará actualizado;
    # este cuadro puede actualizarse cuando la app se rerenderice.
    if webrtc_ctx.video_processor:
        display_action = webrtc_ctx.video_processor.last_display_action
    else:
        display_action = "Inicializando cámara..."

    activity_placeholder.markdown(
        f'<div class="activity-box">{display_action}</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
