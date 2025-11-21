import numpy as np
import pandas as pd
import joblib
from collections import deque, Counter

from .config import MODEL_PATH

# Cargamos paquete entrenado
_paquete = joblib.load(MODEL_PATH)

_model = _paquete["modelo"]
_label_encoder = _paquete["label_encoder"]
_feature_columns = _paquete["feature_columns"]
_scaler = _paquete.get("scaler", None)  # opcional, puede no usarse

# Buffer para suavizar predicciones (votación por mayoría)
_prediction_history = deque(maxlen=7)  # Últimas 7 predicciones


def predict_from_features(feature_dict: dict) -> str:
    """
    Recibe un diccionario con las columnas numéricas:
      ['frames', 'duration_s', 'brightness_avg', 'hip_speed_avg_px_per_frame',
       'shoulder_tilt_avg_px', 'knee_angle_avg_deg', 'movement_avg_px']
    Las ordena y pasa por el modelo.
    Devuelve la etiqueta de acción (string) usando el LabelEncoder.
    """
    if feature_dict is None:
        return "Sin detección"

    # DataFrame con UNA fila
    df = pd.DataFrame([feature_dict])

    # Aseguramos que todas las columnas que el modelo espera estén presentes
    for col in _feature_columns:
        if col not in df.columns:
            df[col] = 0.0

    # Orden exacto de columnas
    df = df[_feature_columns]

    # Si hay scaler, aplicarlo manteniendo el DataFrame
    if _scaler is not None:
        # Transformar y mantener como DataFrame con nombres de columnas
        X_scaled = _scaler.transform(df)
        X_input = pd.DataFrame(X_scaled, columns=_feature_columns)
    else:
        # Si no hay scaler, usar datos sin normalizar como DataFrame
        X_input = df

    # Predicción (siempre con DataFrame para evitar warnings)
    y_pred = _model.predict(X_input)

    # Invertimos la codificación a etiqueta original
    action_raw = _label_encoder.inverse_transform(y_pred)[0]
    
    # Suavizado: usar votación por mayoría de las últimas predicciones
    _prediction_history.append(action_raw)
    
    # Si tenemos suficiente historial, usar la más frecuente
    if len(_prediction_history) >= 5:
        # Contar frecuencias y tomar la más común
        vote_counts = Counter(_prediction_history)
        action = vote_counts.most_common(1)[0][0]
    else:
        action = action_raw

    return str(action)
