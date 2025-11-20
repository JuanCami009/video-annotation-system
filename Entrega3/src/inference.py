import numpy as np
import pandas as pd
import joblib

from .config import MODEL_PATH

# Cargamos paquete entrenado
_paquete = joblib.load(MODEL_PATH)

_model = _paquete["modelo"]
_label_encoder = _paquete["label_encoder"]
_feature_columns = _paquete["feature_columns"]
_scaler = _paquete.get("scaler", None)  # opcional, puede no usarse


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

    # Decidimos cómo alimentar al modelo para evitar warnings
    # Si el modelo fue entrenado con nombres de columnas, usamos DataFrame.
    if hasattr(_model, "feature_names_in_"):
        X_for_model = df  # mantiene nombres de columnas
    else:
        # Modelo entrenado sin nombres -> usamos ndarray
        X_for_model = df.values

    # Si hay scaler y fue pensado para usarse aquí, lo aplicamos SOBRE X_for_model
    if _scaler is not None:
        # El scaler de sklearn soporta DataFrame o ndarray, así que le pasamos lo mismo
        X_scaled = _scaler.transform(X_for_model)
        X_input = X_scaled
    else:
        X_input = X_for_model

    # Predicción
    y_pred = _model.predict(X_input)

    # Invertimos la codificación a etiqueta original
    action = _label_encoder.inverse_transform(y_pred)[0]

    return str(action)
