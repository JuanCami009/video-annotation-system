import os

# Ruta base del proyecto (usa el directorio actual como raíz)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ruta al modelo entrenado
MODEL_PATH = os.path.join(BASE_DIR, "models", "activity_classifier.pkl")

# Tamaño de la ventana de frames para hacer promedio (si luego quieres)
WINDOW_SIZE = 15  # Aumentado para mejor detección temporal
