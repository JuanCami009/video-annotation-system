# Sistema de Anotación de Video

Sistema de detección de actividades e inclinaciones en tiempo real utilizando MediaPipe y Random Forest. Este proyecto permite analizar posturas corporales y clasificar actividades mediante cámara web.

## Requisitos

- **Python**: 3.12
- **Sistema Operativo**: Linux / macOS / Windows
- **Hardware**: Cámara web

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/JuanCami009/video-annotation-system.git
cd video-annotation-system/Entrega3
```

### 2. Crear un entorno virtual

```bash
python3.12 -m venv venv
```

### 3. Activar el entorno virtual

**En Linux/macOS:**
```bash
source venv/bin/activate
```

**En Windows:**
```bash
venv\Scripts\activate
```

### 4. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Estructura del Proyecto

```
Entrega3/
├── app/
│   └── streamlit_app.py      # Aplicación web principal
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuración del proyecto
│   ├── features.py            # Extracción de características
│   ├── inference.py           # Predicción con modelo
│   └── pose_tracking.py       # Seguimiento de poses con MediaPipe
├── models/
│   └── activity_classifier.pkl # Modelo entrenado (Random Forest)
├── docs/                      # Documentación adicional
├── requirements.txt           # Dependencias del proyecto
└── README.md
```

## Uso

### Ejecutar la aplicación

Para iniciar la aplicación de detección en tiempo real:

```bash
streamlit run app/streamlit_app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`.

### Uso de la interfaz

1. **Activar cámara**: Marca la casilla "Iniciar cámara"
2. **Visualización**: Observa el análisis de poses en tiempo real
3. **Predicciones**: Las actividades detectadas se mostrarán en pantalla
4. **Detener**: Desmarca la casilla para detener la cámara

## Características Principales

- **Detección de poses en tiempo real** usando MediaPipe
- **Clasificación de actividades** mediante Random Forest
- **Análisis de inclinaciones** corporales
- **Interfaz web interactiva** con Streamlit
- **Procesamiento de video optimizado** con OpenCV

