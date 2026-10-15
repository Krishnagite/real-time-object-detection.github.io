import os
import socket

import streamlit as st
from dotenv import load_dotenv

from database import DatabaseManager
from detector import ObjectDetector
from ui import render_dashboard

# 1. Page configuration must be the very first Streamlit command
st.set_page_config(
    page_title="Real-Time Object Detection & Logging Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def get_local_wifi_ip() -> str:
    """Detects the host machine's local LAN/Wi-Fi IP address for mobile QR connection."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        # Doesn't actually send packets; resolves local outbound interface IP
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"


@st.cache_resource(show_spinner="Loading YOLOv8n model weights...")
def load_cached_detector() -> ObjectDetector:
    """Loads the YOLOv8n model once and caches it across Streamlit reruns."""
    model_weights = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
    return ObjectDetector(model_path=model_weights)


@st.cache_resource(show_spinner="Connecting to MySQL database...")
def load_cached_database() -> DatabaseManager:
    """Initializes the MySQL DatabaseManager once across sessions."""
    return DatabaseManager()


def main():
    load_dotenv()
    local_ip = get_local_wifi_ip()
    detector = load_cached_detector()
    db_manager = load_cached_database()

    render_dashboard(
        detector=detector,
        db_manager=db_manager,
        local_ip=local_ip,
        port=8501,
    )


if __name__ == "__main__":
    main()
