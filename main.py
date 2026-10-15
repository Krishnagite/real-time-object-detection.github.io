import os
import streamlit as st
from dotenv import load_dotenv
from database import DatabaseManager
from detector import ObjectDetector
from ui import render_dashboard

st.set_page_config(
    page_title="Real-Time Object Detection Platform",
    page_icon="🚀",
    layout="wide",
)


@st.cache_resource
def get_detector():
    return ObjectDetector(os.getenv("YOLO_MODEL_PATH", "yolov8n.pt"))


@st.cache_resource
def get_db():
    return DatabaseManager()


def main():
    load_dotenv()
    render_dashboard(get_detector(), get_db())


if __name__ == "__main__":
    main()
