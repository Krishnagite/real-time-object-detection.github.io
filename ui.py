import os
import time
import tempfile
import urllib.request
import cv2
import numpy as np
import streamlit as st

SAMPLE_VIDEO_URL = "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4"


def ensure_sample_video() -> str:
    path = "sample_video.mp4"
    if not os.path.exists(path):
        urllib.request.urlretrieve(SAMPLE_VIDEO_URL, path)
    return path


def render_dashboard(detector, db_manager):
    if "is_running" not in st.session_state:
        st.session_state.is_running = False

    st.title("🚀 Real-Time Object Detection & Logging Platform")
    st.caption("Assignment 5 • Python + OpenCV + YOLOv8n + MySQL (`vision_platform`)")

    default_cls = [
        c
        for c in ["person", "bicycle", "car", "cell phone", "laptop", "bottle"]
        if c in detector.class_names
    ]

    with st.expander("⚙️ Detection, Object Filter & Database Settings", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            source_mode = st.selectbox(
                "📹 Select Video Input Source",
                [
                    "1. Live Real-Time Stream (Built-in Street & Person Feed)",
                    "2. Phone / Laptop Browser Camera (Live Snapshot)",
                    "3. Upload Local Video File (.mp4)",
                    "4. Local USB Webcam (cv2.VideoCapture(0))",
                ],
            )
            allowed_classes = st.multiselect(
                "🎯 Object Configuration Filter (COCO Classes)",
                detector.class_names,
                default=default_cls,
            )
        with c2:
            conf_threshold = st.slider(
                "🎚️ Confidence Threshold", 0.10, 1.00, 0.65, 0.05
            )
            cooldown = st.slider(
                "⏱️ Database Logging Interval (sec)", 0.2, 5.0, 1.0, 0.2
            )
            st.success("🟢 Database Connected (`vision_platform.detection_logs`)")

    k1, k2, k3 = st.columns(3)
    k1_spot, k2_spot, k3_spot = k1.empty(), k2.empty(), k3.empty()

    def update_kpis():
        m = db_manager.get_summary_metrics()
        k1_spot.metric("📊 Total Logged Detections", m["total_detections"])
        k2_spot.metric("🏆 Top Detected Object", m["top_class"])
        k3_spot.metric("🎯 Avg Confidence", f"{m['avg_confidence']}%")

    update_kpis()
    st.divider()

    feed_col, log_col = st.columns([1.25, 1.0])

    with log_col:
        st.subheader("🗄️ Live Database Logs (`detection_logs`)")
        table_spot = st.empty()
        table_spot.dataframe(
            db_manager.fetch_recent_logs(20),
            use_container_width=True,
            hide_index=True,
        )

    with feed_col:
        st.subheader("🎥 Annotated Detection Feed")

        if source_mode.startswith("1.") or source_mode.startswith("4."):
            b1, b2 = st.columns(2)
            if b1.button("▶️ Start Real-Time Detection", type="primary"):
                st.session_state.is_running = True
            if b2.button("⏹️ Stop Detection"):
                st.session_state.is_running = False

            frame_spot = st.empty()
            if st.session_state.is_running:
                if source_mode.startswith("4."):
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(ensure_sample_video())
                else:
                    cap = cv2.VideoCapture(ensure_sample_video())

                last_ui = 0.0
                while st.session_state.is_running and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    annotated, dets = detector.process_frame(
                        frame, allowed_classes, conf_threshold
                    )
                    for d in dets:
                        db_manager.insert_detection(
                            d["object_class"],
                            d["confidence"],
                            d["bbox_x"],
                            d["bbox_y"],
                            d["bbox_w"],
                            d["bbox_h"],
                            cooldown_seconds=cooldown,
                        )
                    frame_spot.image(
                        annotated, channels="BGR", use_container_width=True
                    )
                    if time.time() - last_ui > 1.0:
                        table_spot.dataframe(
                            db_manager.fetch_recent_logs(20),
                            use_container_width=True,
                            hide_index=True,
                        )
                        update_kpis()
                        last_ui = time.time()
                cap.release()
            else:
                st.info("👆 Click **Start Real-Time Detection** to start the live feed.")

        elif source_mode.startswith("2."):
            snap = st.camera_input("Use your phone or laptop camera")
            if snap is not None:
                raw = np.asarray(bytearray(snap.read()), dtype=np.uint8)
                frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                annotated, dets = detector.process_frame(
                    frame, allowed_classes, conf_threshold
                )
                for d in dets:
                    db_manager.insert_detection(
                        d["object_class"],
                        d["confidence"],
                        d["bbox_x"],
                        d["bbox_y"],
                        d["bbox_w"],
                        d["bbox_h"],
                        cooldown_seconds=0.0,
                    )
                st.image(annotated, channels="BGR", use_container_width=True)
                table_spot.dataframe(
                    db_manager.fetch_recent_logs(20),
                    use_container_width=True,
                    hide_index=True,
                )
                update_kpis()

        elif source_mode.startswith("3."):
            up_vid = st.file_uploader("Upload .mp4 Video", type=["mp4", "avi", "mov"]) 
            if up_vid and st.button("▶️ Process Uploaded Video", type="primary"):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(up_vid.read())
                tfile.flush()
                cap = cv2.VideoCapture(tfile.name)
                frame_spot = st.empty()
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    annotated, dets = detector.process_frame(
                        frame, allowed_classes, conf_threshold
                    )
                    for d in dets:
                        db_manager.insert_detection(
                            d["object_class"],
                            d["confidence"],
                            d["bbox_x"],
                            d["bbox_y"],
                            d["bbox_w"],
                            d["bbox_h"],
                            cooldown_seconds=cooldown,
                        )
                    frame_spot.image(
                        annotated, channels="BGR", use_container_width=True
                    )
                cap.release()
                table_spot.dataframe(
                    db_manager.fetch_recent_logs(20),
                    use_container_width=True,
                    hide_index=True,
                )
                update_kpis()
