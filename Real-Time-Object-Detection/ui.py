import io
import tempfile
import time

import cv2
import numpy as np
import qrcode
import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration


def generate_qr_code_bytes(url: str) -> bytes:
    """Generates a PNG QR code image in memory for the local Wi-Fi URL."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=6,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def inject_mobile_responsive_css():
    """Injects custom CSS so KPI cards, video feeds, and tables stack cleanly on phones."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1.5rem;
        }
        div[data-testid="stMetric"] {
            background-color: rgba(28, 131, 225, 0.08);
            border: 1px solid rgba(28, 131, 225, 0.25);
            padding: 12px 16px;
            border-radius: 10px;
        }
        .status-badge-ok {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            background-color: #16a34a;
            color: white;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .status-badge-err {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            background-color: #dc2626;
            color: white;
            font-weight: 600;
            font-size: 0.85rem;
        }
        @media (max-width: 768px) {
            .block-container {
                padding-left: 0.6rem !important;
                padding-right: 0.6rem !important;
                padding-top: 0.8rem !important;
            }
            h1 {
                font-size: 1.45rem !important;
            }
            div[data-testid="stMetricValue"] {
                font-size: 1.3rem !important;
            }
            button[kind="primary"], button[kind="secondary"] {
                width: 100% !important;
                min-height: 2.8rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(detector, db_manager, local_ip: str, port: int = 8501):
    """Renders the full Desktop + Mobile Streamlit Object Detection & Logging Dashboard."""
    inject_mobile_responsive_css()

    # Initialize session state variables
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "frame_detections_count" not in st.session_state:
        st.session_state.frame_detections_count = 0

    network_url = f"http://{local_ip}:{port}"

    # Header
    st.title("🚀 Real-Time Object Detection & MySQL Logging Platform")
    st.caption(
        "Assignment 5 • YOLOv8n + OpenCV + MySQL (`vision_platform`) • Desktop & Mobile Ready"
    )

    # Collapsible Mobile Connection & QR Code Card
    with st.expander(
        "📱 Connect Your Mobile Phone (Scan QR Code or Setup Mobile Camera)",
        expanded=False,
    ):
        qr_col1, qr_col2 = st.columns([1, 2])
        with qr_col1:
            qr_bytes = generate_qr_code_bytes(network_url)
            st.image(qr_bytes, caption=f"Scan on same Wi-Fi: {network_url}", width=180)
        with qr_col2:
            st.markdown(
                f"""
                **1. Open Dashboard on Your Phone:**
                - Connect your phone and laptop to the **same Wi-Fi network** (or mobile hotspot).
                - Scan the QR code or open **`{network_url}`** in your mobile browser.

                **2. Three Ways to Use Your Mobile Camera:**
                - **Option A — Mobile IP Camera Stream (Recommended for Fast FPS):** Install the free **IP Webcam** (Android) or **DroidCam** (iOS/Android) app, start the server, and paste the stream URL (e.g., `http://192.168.1.5:8080/video`) into Mode 2 below.
                - **Option B — Mobile Browser Live Camera (WebRTC):** Select Mode 3 below to stream directly from your phone browser's front or rear camera.
                - **Option C — Mobile Camera Snapshot:** Select Mode 4 below to snap photos directly from your mobile camera browser without HTTPS setup.
                """
            )

    # Default COCO classes for quick filtering
    default_classes = ["person", "cell phone", "laptop", "bottle", "chair", "book", "cup"]
    valid_defaults = [c for c in default_classes if c in detector.class_names]

    # Top Mobile-Friendly Settings Expander (also mirrored in Sidebar)
    with st.expander(
        "⚙️ Detection, Filter & Camera Settings (Touch-Friendly)",
        expanded=True,
    ):
        cfg_col1, cfg_col2, cfg_col3 = st.columns([1.3, 1.5, 1.2])

        with cfg_col1:
            source_mode = st.selectbox(
                "📹 Video Input Source",
                options=[
                    "1. Laptop / USB Webcam",
                    "2. Mobile IP Camera Stream (Wi-Fi)",
                    "3. Mobile Browser Live Camera (WebRTC)",
                    "4. Mobile Camera Snapshot / Local Video File",
                ],
                index=0,
            )

            webcam_index = 0
            ip_stream_url = "http://192.168.1.5:8080/video"
            webrtc_facing = "environment"

            if source_mode.startswith("1."):
                webcam_index = st.number_input(
                    "Webcam Device Index", min_value=0, max_value=5, value=0, step=1
                )
            elif source_mode.startswith("2."):
                ip_stream_url = st.text_input(
                    "Mobile IP Stream URL (HTTP / RTSP)",
                    value="http://192.168.1.5:8080/video",
                    help="Example from Android IP Webcam app: http://192.168.1.5:8080/video",
                )
            elif source_mode.startswith("3."):
                cam_choice = st.radio(
                    "Phone Camera Lens",
                    options=["Rear Camera (Environment)", "Front Camera (User)"],
                    horizontal=True,
                )
                webrtc_facing = "environment" if "Rear" in cam_choice else "user"

        with cfg_col2:
            allowed_classes = st.multiselect(
                "🎯 Object Configuration Filter (COCO Classes)",
                options=detector.class_names,
                default=valid_defaults,
                help="Only objects selected here will be drawn and logged to MySQL.",
            )
            conf_threshold = st.slider(
                "🎚️ Confidence Threshold",
                min_value=0.10,
                max_value=1.00,
                value=0.70,
                step=0.05,
                help="Only detections with confidence >= this value are logged (e.g., 0.70 = 70%).",
            )

        with cfg_col3:
            enable_db_logging = st.toggle("💾 Enable MySQL Event Logging", value=True)
            cooldown_seconds = st.slider(
                "⏱️ DB Log Cooldown (sec/class)",
                min_value=0.2,
                max_value=10.0,
                value=1.5,
                step=0.1,
                help="Prevents flooding MySQL by waiting N seconds before re-logging the same class.",
            )
            db_ok = db_manager.check_connection()
            if db_ok:
                st.markdown(
                    '<span class="status-badge-ok">🟢 MySQL Connected (vision_platform)</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="status-badge-err">🔴 MySQL Disconnected (Check .env / XAMPP)</span>',
                    unsafe_allow_html=True,
                )
                if st.button("🔄 Retry Database Connection"):
                    db_manager.init_db()
                    st.rerun()

    with st.sidebar:
        st.header("🧭 Platform Status")
        st.write(f"**Local Wi-Fi URL:** `{network_url}`")
        st.write(f"**Selected Classes:** `{len(allowed_classes)}`")
        st.write(f"**Confidence Cutoff:** `{int(conf_threshold * 100)}%`")
        if st.button("🔃 Refresh Database Logs"):
            st.rerun()

    metrics = db_manager.get_summary_metrics()
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1_placeholder = kpi1.empty()
    kpi2_placeholder = kpi2.empty()
    kpi3_placeholder = kpi3.empty()

    def update_kpi_cards(active_frame_count: int):
        current_metrics = db_manager.get_summary_metrics()
        kpi1_placeholder.metric(
            "📊 Total Logged Detections (MySQL)",
            f"{current_metrics['total_detections']:,}",
        )
        kpi2_placeholder.metric(
            "🎯 Detections in Current Frame",
            f"{active_frame_count}",
        )
        kpi3_placeholder.metric(
            "🏆 Top Logged Object (Avg Conf)",
            f"{current_metrics['top_class']}",
            delta=f"{current_metrics['avg_confidence']}% avg conf",
        )

    update_kpi_cards(st.session_state.frame_detections_count)

    st.divider()

    feed_col, log_col = st.columns([1.35, 1.0])

    with log_col:
        st.subheader("🗄️ Recent MySQL Detection Logs")
        table_placeholder = st.empty()

        def refresh_log_table():
            df_logs = db_manager.fetch_recent_logs(limit=20)
            table_placeholder.dataframe(
                df_logs,
                use_container_width=True,
                hide_index=True,
                height=420,
            )

        refresh_log_table()

    with feed_col:
        st.subheader("🎥 Live Annotated Detection Feed")

        if source_mode.startswith("1.") or source_mode.startswith("2."):
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("▶️ Start Real-Time Detection", type="primary"):
                    st.session_state.is_running = True
            with btn_col2:
                if st.button("⏹️ Stop Detection", type="secondary"):
                    st.session_state.is_running = False

            frame_placeholder = st.empty()
            status_placeholder = st.empty()

            if st.session_state.is_running:
                capture_target = (
                    int(webcam_index)
                    if source_mode.startswith("1.")
                    else ip_stream_url.strip()
                )
                try:
                    cap = cv2.VideoCapture(capture_target)
                    if not cap.isOpened():
                        status_placeholder.error(
                            f"❌ Could not open video source `{capture_target}`. "
                            "Check that your webcam is free or your Mobile IP Webcam URL is reachable."
                        )
                        st.session_state.is_running = False
                    else:
                        status_placeholder.info(
                            f"🟢 Streaming live from `{capture_target}`... Click **Stop Detection** to end."
                        )
                        last_ui_table_refresh = 0.0
                        dropped_frames = 0

                        while st.session_state.is_running:
                            ret, frame = cap.read()
                            if not ret or frame is None:
                                dropped_frames += 1
                                if dropped_frames > 30:
                                    status_placeholder.warning(
                                        "⚠️ Video stream ended or lost connection."
                                    )
                                    break
                                time.sleep(0.05)
                                continue

                            dropped_frames = 0
                            annotated_frame, detections = detector.process_frame(
                                frame=frame,
                                allowed_classes=allowed_classes,
                                conf_threshold=conf_threshold,
                                resize_width=640,
                            )

                            st.session_state.frame_detections_count = len(detections)

                            logged_any = False
                            if enable_db_logging and detections:
                                for det in detections:
                                    ok = db_manager.insert_detection(
                                        object_class=det["object_class"],
                                        confidence=det["confidence"],
                                        bbox_x=det["bbox_x"],
                                        bbox_y=det["bbox_y"],
                                        bbox_w=det["bbox_w"],
                                        bbox_h=det["bbox_h"],
                                        cooldown_seconds=cooldown_seconds,
                                    )
                                    if ok:
                                        logged_any = True

                            frame_placeholder.image(
                                annotated_frame,
                                channels="BGR",
                                use_container_width=True,
                            )

                            now = time.time()
                            if logged_any or (now - last_ui_table_refresh) > 2.0:
                                refresh_log_table()
                                update_kpi_cards(len(detections))
                                last_ui_table_refresh = now

                        cap.release()
                except Exception as e:
                    status_placeholder.error(f"❌ Camera stream error: {e}")
                    st.session_state.is_running = False
            else:
                frame_placeholder.info(
                    "👆 Click **Start Real-Time Detection** above to launch the live YOLO video feed."
                )

        elif source_mode.startswith("3."):
            st.caption(
                "📱 Streams directly from your mobile browser or laptop browser camera. "
                "(Note: Mobile browsers require `localhost` or an HTTPS tunnel like `ngrok http 8501` to grant camera permissions)."
            )

            rtc_config = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )

            def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                annotated_img, detections = detector.process_frame(
                    frame=img,
                    allowed_classes=allowed_classes,
                    conf_threshold=conf_threshold,
                    resize_width=640,
                )
                if enable_db_logging and detections:
                    for det in detections:
                        db_manager.insert_detection(
                            object_class=det["object_class"],
                            confidence=det["confidence"],
                            bbox_x=det["bbox_x"],
                            bbox_y=det["bbox_y"],
                            bbox_w=det["bbox_w"],
                            bbox_h=det["bbox_h"],
                            cooldown_seconds=cooldown_seconds,
                        )
                return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

            webrtc_streamer(
                key="yolo-mobile-webrtc",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_config,
                media_stream_constraints={
                    "video": {"facingMode": webrtc_facing},
                    "audio": False,
                },
                video_frame_callback=video_frame_callback,
                async_processing=True,
            )

            if st.button("🔃 Refresh MySQL Logs Table After WebRTC Detections"):
                refresh_log_table()
                update_kpi_cards(0)

        elif source_mode.startswith("4."):
            tab_snap, tab_video = st.tabs(
                ["📸 Mobile Camera Snapshot", "🎞️ Upload Local Video (.mp4)"]
            )

            with tab_snap:
                snap_file = st.camera_input("Take a snapshot with your phone or webcam")
                if snap_file is not None:
                    file_bytes = np.asarray(bytearray(snap_file.read()), dtype=np.uint8)
                    bgr_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    annotated_frame, detections = detector.process_frame(
                        frame=bgr_frame,
                        allowed_classes=allowed_classes,
                        conf_threshold=conf_threshold,
                        resize_width=640,
                    )
                    if enable_db_logging and detections:
                        for det in detections:
                            db_manager.insert_detection(
                                object_class=det["object_class"],
                                confidence=det["confidence"],
                                bbox_x=det["bbox_x"],
                                bbox_y=det["bbox_y"],
                                bbox_w=det["bbox_w"],
                                bbox_h=det["bbox_h"],
                                cooldown_seconds=0.0,
                            )
                    st.image(
                        annotated_frame,
                        caption=f"Detected {len(detections)} matching object(s)",
                        channels="BGR",
                        use_container_width=True,
                    )
                    refresh_log_table()
                    update_kpi_cards(len(detections))

            with tab_video:
                uploaded_video = st.file_uploader(
                    "Upload a local video file (.mp4, .avi, .mov)",
                    type=["mp4", "avi", "mov"],
                )
                if uploaded_video is not None:
                    if st.button("▶️ Run YOLO on Uploaded Video", type="primary"):
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        tfile.write(uploaded_video.read())
                        tfile.flush()

                        vid_cap = cv2.VideoCapture(tfile.name)
                        vid_frame_placeholder = st.empty()
                        last_refresh = 0.0

                        while vid_cap.isOpened():
                            ret, frame = vid_cap.read()
                            if not ret:
                                break
                            annotated_frame, detections = detector.process_frame(
                                frame=frame,
                                allowed_classes=allowed_classes,
                                conf_threshold=conf_threshold,
                                resize_width=640,
                            )
                            if enable_db_logging and detections:
                                for det in detections:
                                    db_manager.insert_detection(
                                        object_class=det["object_class"],
                                        confidence=det["confidence"],
                                        bbox_x=det["bbox_x"],
                                        bbox_y=det["bbox_y"],
                                        bbox_w=det["bbox_w"],
                                        bbox_h=det["bbox_h"],
                                        cooldown_seconds=cooldown_seconds,
                                    )
                            vid_frame_placeholder.image(
                                annotated_frame,
                                channels="BGR",
                                use_container_width=True,
                            )
                            now = time.time()
                            if (now - last_refresh) > 1.5:
                                refresh_log_table()
                                update_kpi_cards(len(detections))
                                last_refresh = now

                        vid_cap.release()
                        refresh_log_table()
                        st.success("✅ Finished processing uploaded video and logged events to MySQL.")
