import json
import hashlib
import tempfile

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

from camera import enhance_live_frame, process_frame, process_video_file
from pothole_core import compute_overall_road_assessment

from mongodb_utils import (
    save_detection,
    get_history,
    delete_all_history,
    mongodb_available,
)


st.set_page_config(
    page_title="X-PotholeNet Pro",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
.block-container {
    max-width: 1480px;
    padding-top: 2.5rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

.main-title {
    font-size: 2.5rem;
    font-weight: 800;
    line-height: 1.25;
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
    padding-top: 0.2rem;
    overflow: visible;
}
.sub-title {
    color: #94A3B8;
    margin-bottom: 1rem;
}
.metric-card {
    background: linear-gradient(180deg, #0F172A, #111827);
    border: 1px solid #1E293B;
    border-radius: 20px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 10px 24px rgba(0,0,0,0.22);
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
}
.metric-label {
    color: #94A3B8;
    font-size: 0.95rem;
}
.alert-box {
    padding: 14px 18px;
    border-radius: 14px;
    font-weight: 700;
    margin-top: 8px;
    margin-bottom: 14px;
}
.legend-pill {
    display:inline-block;
    padding:7px 14px;
    border-radius:999px;
    margin-right:8px;
    margin-bottom:6px;
    font-size:0.95rem;
    font-weight:700;
}
hr {
    border: none;
    border-top: 1px solid #1E293B;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🛣️ X-PotholeNet Pro</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Stable multi-model pothole dashboard with ML-based severity, road danger assessment, tracking, and explainable AI.</div>',
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("⚙️ Controls")

    mode = st.selectbox(
        "Detection Mode",
        ["Standard", "Rainy / Wet Road", "Low Light / Night"],
        index=0,
    )

    conf_threshold = st.slider(
        "Confidence Threshold", 0.10, 0.90, 0.25, 0.05
    )
    iou_threshold = st.slider(
        "YOLO IoU Threshold", 0.10, 0.90, 0.45, 0.05
    )
    overlap_threshold = st.slider(
        "Pothole Merge Threshold", 0.10, 0.90, 0.20, 0.01
    )
    min_width = st.slider("Minimum Box Width", 8, 150, 16, 2)
    min_height = st.slider("Minimum Box Height", 8, 150, 12, 2)

    st.markdown("---")
    use_validation = st.checkbox("Use frame validation", value=True)
    allow_dark_frames = st.checkbox("Allow dark frames", value=True)
    allow_blurry_frames = st.checkbox("Allow blurry frames", value=True)
    require_road_scene = st.checkbox("Reject non-road scenes", value=True)
    road_scene_threshold = st.slider(
        "Road Scene Threshold", 0.20, 0.90, 0.42, 0.02
    )

    st.markdown("---")
    show_general_objects = st.checkbox(
        "Show filtered objects count", value=True
    )
    show_explanations = st.checkbox(
        "Show explainable AI table", value=True
    )

    st.markdown("---")
    if mongodb_available():
        st.success("🟢 MongoDB connected")
    else:
        st.warning("🟡 MongoDB unavailable")


def render_alert(alert: str):
    if "UNSAFE" in alert or "EXTREME" in alert or "HIGH" in alert:
        bg = "#7F1D1D"
        border = "#DC2626"
    elif "MODERATE" in alert or "MINOR" in alert:
        bg = "#78350F"
        border = "#F59E0B"
    else:
        bg = "#064E3B"
        border = "#10B981"

    st.markdown(
        f'<div class="alert-box" style="background:{bg}; border:1px solid {border};">{alert}</div>',
        unsafe_allow_html=True,
    )


def render_metrics(counts, risk_score, general_count, road_status):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    total = counts["Low"] + counts["Medium"] + counts["High"]

    data = [
        (c1, counts["Low"], "Low"),
        (c2, counts["Medium"], "Medium"),
        (c3, counts["High"], "High"),
        (c4, total, "Total Potholes"),
        (c5, risk_score, "Risk Score"),
        (c6, road_status, "Road Status"),
    ]

    for col, value, label in data:
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if show_general_objects:
        st.caption(
            f"Filtered general objects detected by COCO model: {general_count}"
        )


def render_legend():
    st.markdown(
        """
        <span class="legend-pill" style="background:#14532d;color:#fff;">Low</span>
        <span class="legend-pill" style="background:#854d0e;color:#fff;">Medium</span>
        <span class="legend-pill" style="background:#7f1d1d;color:#fff;">High</span>
        <span class="legend-pill" style="background:#1d4ed8;color:#fff;">Explainable AI</span>
        """,
        unsafe_allow_html=True,
    )


def render_charts(counts):
    df = pd.DataFrame(list(counts.items()), columns=["Severity", "Count"])
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(5.8, 3.5))
        ax.bar(df["Severity"], df["Count"])
        ax.set_title("Severity Distribution")
        ax.set_xlabel("Severity")
        ax.set_ylabel("Count")
        st.pyplot(fig, clear_figure=True)

    with c2:
        if df["Count"].sum() > 0:
            fig2, ax2 = plt.subplots(figsize=(5.8, 3.5))
            ax2.pie(
                df["Count"],
                labels=df["Severity"],
                autopct="%1.1f%%",
            )
            ax2.set_title("Pothole Share")
            st.pyplot(fig2, clear_figure=True)
        else:
            st.info("No detections to plot.")


def save_image_result(uploaded_file, mode, counts, risk_score, road_status,
                      alert, detections, general_objects, session_prefix):
    image_bytes = uploaded_file.getvalue()
    image_key = (
        session_prefix
        + hashlib.sha256(image_bytes).hexdigest()
    )

    if image_key in st.session_state:
        return

    inserted_id = save_detection(
        source_type="image",
        filename=uploaded_file.name,
        mode=mode,
        counts=counts,
        risk_score=risk_score,
        road_status=road_status,
        alert=alert,
        detections=detections,
        general_objects_count=len(general_objects),
    )

    st.session_state[image_key] = inserted_id or "not_saved"

    if inserted_id:
        st.success("✅ Detection saved to MongoDB.")
    else:
        st.info(
            "MongoDB unavailable. Detection was not saved."
        )


def render_results(
    original_img,
    output_img,
    detections,
    counts,
    alert,
    risk_score,
    general_objects,
    road_status,
    title="Processed Output",
    key_prefix="default",
):
    st.markdown("### 🚨 Alert")
    render_alert(alert)

    st.markdown("### 📊 Summary")
    render_metrics(
        counts,
        risk_score,
        len(general_objects),
        road_status,
    )

    st.markdown("### 🧭 Legend")
    render_legend()

    st.markdown("### 🖼️ Visual Output")
    c1, c2 = st.columns(2)

    with c1:
        st.image(
            cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB),
            caption="Original Input",
            width="stretch",
        )

    with c2:
        st.image(
            cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB),
            caption=title,
            width="stretch",
        )

    st.markdown("### 📈 Analytics")
    render_charts(counts)

    st.markdown("### 📋 Detection Table")
    if detections:
        df = pd.DataFrame(detections)
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No pothole detections available.")

    if show_explanations and detections:
        st.markdown("### 🧠 Explainable AI")

        explain_df = pd.DataFrame(
            [
                {
                    "bbox": d["bbox"],
                    "severity": d["severity"],
                    "confidence": d["confidence"],
                    "ml_confidence": d["ml_confidence"],
                    "model_votes": d["model_votes"],
                    "explanation": d["explanation"],
                    "area_ratio": d["area_ratio"],
                    "darkness": d["darkness"],
                    "texture": d["texture"],
                    "bright_ratio": d["bright_ratio"],
                }
                for d in detections
            ]
        )

        st.dataframe(
            explain_df,
            width="stretch",
            hide_index=True,
        )

    st.markdown("### 📁 Export")
    st.download_button(
        "Download Detection JSON",
        data=json.dumps(detections, indent=4),
        file_name=f"{key_prefix}_detections.json",
        mime="application/json",
        key=f"download_json_{key_prefix}",
    )


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class LivePotholeProcessor(VideoProcessorBase):
    frame_count = 0
    cached_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = enhance_live_frame(img)
        self.frame_count += 1

        if self.frame_count % 3 == 0 or self.cached_frame is None:
            output_img, _, _, _, _, _, _ = process_frame(
                img=img,
                mode="Low Light / Night",
                conf_threshold=0.25,
                iou_threshold=0.45,
                overlap_threshold=0.20,
                min_width=18,
                min_height=14,
                use_validation=True,
                allow_dark_frames=True,
                allow_blurry_frames=True,
                max_width=640,
                require_road_scene=True,
                road_scene_threshold=0.42,
            )
            self.cached_frame = output_img
        else:
            output_img = self.cached_frame

        return frame.from_ndarray(
            output_img,
            format="bgr24",
        )


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🖼️ Image Upload",
        "📸 Capture Photo",
        "📹 Live Webcam",
        "🎥 Video Upload",
        "📚 History",
    ]
)


with tab1:
    uploaded_file = st.file_uploader(
        "Upload a road image",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_upload",
    )

    if uploaded_file:
        file_bytes = np.asarray(
            bytearray(uploaded_file.getvalue()),
            dtype=np.uint8,
        )
        img = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR,
        )

        if img is None:
            st.error("Could not read the uploaded image.")
        else:
            (
                output_img,
                detections,
                counts,
                alert,
                risk_score,
                general_objects,
                road_status,
            ) = process_frame(
                img=img,
                mode=mode,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                overlap_threshold=overlap_threshold,
                min_width=min_width,
                min_height=min_height,
                use_validation=use_validation,
                allow_dark_frames=allow_dark_frames,
                allow_blurry_frames=allow_blurry_frames,
                max_width=960,
                require_road_scene=require_road_scene,
                road_scene_threshold=road_scene_threshold,
            )

            render_results(
                img,
                output_img,
                detections,
                counts,
                alert,
                risk_score,
                general_objects,
                road_status,
                title="Processed Output",
                key_prefix="image_upload",
            )

            save_image_result(
                uploaded_file,
                mode,
                counts,
                risk_score,
                road_status,
                alert,
                detections,
                general_objects,
                "saved_image_",
            )


with tab2:
    camera_file = st.camera_input("Capture one road image")

    if camera_file:
        file_bytes = np.asarray(
            bytearray(camera_file.getvalue()),
            dtype=np.uint8,
        )
        img = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR,
        )

        if img is None:
            st.error("Could not read the captured image.")
        else:
            (
                output_img,
                detections,
                counts,
                alert,
                risk_score,
                general_objects,
                road_status,
            ) = process_frame(
                img=img,
                mode=mode,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                overlap_threshold=overlap_threshold,
                min_width=min_width,
                min_height=min_height,
                use_validation=use_validation,
                allow_dark_frames=allow_dark_frames,
                allow_blurry_frames=allow_blurry_frames,
                max_width=960,
                require_road_scene=require_road_scene,
                road_scene_threshold=road_scene_threshold,
            )

            render_results(
                img,
                output_img,
                detections,
                counts,
                alert,
                risk_score,
                general_objects,
                road_status,
                title="Captured Image Output",
                key_prefix="camera_capture",
            )

            camera_key = (
                "saved_camera_"
                + hashlib.sha256(
                    camera_file.getvalue()
                ).hexdigest()
            )

            if camera_key not in st.session_state:
                inserted_id = save_detection(
                    source_type="camera",
                    filename="camera_capture.jpg",
                    mode=mode,
                    counts=counts,
                    risk_score=risk_score,
                    road_status=road_status,
                    alert=alert,
                    detections=detections,
                    general_objects_count=len(general_objects),
                )

                st.session_state[camera_key] = (
                    inserted_id or "not_saved"
                )

                if inserted_id:
                    st.success(
                        "✅ Camera detection saved to MongoDB."
                    )
                else:
                    st.info(
                        "MongoDB unavailable. "
                        "Detection was not saved."
                    )


with tab3:
    st.markdown("### 📹 Live Webcam Detection")
    st.caption(
        "Indoor scenes may be rejected. For best results, point the webcam "
        "at a road scene or road video on another screen."
    )

    webrtc_streamer(
        key="live_pothole_detection",
        video_processor_factory=LivePotholeProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )


with tab4:
    uploaded_video = st.file_uploader(
        "Upload road video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload",
    )

    if uploaded_video:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4",
        ) as temp_in:
            temp_in.write(uploaded_video.getvalue())
            input_video_path = temp_in.name

        st.video(input_video_path)

        if st.button(
            "Process Video",
            type="primary",
            key="process_video_button",
        ):
            with st.spinner("Processing video..."):
                (
                    output_video_path,
                    detections,
                    counts,
                    alert,
                ) = process_video_file(
                    video_path=input_video_path,
                    mode=mode,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    overlap_threshold=overlap_threshold,
                    min_width=min_width,
                    min_height=min_height,
                    use_validation=use_validation,
                    allow_dark_frames=allow_dark_frames,
                    allow_blurry_frames=allow_blurry_frames,
                    max_width=640,
                    frame_skip=3,
                    require_road_scene=require_road_scene,
                    road_scene_threshold=road_scene_threshold,
                )

            risk_score, road_status = compute_overall_road_assessment(
                counts
            )

            st.markdown("### 🚨 Video Alert")
            render_alert(alert)

            st.markdown("### 📊 Video Summary")
            render_metrics(
                counts,
                risk_score,
                0,
                road_status,
            )

            st.markdown("### 📈 Video Analytics")
            render_charts(counts)

            st.markdown("### 🎬 Processed Video")
            st.video(output_video_path)

            st.markdown("### 📋 Unique Potholes Across Video")
            if detections:
                st.dataframe(
                    pd.DataFrame(detections),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("No detections available.")

            with open(output_video_path, "rb") as f:
                st.download_button(
                    "Download Processed Video",
                    data=f,
                    file_name="processed_output.mp4",
                    mime="video/mp4",
                    key="download_processed_video",
                )

            st.download_button(
                "Download Video Detection JSON",
                data=json.dumps(detections, indent=4),
                file_name="video_detections.json",
                mime="application/json",
                key="download_video_json",
            )

            video_key = (
                "saved_video_"
                + uploaded_video.name
                + "_"
                + str(uploaded_video.size)
            )

            if video_key not in st.session_state:
                inserted_id = save_detection(
                    source_type="video",
                    filename=uploaded_video.name,
                    mode=mode,
                    counts=counts,
                    risk_score=risk_score,
                    road_status=road_status,
                    alert=alert,
                    detections=detections,
                    general_objects_count=0,
                )

                st.session_state[video_key] = (
                    inserted_id or "not_saved"
                )

                if inserted_id:
                    st.success(
                        "✅ Video detection saved to MongoDB."
                    )
                else:
                    st.info(
                        "MongoDB unavailable. "
                        "Video result was not saved."
                    )


with tab5:
    st.markdown("### 📚 Detection History")

    st.caption(
        "Previous pothole detection results stored in MongoDB."
    )

    history = get_history(limit=100)

    if not history:
        st.info(
            "No saved detections found. "
            "Run an image, camera, or video detection first."
        )
    else:
        rows = []

        for item in history:
            timestamp = item.get("timestamp")

            if timestamp is not None:
                try:
                    timestamp = (
                        timestamp
                        .astimezone()
                        .strftime("%Y-%m-%d %H:%M:%S")
                    )
                except Exception:
                    timestamp = str(timestamp)

            severity = item.get("severity", {})

            rows.append(
                {
                    "Date/Time": timestamp,
                    "Type": item.get("source_type", ""),
                    "File": item.get("filename", ""),
                    "Low": severity.get("low", 0),
                    "Medium": severity.get("medium", 0),
                    "High": severity.get("high", 0),
                    "Total": item.get("pothole_count", 0),
                    "Risk": item.get("risk_score", 0),
                    "Road Status": item.get("road_status", ""),
                }
            )

        history_df = pd.DataFrame(rows)

        st.dataframe(
            history_df,
            width="stretch",
            hide_index=True,
        )

        st.markdown("### 🔎 View Saved Detection")

        options = list(range(len(history)))

        selected = st.selectbox(
            "Select a detection",
            options,
            format_func=lambda i: (
                f"{rows[i]['Date/Time']} — "
                f"{rows[i]['Type']} — "
                f"{rows[i]['File']}"
            ),
        )

        selected_doc = history[selected]
        detail = dict(selected_doc)

        detail.pop("_id", None)

        if hasattr(detail.get("timestamp"), "isoformat"):
            detail["timestamp"] = (
                detail["timestamp"].isoformat()
            )

        st.json(
            detail,
            expanded=False,
        )

        st.markdown("---")

        if st.button("🗑️ Clear All Detection History"):
            if delete_all_history():
                st.success("Detection history cleared.")
                st.rerun()
            else:
                st.error("Could not clear MongoDB history.")