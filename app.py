"""
Aerial Intelligence & Tracking System — Command Center Dashboard
Streamlit application with two modes: Static Analysis (MP4) and Live Feed (drone).
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import os

from detection import DroneVision
from utils import (
    draw_boxes,
    draw_trails,
    draw_zone,
    generate_heatmap,
    build_zone_polygon,
    draw_locked_target_info,
)

# ══════════════════════════════════════════════════════════════════════
# Page config (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Aerial Intelligence & Tracking System",
    page_icon="\U0001F6F0",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
# Custom CSS — Command Center aesthetic
# ══════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
    div[data-testid="stMetric"] {
        background: #1A1D23;
        border: 1px solid #2A2D35;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {
        color: #8B8FA3 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-family: 'Courier New', monospace;
        font-size: 1.6rem !important;
    }
    @keyframes pulse-red {
        0%, 100% { border-left-color: #FF4B4B; }
        50% { border-left-color: #FF8080; box-shadow: 0 0 12px rgba(255,75,75,0.3); }
    }
    .alert-critical {
        border-left: 4px solid #FF4B4B;
        background: rgba(255, 75, 75, 0.08);
        padding: 8px 12px; border-radius: 4px; margin: 4px 0;
        font-family: monospace; font-size: 0.85rem;
        animation: pulse-red 2s ease-in-out infinite;
    }
    .alert-nominal {
        border-left: 4px solid #00C853;
        background: rgba(0, 200, 83, 0.08);
        padding: 8px 12px; border-radius: 4px; margin: 4px 0;
        font-family: monospace; font-size: 0.85rem;
    }
    .main-header {
        font-family: monospace; font-size: 1.1rem; color: #8B8FA3;
        text-transform: uppercase; letter-spacing: 0.15em;
        border-bottom: 1px solid #2A2D35;
        padding-bottom: 8px; margin-bottom: 16px;
    }
    .live-badge {
        background: #FF4B4B; color: white; padding: 2px 10px;
        border-radius: 4px; font-size: 0.8rem; margin-right: 10px;
        animation: pulse-red 1.5s ease-in-out infinite;
    }
    section[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════
# Session state initialization
# ══════════════════════════════════════════════════════════════════════
_defaults = {
    # Static tab state
    "drone_vision": None,
    "video_cap": None,
    "is_processing": False,
    "alerts": [],
    "heatmap_acc": None,
    "total_people_seen": set(),
    "total_vehicles_seen": set(),
    "temp_file_path": None,
    "uploaded_file_name": None,
    "video_fps": 30.0,
    "video_width": 0,
    "video_height": 0,
    "video_total_frames": 0,
    # Live tab state
    "live_drone_vision": None,
    "live_video_cap": None,
    "live_feed_active": False,
    "live_alerts": [],
    "live_heatmap_acc": None,
    "live_total_people_seen": set(),
    "live_total_vehicles_seen": set(),
    "live_start_time": None,
    "live_reconnect_count": 0,
    "live_source": None,
    "live_video_width": 0,
    "live_video_height": 0,
    # Shared / lock-on
    "locked_track_id": None,
    "lock_first_seen": None,
    "_current_visible_ids": [],
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════════════════════════════════════
# Sidebar — Shared Controls
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## \U0001F3AF CONTROL PANEL")
    st.caption("Aerial Intelligence & Tracking System v1.0")

    st.divider()

    # ── Detection settings ──
    st.markdown("### Detection")
    confidence = st.slider("Confidence Threshold", 0.10, 1.0, 0.40, 0.05)
    iou_thresh = st.slider("IOU Threshold", 0.10, 1.0, 0.50, 0.05)

    st.divider()

    # ── Zone intrusion ──
    st.markdown("### Zone Intrusion")
    enable_zone = st.toggle("Enable Zone Monitoring", value=True)
    zone_x = st.slider("Zone X (%)", 0, 80, 30, 5)
    zone_y = st.slider("Zone Y (%)", 0, 80, 30, 5)
    zone_w = st.slider("Zone Width (%)", 10, 70, 40, 5)
    zone_h = st.slider("Zone Height (%)", 10, 70, 40, 5)

    st.divider()

    # ── Overlays ──
    st.markdown("### Overlays")
    show_boxes = st.toggle("Bounding Boxes", value=True)
    show_trails = st.toggle("Movement Trails", value=True)
    show_heatmap = st.toggle("Heatmap Overlay", value=False)

    st.divider()

    # ── Target Lock-On ──
    st.markdown("### Target Lock-On")
    visible_ids = st.session_state.get("_current_visible_ids", [])
    lock_options = [None] + sorted(visible_ids)
    st.selectbox(
        "Lock Target ID",
        options=lock_options,
        format_func=lambda x: "-- None --" if x is None else f"ID:{x}",
        key="locked_track_id",
    )


# ══════════════════════════════════════════════════════════════════════
# Init functions
# ══════════════════════════════════════════════════════════════════════
def _init_video(uploaded_file):
    """Write uploaded file to disk, open VideoCapture, init DroneVision."""
    if st.session_state.video_cap is not None:
        st.session_state.video_cap.release()
        st.session_state.video_cap = None

    if st.session_state.temp_file_path and os.path.exists(
        st.session_state.temp_file_path
    ):
        try:
            os.unlink(st.session_state.temp_file_path)
        except PermissionError:
            pass

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    st.session_state.temp_file_path = tmp.name

    cap = cv2.VideoCapture(tmp.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    st.session_state.video_cap = cap
    st.session_state.video_fps = fps
    st.session_state.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    st.session_state.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    st.session_state.video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    dv = DroneVision(confidence=confidence, iou_thresh=iou_thresh)
    dv.fps = fps
    st.session_state.drone_vision = dv

    st.session_state.alerts = []
    st.session_state.heatmap_acc = None
    st.session_state.total_people_seen = set()
    st.session_state.total_vehicles_seen = set()
    st.session_state.is_processing = False
    st.session_state.uploaded_file_name = uploaded_file.name


def _init_live_feed(source) -> bool:
    """Open VideoCapture from a live source (URL or device index).

    Returns True on success, False on failure.
    """
    if st.session_state.live_video_cap is not None:
        st.session_state.live_video_cap.release()
        st.session_state.live_video_cap = None

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        st.error(f"Failed to connect to: {source}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    st.session_state.live_video_cap = cap
    st.session_state.live_source = source
    st.session_state.live_start_time = time.time()
    st.session_state.live_reconnect_count = 0
    st.session_state.live_video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    st.session_state.live_video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    dv = DroneVision(confidence=confidence, iou_thresh=iou_thresh)
    dv.fps = fps
    st.session_state.live_drone_vision = dv

    st.session_state.live_alerts = []
    st.session_state.live_heatmap_acc = None
    st.session_state.live_total_people_seen = set()
    st.session_state.live_total_vehicles_seen = set()

    return True


# ══════════════════════════════════════════════════════════════════════
# Shared processing loop
# ══════════════════════════════════════════════════════════════════════
def _run_processing_loop(
    cap,
    dv,
    video_ph,
    metric_people,
    metric_vehicles,
    metric_status,
    metric_latency,
    progress_ph,
    alert_container,
    alerts_key: str,
    heatmap_key: str,
    people_key: str,
    vehicles_key: str,
    active_flag: str,
    is_live: bool,
):
    """Run the detect → track → draw → display loop.

    Works for both static (file) and live (stream) modes.
    """
    dv.confidence = confidence
    dv.iou_thresh = iou_thresh

    if is_live:
        frame_w = st.session_state.live_video_width
        frame_h = st.session_state.live_video_height
        total_frames = 0
    else:
        frame_w = st.session_state.video_width
        frame_h = st.session_state.video_height
        total_frames = st.session_state.video_total_frames

    scale = min(1.0, 1280 / frame_w) if frame_w > 0 else 1.0
    frame_idx = 0

    while st.session_state[active_flag]:
        loop_start = time.time()

        ret, frame = cap.read()

        # ── Handle frame read failure ──
        if not ret:
            if is_live:
                # Attempt reconnection
                source = st.session_state.live_source
                reconnected = False
                for attempt in range(1, 6):
                    video_ph.warning(
                        f"Connection lost. Reconnecting... ({attempt}/5)"
                    )
                    time.sleep(2)
                    cap.release()
                    cap = cv2.VideoCapture(source)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            st.session_state.live_video_cap = cap
                            st.session_state.live_reconnect_count += 1
                            reconnected = True
                            break
                if not reconnected:
                    st.session_state[active_flag] = False
                    video_ph.error(
                        "Connection lost after 5 attempts. Click CONNECT to retry."
                    )
                    break
            else:
                st.session_state[active_flag] = False
                progress_ph.empty()
                video_ph.success(
                    "\u2705 Video processing complete. Upload new footage to continue."
                )
                break

        frame_idx += 1

        # Resolve dimensions on first successful frame (live feeds may report 0 initially)
        if frame_w == 0 or frame_h == 0:
            frame_h, frame_w = frame.shape[:2]
            scale = min(1.0, 1280 / frame_w) if frame_w > 0 else 1.0

        # Resize for performance
        if scale < 1.0:
            proc_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        else:
            proc_frame = frame.copy()

        # ── Detect & Track ──
        tracked_objects = dv.detect_and_track(proc_frame)

        # ── Update visible IDs for lock-on selectbox ──
        st.session_state._current_visible_ids = [
            obj["track_id"] for obj in tracked_objects
        ]

        # ── Lock-on logic ──
        locked_id = st.session_state.locked_track_id
        locked_obj = None
        if locked_id is not None:
            locked_obj = next(
                (o for o in tracked_objects if o["track_id"] == locked_id), None
            )
            if locked_obj is None:
                st.session_state.locked_track_id = None
                st.session_state.lock_first_seen = None
            elif st.session_state.lock_first_seen is None:
                st.session_state.lock_first_seen = time.time()

        # ── Zone intrusion ──
        intruders = []
        zone_poly = None
        if enable_zone:
            ph, pw = proc_frame.shape[:2]
            zone_poly = build_zone_polygon(pw, ph, zone_x, zone_y, zone_w, zone_h)
            intruders = dv.check_zone_intrusion(tracked_objects, zone_poly)

            alerts_list = st.session_state[alerts_key]
            for intruder in intruders:
                alert_key = f"{intruder['track_id']}_{dv.frame_count // max(int(dv.fps), 1)}"
                existing_keys = {a.get("key") for a in alerts_list}
                if alert_key not in existing_keys:
                    alerts_list.insert(
                        0,
                        {
                            "key": alert_key,
                            "time": time.strftime("%H:%M:%S"),
                            "message": (
                                f"ZONE INTRUSION: {intruder['class_name'].upper()} "
                                f"ID:{intruder['track_id']} | "
                                f"Velocity: {intruder['velocity']:.0f}px/s"
                            ),
                            "class": intruder["class_name"],
                        },
                    )
                    if len(alerts_list) > 100:
                        st.session_state[alerts_key] = alerts_list[:100]

        # ── Draw overlays ──
        display = proc_frame.copy()

        if enable_zone and zone_poly is not None:
            display = draw_zone(display, zone_poly, len(intruders) > 0)

        if show_boxes:
            display = draw_boxes(display, tracked_objects, locked_track_id=locked_id)

        if show_trails:
            display = draw_trails(display, dv.trails, dv.class_map)

        if show_heatmap:
            display, st.session_state[heatmap_key] = generate_heatmap(
                display, dv.trails, st.session_state[heatmap_key]
            )

        # Locked target info overlay
        if locked_obj is not None and st.session_state.lock_first_seen is not None:
            duration = time.time() - st.session_state.lock_first_seen
            display = draw_locked_target_info(display, locked_obj, duration)

        # ── Display frame (BGR → RGB) ──
        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        video_ph.image(display_rgb, channels="RGB", use_container_width=True)

        # ── Update counters ──
        counts = dv.get_counts(tracked_objects)
        people_set = st.session_state[people_key]
        vehicles_set = st.session_state[vehicles_key]
        for obj in tracked_objects:
            if obj["class_name"] == "person":
                people_set.add(obj["track_id"])
            elif obj["class_name"] in ("car", "truck"):
                vehicles_set.add(obj["track_id"])

        # ── Update metrics ──
        latency_ms = (time.time() - loop_start) * 1000
        effective_fps = 1000 / latency_ms if latency_ms > 0 else 0

        metric_people.metric(
            "People (Now / Total)",
            f"{counts['person']} / {len(people_set)}",
        )
        metric_vehicles.metric(
            "Vehicles (Now / Total)",
            f"{counts['car'] + counts['truck']} / {len(vehicles_set)}",
        )

        if len(intruders) > 0:
            metric_status.metric("Security Status", "\U0001F6A8 ZONE BREACH")
        else:
            metric_status.metric("Security Status", "\u2705 ALL CLEAR")

        metric_latency.metric(
            "Latency",
            f"{latency_ms:.0f}ms ({effective_fps:.1f} FPS)",
        )

        # ── Progress / Uptime ──
        if is_live:
            elapsed = time.time() - (st.session_state.live_start_time or time.time())
            hrs, remainder = divmod(int(elapsed), 3600)
            mins, secs = divmod(remainder, 60)
            progress_ph.markdown(
                f"<span class='live-badge'>LIVE</span> "
                f"Uptime: **{hrs:02d}:{mins:02d}:{secs:02d}** | "
                f"Frames: **{frame_idx}** | "
                f"Reconnects: {st.session_state.live_reconnect_count}",
                unsafe_allow_html=True,
            )
        elif total_frames > 0:
            progress_ph.progress(
                min(frame_idx / total_frames, 1.0),
                text=f"Frame {frame_idx} / {total_frames}",
            )

        # ── Update alert log ──
        alerts_list = st.session_state[alerts_key]
        with alert_container:
            if not alerts_list:
                st.markdown(
                    '<div class="alert-nominal">System nominal — no intrusions detected.</div>',
                    unsafe_allow_html=True,
                )
            else:
                for alert in alerts_list[:25]:
                    st.markdown(
                        f'<div class="alert-critical">'
                        f'[{alert["time"]}] {alert["message"]}'
                        f"</div>",
                        unsafe_allow_html=True,
                    )


# ══════════════════════════════════════════════════════════════════════
# Main panel — Header
# ══════════════════════════════════════════════════════════════════════
if st.session_state.live_feed_active:
    st.markdown(
        '<div class="main-header">'
        '<span class="live-badge">LIVE</span>'
        '\U0001F6F0\uFE0F  Aerial Intelligence & Tracking System</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="main-header">\U0001F6F0\uFE0F  Aerial Intelligence & Tracking System</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════
# Two-tab layout
# ══════════════════════════════════════════════════════════════════════
tab_static, tab_live = st.tabs(["\U0001F4C1 Static Analysis", "\U0001F4E1 Live Feed"])

# ──────────────────────────────────────────────────────────────────────
# TAB 1: Static Analysis (upload video)
# ──────────────────────────────────────────────────────────────────────
with tab_static:
    uploaded_file = st.file_uploader(
        "Upload Drone Footage", type=["mp4", "avi", "mov"]
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.uploaded_file_name:
            _init_video(uploaded_file)

    # Metrics row
    s1, s2, s3, s4 = st.columns(4)
    sm_people = s1.empty()
    sm_vehicles = s2.empty()
    sm_status = s3.empty()
    sm_latency = s4.empty()

    sm_people.metric("People (Now / Total)", "0 / 0")
    sm_vehicles.metric("Vehicles (Now / Total)", "0 / 0")
    sm_status.metric("Security Status", "STANDBY")
    sm_latency.metric("Latency", "-- ms")

    # Video feed
    static_video_ph = st.empty()
    static_progress_ph = st.empty()

    # Controls
    sc1, sc2 = st.columns(2)
    with sc1:
        static_start = st.button(
            "\u25B6 START", key="static_start", type="primary", use_container_width=True
        )
    with sc2:
        static_stop = st.button(
            "\u25A0 STOP", key="static_stop", use_container_width=True
        )

    # Alert log
    st.markdown("### \U0001F6A8 Alert Log")
    static_alert_container = st.container(height=220)

    # Prompt
    if st.session_state.video_cap is None and not st.session_state.is_processing:
        static_video_ph.info(
            "\U0001F4F7 Upload drone footage above to begin analysis."
        )

# ──────────────────────────────────────────────────────────────────────
# TAB 2: Live Feed
# ──────────────────────────────────────────────────────────────────────
with tab_live:
    lc1, lc2 = st.columns([2, 1])
    with lc1:
        feed_type = st.selectbox(
            "Connection Type",
            ["RTSP Stream", "USB / Webcam", "HTTP/MJPEG Stream"],
            key="live_feed_type_select",
        )
    with lc2:
        if feed_type == "USB / Webcam":
            device_index = st.number_input(
                "Camera Index", min_value=0, max_value=10, value=0, step=1
            )
        else:
            placeholder = (
                "rtsp://192.168.1.1:554/stream"
                if feed_type == "RTSP Stream"
                else "http://192.168.1.100:8080/stream"
            )
            feed_url = st.text_input("Stream URL", placeholder=placeholder)

    st.caption(
        "Most DJI drones use RTSP. Phone camera apps like DroidCam use USB / Webcam."
    )

    # Metrics row
    l1, l2, l3, l4 = st.columns(4)
    lm_people = l1.empty()
    lm_vehicles = l2.empty()
    lm_status = l3.empty()
    lm_latency = l4.empty()

    lm_people.metric("People (Now / Total)", "0 / 0")
    lm_vehicles.metric("Vehicles (Now / Total)", "0 / 0")
    lm_status.metric("Security Status", "STANDBY")
    lm_latency.metric("Latency", "-- ms")

    # Video feed
    live_video_ph = st.empty()
    live_uptime_ph = st.empty()

    # Controls
    lbc1, lbc2 = st.columns(2)
    with lbc1:
        live_connect = st.button(
            "\u25B6 CONNECT", key="live_connect", type="primary", use_container_width=True
        )
    with lbc2:
        live_disconnect = st.button(
            "\u25A0 DISCONNECT", key="live_disconnect", use_container_width=True
        )

    # Alert log
    st.markdown("### \U0001F6A8 Alert Log")
    live_alert_container = st.container(height=220)

    # Prompt
    if not st.session_state.live_feed_active:
        live_video_ph.info(
            "\U0001F4E1 Configure your feed source above and click CONNECT."
        )


# ══════════════════════════════════════════════════════════════════════
# Processing triggers
# ══════════════════════════════════════════════════════════════════════

# -- Static tab --
if static_stop:
    st.session_state.is_processing = False

if static_start and st.session_state.video_cap is not None:
    st.session_state.is_processing = True

# -- Live tab --
if live_disconnect:
    st.session_state.live_feed_active = False
    if st.session_state.live_video_cap is not None:
        st.session_state.live_video_cap.release()
        st.session_state.live_video_cap = None

if live_connect:
    if feed_type == "USB / Webcam":
        source = int(device_index)
    else:
        source = feed_url.strip() if feed_url else ""
    if isinstance(source, str) and not source:
        st.sidebar.error("Please enter a stream URL.")
    else:
        if _init_live_feed(source):
            st.session_state.live_feed_active = True

# ══════════════════════════════════════════════════════════════════════
# Run active processing loop
# ══════════════════════════════════════════════════════════════════════
if st.session_state.is_processing and st.session_state.video_cap is not None:
    _run_processing_loop(
        cap=st.session_state.video_cap,
        dv=st.session_state.drone_vision,
        video_ph=static_video_ph,
        metric_people=sm_people,
        metric_vehicles=sm_vehicles,
        metric_status=sm_status,
        metric_latency=sm_latency,
        progress_ph=static_progress_ph,
        alert_container=static_alert_container,
        alerts_key="alerts",
        heatmap_key="heatmap_acc",
        people_key="total_people_seen",
        vehicles_key="total_vehicles_seen",
        active_flag="is_processing",
        is_live=False,
    )

if st.session_state.live_feed_active and st.session_state.live_video_cap is not None:
    _run_processing_loop(
        cap=st.session_state.live_video_cap,
        dv=st.session_state.live_drone_vision,
        video_ph=live_video_ph,
        metric_people=lm_people,
        metric_vehicles=lm_vehicles,
        metric_status=lm_status,
        metric_latency=lm_latency,
        progress_ph=live_uptime_ph,
        alert_container=live_alert_container,
        alerts_key="live_alerts",
        heatmap_key="live_heatmap_acc",
        people_key="live_total_people_seen",
        vehicles_key="live_total_vehicles_seen",
        active_flag="live_feed_active",
        is_live=True,
    )
