# Drone Surveillance & Tracking System (YOLOv8 + DeepSORT)

**Real-time Aerial Intelligence & Tracking System**

Turn any drone into a smart surveillance platform. SkyWatch AI detects, tracks, and analyzes people, vehicles, and hazards from drone footage — with zone intrusion alerts, movement trails, heatmaps, and target lock-on — all through a dark-themed Command Center dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---
## Upload :

<img width="1919" height="889" alt="image" src="https://github.com/user-attachments/assets/9ec9f4db-7dd7-44f3-8ee7-c98c1068a977" />
<img width="1904" height="887" alt="image" src="https://github.com/user-attachments/assets/5cea0a06-cca3-4862-aa25-644cfc3fd0b3" />

---
## Live :

<img width="1908" height="887" alt="image" src="https://github.com/user-attachments/assets/3cd079a4-e651-49da-94da-0063190f9cef" />
<img width="1898" height="838" alt="image" src="https://github.com/user-attachments/assets/803665fc-b661-41b6-99e6-eb2f5b53753c" />

---
## Features

### Detection & Tracking
- **YOLOv8** object detection — identifies people, cars, and trucks in real time
- **DeepSORT** multi-object tracking with MobileNet re-identification — objects keep their unique ID even after occlusion
- **Velocity estimation** — pixel-displacement-based speed calculation for every tracked object

### Spatial Intelligence
- **Zone Intrusion Detection** — define a monitored zone polygon; triggers alerts when objects enter the restricted area using `cv2.pointPolygonTest`
- **Movement Trails (Breadcrumbs)** — last 30 positions per tracked object drawn as polylines, showing intent and direction
- **Heatmap Overlay** — cumulative activity heatmap with Gaussian blur and JET colormap

### Target Lock-On
- Select any tracked object by ID from the sidebar
- Locked target gets **cyan highlight with corner brackets** and a dedicated on-frame info panel showing class, speed, coordinates, and time tracked
- Auto-clears when target leaves the frame

### Dual-Mode Operation
| Static Analysis | Live Feed |
|----------------|-----------|
| Upload MP4/AVI/MOV drone footage | Connect to RTSP, USB/Webcam, or HTTP/MJPEG streams |
| Frame-by-frame processing with progress bar | Real-time processing with uptime counter |
| Post-incident review | Live surveillance monitoring |
| | Auto-reconnection on connection drop (5 retries) |

### Command Center Dashboard
- Dark monospace theme optimized for long monitoring sessions
- 4-column metrics row: People count, Vehicle count, Security Status, Latency/FPS
- Pulsing red alert log for zone intrusion events
- LIVE badge indicator during active drone feed

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | [YOLOv8](https://docs.ultralytics.com/) (Ultralytics) |
| Object Tracking | [DeepSORT](https://github.com/levan92/deep_sort_realtime) with MobileNet embedder |
| Dashboard | [Streamlit](https://streamlit.io/) |
| Computer Vision | [OpenCV](https://opencv.org/) |
| Numerical Processing | [NumPy](https://numpy.org/) |

---

## Project Structure

```
skywatch-ai/
├── .streamlit/
│   └── config.toml          # Dark theme + upload size config
├── app.py                    # Streamlit dashboard (two-tab layout)
├── detection.py              # DroneVision class (YOLO + DeepSORT engine)
├── utils.py                  # Drawing: boxes, trails, zones, heatmap, lock-on
├── drone_controller.py       # Abstract drone follow interface (future)
├── requirements.txt          # Dependencies
├── assets/                   # Place your drone footage here
└── README.md
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/skywatch-ai.git
cd skywatch-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On first run, YOLOv8 will auto-download the `yolov8n.pt` model (~6MB) and DeepSORT will download MobileNetV2 weights. Internet is required for the first launch.

> **GPU vs CPU:** The system auto-detects GPU availability. If no CUDA GPU is found, it falls back to CPU mode. For CPU-only PyTorch (smaller install):
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install ultralytics deep-sort-realtime opencv-python streamlit numpy
> ```

### 3. Run the dashboard

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Usage

### Static Analysis (Uploaded Video)

1. Open the **Static Analysis** tab
2. Upload an MP4/AVI/MOV file using the file uploader
3. Adjust **Confidence** and **IOU** thresholds in the sidebar
4. Click **START** to begin processing
5. Watch real-time detection, tracking, and alerts

### Live Feed (Drone/Camera)

1. Open the **Live Feed** tab
2. Select your connection type:
   - **RTSP Stream** — enter URL (e.g., `rtsp://192.168.1.1:554/stream`)
   - **USB / Webcam** — select device index (0 for default camera)
   - **HTTP/MJPEG** — enter URL (e.g., `http://192.168.1.100:8080/stream`)
3. Click **CONNECT** to start the live feed
4. Detection and tracking begin immediately

### Target Lock-On

1. While processing is active (either mode), tracked IDs appear in the **Target Lock-On** dropdown in the sidebar
2. Select an ID to lock onto that target
3. The locked target gets a cyan highlight with an info panel overlay
4. Select "-- None --" to release the lock

### Zone Intrusion

1. Enable **Zone Monitoring** in the sidebar
2. Adjust the zone position and size using the percentage sliders
3. When a tracked object enters the zone:
   - The zone turns **red**
   - Security Status shows **ZONE BREACH**
   - An alert is logged with timestamp, class, ID, and velocity

---

## Configuration

### Detection Settings (Sidebar)

| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| Confidence Threshold | 0.10 - 1.00 | 0.40 | Minimum detection confidence |
| IOU Threshold | 0.10 - 1.00 | 0.50 | Non-max suppression overlap threshold |

### Zone Settings (Sidebar)

| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| Zone X (%) | 0 - 80 | 30 | Horizontal position of zone |
| Zone Y (%) | 0 - 80 | 30 | Vertical position of zone |
| Zone Width (%) | 10 - 70 | 40 | Zone width as percentage of frame |
| Zone Height (%) | 10 - 70 | 40 | Zone height as percentage of frame |

### Theme

Edit `.streamlit/config.toml` to customize colors:

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#1A1D23"
textColor = "#FAFAFA"
font = "monospace"
```

---

## Architecture

```
Video Source (MP4 / RTSP / USB / HTTP)
    |
    v
cv2.VideoCapture  -->  frame (BGR numpy array)
    |
    v
DroneVision.detect_and_track(frame)
    |-- YOLOv8 inference --> filter by class --> [x1,y1,w,h] + conf
    |-- DeepSort.update_tracks(detections, frame=frame)
    |-- Confirmed tracks --> bbox, class, centroid, trail, velocity
    |-- Returns list[dict] per tracked object
    |
    v
Zone intrusion check (cv2.pointPolygonTest)  -->  alerts
    |
    v
Drawing pipeline:
    draw_zone() --> draw_boxes() --> draw_trails() --> generate_heatmap()
    draw_locked_target_info()  (if target locked)
    |
    v
BGR --> RGB --> st.empty().image()  -->  Dashboard display
```

---

## Drone Follow Interface (Future)

The `drone_controller.py` module provides an abstract `DroneController` class designed for future physical drone-follow integration:

```python
from drone_controller import SimulatedDroneController, FollowState

controller = SimulatedDroneController()
controller.connect("serial:///dev/ttyUSB0")

# Vision system computes target offset from frame center
state = FollowState(
    target_offset_x=150,    # pixels right of center
    target_offset_y=-30,    # pixels above center
    frame_width=1280,
    frame_height=720,
    target_velocity_px_s=45.0,
    target_class="person",
    target_bbox=(400, 200, 500, 450),
)

gimbal_cmd, move_cmd = controller.compute_follow_commands(state)
controller.send_gimbal_command(gimbal_cmd)
controller.send_movement_command(move_cmd)
```

**Planned SDK support:**
- DJI Mobile SDK (Phantom, Mavic, Matrice series)
- MAVLink (ArduPilot / PX4)
- DJI Tello (UDP control)

---

## Common Drone RTSP URLs

| Drone | RTSP URL |
|-------|----------|
| DJI Phantom 4 | `rtsp://192.168.1.1:554/stream1` |
| DJI Mavic | `rtsp://192.168.1.1:554/` |
| Parrot Bebop | `rtsp://192.168.42.1/media/stream2` |
| Generic IP Camera | `rtsp://admin:password@192.168.1.100:554/stream` |

---

## Requirements

- Python 3.10+
- 4GB+ RAM recommended
- GPU optional (CUDA) — falls back to CPU automatically
- Webcam or drone feed for live mode

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [deep-sort-realtime](https://github.com/levan92/deep_sort_realtime) for the DeepSORT implementation
- [Streamlit](https://streamlit.io/) for the dashboard framework
- COCO dataset for pre-trained object classes

---

**Built with purpose. Designed for surveillance. Ready for deployment.**
