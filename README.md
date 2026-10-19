# 🚦 Traffic Analysis System

A computer-vision-based traffic analysis system that processes traffic video to detect, classify, track, and analyze vehicles.

The project uses **YOLOv8** for vehicle detection and **ByteTrack** for multi-object tracking. The system is being developed as a 30–45 day two-member project, with the goal of building a practical and explainable traffic-monitoring system rather than simply combining many independent features.

---

## 📌 Project Overview

Traffic monitoring traditionally requires continuous human observation or expensive dedicated infrastructure.

This project explores how computer vision can automatically extract useful traffic information from ordinary video footage.

The system processes traffic video and aims to provide:

* Vehicle detection
* Vehicle classification
* Vehicle tracking
* Vehicle counting
* Traffic-density estimation
* Wrong-way vehicle detection
* Helmet-violation detection
* Traffic analytics through a dashboard

The project is being developed incrementally, starting from an existing YOLO + ByteTrack traffic-analysis prototype and improving its architecture, reliability, and functionality.

---

# 🎯 Project Objectives

The main objectives are:

1. Detect vehicles from traffic video.
2. Classify detected vehicles into different categories.
3. Track vehicles across video frames using persistent IDs.
4. Count vehicles crossing a predefined region or line.
5. Estimate traffic density.
6. Detect vehicles moving in the wrong direction.
7. Detect motorcycle riders without helmets.
8. Present the collected traffic information through an analytics dashboard.
9. Produce structured traffic statistics and violation events.
10. Build a system that is understandable, testable, and suitable for further development.

---

# 🚀 Features

## 1. 🚗 Vehicle Detection & Classification

The system uses YOLOv8 to detect vehicles in each video frame.

Currently supported vehicle categories:

* Car
* Bike
* Bus
* Truck

For each detection, the system can obtain information such as:

* Bounding box
* Vehicle class
* Detection confidence

### Pipeline

```text
Traffic Frame
     ↓
YOLOv8
     ↓
Vehicle Detection
     ↓
Classification
     ↓
Car / Bike / Bus / Truck
```

---

## 2. 🆔 Vehicle Tracking

Detected vehicles are tracked across consecutive frames using **ByteTrack**.

Tracking provides a persistent ID for each vehicle.

For example:

```text
Frame 100 → Car → ID 17
Frame 101 → Car → ID 17
Frame 102 → Car → ID 17
Frame 103 → Car → ID 17
```

This allows the system to understand vehicle movement rather than treating every detection as a new vehicle.

---

## 3. 🔢 Vehicle Counting

Vehicles can be counted when they cross a predefined counting line or region.

The tracking ID is used to prevent the same vehicle from being counted repeatedly.

Example:

```text
             TRAFFIC FLOW
                  ↓
                  ↓
        ─────────────────────
             COUNT LINE
        ─────────────────────
                  ↓
                  ↓

          Vehicles counted
```

The system maintains counts for different vehicle categories.

Example:

```text
Cars:    82
Bikes:   51
Buses:   12
Trucks:  12

Total: 157
```

The default count line is at `y=500` for `india_trafic.mp4`; set the
`COUNT_LINE_Y` environment variable to calibrate it for another camera/video.
For example, `COUNT_LINE_Y=330 HEADLESS=1 ./venv/bin/python Terr_v2.py`.

The `total` and per-class values in `stats.json` are **line-crossing event
counts**, not unique-vehicle counts. `unique_vehicle_count` is a separate,
experimental estimate based on tracker IDs, 2x2 spatial HSV crop histograms,
class compatibility, and a pixel-motion gate. Treat it as an estimate, not
ground-truth identity.

When ReID is enabled, the video overlay shows both the raw tracker ID (`ID`)
and estimated canonical ID (`C`). `reid_frame_log.csv` records each tracked
observation's frame, class, bounding box, centroid, canonical mapping, match
similarity, decision reason, and whether the raw track crossed the configured
line. A manual validation should inspect the source video around each
`matched_existing=True` row and confirm the same physical vehicle; color
histograms alone cannot establish identity.

In the 750-frame `india_trafic.mp4` validation, the previous `y=330` line had
zero tracked-centroid crossings: all logged centroids were between `y=355`
and `y=627`, placing that line above the traffic. At `y=500`, the same recorded
trajectories crossed 14 times (2 Car, 10 Bike, 2 Truck), and the final pipeline
run saved the corresponding 14 `vehicle_crossing` events. A visually confirmed
false Bike handoff at frames 136→179 scored 0.807 using the old whole-crop
histogram; the spatial histogram scored 0.448 and rejected it under the current
0.70 threshold. This threshold is conservative and remains camera-specific;
the estimate still requires manual review.

To run a bounded headless sample with the project environment:

```bash
HEADLESS=1 MAX_FRAMES=180 ./venv/bin/python Terr_v2.py
```

Each pipeline execution writes its outputs to a new unique directory under
`outputs/runs/`, including `stats.json`, `events.json`, `frame_log.csv`, and
`reid_frame_log.csv`. A completed run is added to `outputs/run_manifest.json`
only after its final statistics and events have been written. Incomplete runs
remain unlisted and do not replace the latest completed run.

The dashboard uses the latest completed run by default and provides a selector
for earlier runs. To inspect an existing results directory that is not in the
run manifest, set `TRAFFIC_RESULTS_DIR` to that directory when starting the
dashboard; a relative path is resolved from the repository root. This explicit
override bypasses run history and reads both `stats.json` and `events.json`
from that one directory. Existing files such as
`live_test_20261019/outputs/` are preserved but are not automatically added to
the history or represented as newly processed runs.

---

## 4. 🛣️ Traffic Density Estimation

Traffic density is estimated from the traffic conditions observed in the monitored region.

The initial system uses vehicle counts and configurable thresholds to classify traffic as:

```text
LOW
MEDIUM
HIGH
```

Example:

```text
Vehicles
   ↓
Traffic Analysis
   ↓
┌──────────────┐
│ LOW          │
│ MEDIUM       │
│ HIGH         │
└──────────────┘
```

The density calculation can be improved later using additional information such as road-region occupancy and time-based traffic measurements.

---

## 5. 🚨 Wrong-Way Detection

The system analyzes vehicle trajectories to determine whether a vehicle is moving in the expected traffic direction.

Conceptually:

```text
Vehicle Track
      ↓
Position History
      ↓
Movement Direction
      ↓
Compare With Expected Direction
      ↓
┌───────────────┬───────────────┐
│ Normal        │ Wrong-way 🚨  │
└───────────────┴───────────────┘
```

Persistent tracking and multiple observations are used to reduce false detections caused by temporary tracking noise.

---

## 6. 🪖 Helmet Violation Detection

Helmet detection is part of the project's planned core functionality.

The intended pipeline is:

```text
Traffic Frame
      ↓
Vehicle Detection
      ↓
Motorcycle/Bike
      ↓
Rider Region
      ↓
Helmet Detection
      ↓
┌──────────────┬────────────────┐
│ Helmet       │ No Helmet 🚨   │
└──────────────┴────────────────┘
```

The helmet detector will be integrated into the existing traffic-analysis pipeline rather than processing the entire video independently.

---

# 📊 Traffic Analytics Dashboard

The final system will provide a dashboard for viewing traffic statistics and detected events.

The dashboard is planned to display information such as:

```text
┌─────────────────────────────────────────────┐
│           TRAFFIC ANALYSIS SYSTEM           │
├──────────┬──────────┬──────────┬───────────┤
│ Vehicles │ Cars     │ Bikes    │ Density   │
│   157    │   82     │   51     │ HIGH      │
├──────────┴──────────┴──────────┴───────────┤
│                                             │
│              LIVE TRAFFIC VIDEO             │
│                                             │
├──────────────────────┬──────────────────────┤
│ Vehicle Statistics   │ Violations           │
│                      │                      │
│ Car       82         │ Wrong-way       3    │
│ Bike      51         │ No Helmet       7    │
│ Bus       12         │                      │
│ Truck     12         │                      │
└──────────────────────┴──────────────────────┘
```

Planned dashboard information includes:

* Total vehicles
* Vehicle-type counts
* Traffic density
* Wrong-way events
* Helmet violations
* Traffic statistics over time

---

# 🧠 System Architecture

The overall target architecture is:

```text
                       TRAFFIC VIDEO
                             │
                             ▼
                  ┌────────────────────┐
                  │ Vehicle Detection  │
                  │      YOLOv8        │
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │      ByteTrack     │
                  │      Tracking      │
                  └─────────┬──────────┘
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
         Counting       Direction      Trajectory
             │              │              │
             ▼              ▼              │
         Density       Wrong-way            │
                                           │
                                           ▼
                                  Helmet Detection
                                           │
                                           ▼
                                  Traffic Analytics
                                           │
                         ┌─────────────────┴─────────────────┐
                         │                                   │
                         ▼                                   ▼
                    CSV / JSON                         Dashboard
```

---

# 🛠️ Technology Stack

## Computer Vision

* Python
* OpenCV
* Ultralytics YOLO
* YOLOv8
* ByteTrack

## Data & Analytics

* Python data structures
* CSV
* JSON

## Dashboard

* Streamlit

## Development

* Git
* GitHub
* Python virtual environment

---

# 📁 Project Structure

The project is being migrated from the original prototype toward the following structure:

```text
traffic-analysis-system/
│
├── app/
│   ├── main.py
│   ├── config.py
│   │
│   ├── detection/
│   │   ├── vehicle_detector.py
│   │   └── helmet_detector.py
│   │
│   ├── tracking/
│   │   └── tracker.py
│   │
│   ├── analytics/
│   │   ├── counter.py
│   │   ├── density.py
│   │   └── wrong_way.py
│   │
│   ├── visualization/
│   │   └── renderer.py
│   │
│   └── storage/
│       └── event_logger.py
│
├── dashboard/
│   └── app.py
│
├── configs/
│   └── config.yaml
│
├── models/
│   ├── vehicle/
│   └── helmet/
│
├── data/
│   └── videos/
│
├── outputs/
│   ├── videos/
│   ├── events/
│   └── statistics/
│
├── tests/
│   ├── test_counter.py
│   ├── test_density.py
│   └── test_wrong_way.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

> **Note:** The repository is currently being refactored from the original prototype structure. Some of the directories shown above are part of the target architecture and will be introduced progressively.

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd traffic-analysis-system
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Required Model

The vehicle-detection pipeline uses a YOLO model compatible with the Ultralytics framework.

Place the required model inside the appropriate model directory:

```text
models/
└── vehicle/
    └── <vehicle-model>.pt
```

The exact model path will be configured through the project configuration file as the architecture is finalized.

---

# 🎥 Input Video

Place traffic video files inside:

```text
data/videos/
```

Example:

```text
data/
└── videos/
    └── input.mp4
```

The video source should be configurable rather than hard-coded into the application.

---

# ▶️ Running the Project

The current prototype can be run through the existing implementation.

As the refactoring progresses, the main application will use:

```bash
python -m app.main
```

The dashboard will eventually be launched using:

```bash
streamlit run dashboard/app.py
```

These commands will become the official execution commands once the new architecture is fully integrated.

---

# 📈 Output

The system can generate structured traffic information such as:

```text
Vehicle Type     Count
----------------------
Car               82
Bike              51
Bus               12
Truck             12
----------------------
Total             157
```

The system can also record traffic events and statistics in machine-readable formats such as:

```text
outputs/
├── events/
└── statistics/
```

Planned event information includes:

* Event type
* Vehicle ID
* Vehicle class
* Frame number
* Timestamp
* Detection information

---

# 🧪 Testing

Testing will be added progressively as the project is refactored.

Important components to test include:

* Vehicle counting
* Line crossing
* Density classification
* Direction detection
* Wrong-way detection
* Event generation

Example:

```bash
pytest
```

The goal is to test the analytical logic independently from the YOLO model and video-processing pipeline wherever possible.

---

# 🗓️ Development Roadmap

The project is planned as a **30–45 day development and learning project**.

## Phase 1 — Foundation

* Understand the existing implementation
* Establish baseline behavior
* Refactor configuration
* Separate detection and tracking
* Preserve existing functionality

## Phase 2 — Vehicle Detection & Classification

* YOLO vehicle detection
* Car/Bike/Bus/Truck classification
* Detection evaluation
* Performance measurement

## Phase 3 — Tracking & Counting

* ByteTrack integration
* Persistent vehicle IDs
* Counting-line logic
* Direction-aware counting

## Phase 4 — Traffic Density

* Vehicle-based density estimation
* Configurable thresholds
* Time-based traffic statistics

## Phase 5 — Wrong-Way Detection

* Track trajectory analysis
* Expected traffic direction
* Wrong-way event detection
* False-positive reduction

## Phase 6 — Helmet Detection

* Select and evaluate helmet-detection model
* Detect helmet/no-helmet cases
* Integrate helmet detection with tracked motorcycles
* Generate helmet-violation events

## Phase 7 — Analytics & Dashboard

* Centralized traffic statistics
* Event logging
* Streamlit dashboard
* Charts and traffic summaries

## Phase 8 — Testing & Documentation

* Integration testing
* Performance testing
* Accuracy evaluation
* README updates
* Project documentation
* Presentation and demonstration

---

# 👥 Team Development

This is a two-member project.

### Member 1 — Computer Vision / AI

Primary responsibilities:

* YOLO vehicle detection
* Vehicle classification
* ByteTrack
* Tracking
* Vehicle trajectories
* Wrong-way detection
* Helmet detection
* Computer-vision evaluation

### Member 2 — Analytics / Application

Primary responsibilities:

* Vehicle counting
* Traffic-density estimation
* Analytics engine
* Event logging
* CSV/JSON statistics
* Streamlit dashboard
* Visualization

### Shared Responsibilities

Both members will work together on:

* Architecture
* Integration
* Testing
* Debugging
* Documentation
* Presentation
* Project understanding

Both team members should understand the complete end-to-end pipeline.

---

# 🚫 Current Scope

To keep the project achievable within the planned development period, the following features are **not part of the current core scope**:

* Number-plate detection
* OCR / number-plate recognition
* Red-light violation detection
* Automatic traffic-light recognition
* Cloud deployment
* Large-scale production infrastructure

These may be considered later as optional extensions if the core system is completed successfully.

---

# 🔮 Future Improvements

Possible future extensions include:

* Lane-level traffic analysis
* Vehicle speed estimation
* Automatic traffic-light recognition
* Number-plate recognition
* OCR
* Accident detection
* Traffic-flow prediction
* Historical traffic analytics
* Database integration
* Cloud deployment
* Real-time camera/RTSP support

These are intentionally outside the initial 30–45 day scope.

---

# 🎓 Learning Goals

This project is also intended as a practical learning experience.

The main concepts covered are:

### Computer Vision

* Image and video processing
* OpenCV
* Object detection
* Bounding boxes
* Confidence scores
* Tracking
* Trajectories

### Deep Learning

* YOLO
* Object detection
* Model inference
* Model evaluation
* Multiple detection models

### Algorithms

* Line-crossing detection
* Vehicle counting
* Direction estimation
* Wrong-way detection
* Density estimation

### Software Engineering

* Python project structure
* Modular architecture
* Configuration management
* Testing
* Logging
* Git/GitHub
* Team development

### Application Development

* Data processing
* Analytics
* Visualization
* Streamlit dashboards

---

# ⚠️ Project Status

**Status: In Development 🚧**

### Currently available

* ✅ YOLO-based vehicle detection
* ✅ Vehicle classification
* ✅ ByteTrack vehicle tracking
* ✅ Vehicle counting
* ✅ Basic traffic-density estimation
* ✅ Basic wrong-way detection
* ✅ CSV/JSON traffic statistics
* ✅ OpenCV visualization

### Being improved / developed

* 🔄 Modular project architecture
* 🔄 More robust vehicle counting
* 🔄 Improved direction and wrong-way detection
* 🔄 Helmet violation detection
* 🔄 Centralized analytics
* 🔄 Traffic analytics dashboard
* 🔄 Automated testing

---

# 📄 Project Philosophy

The objective is not to build the largest possible traffic-monitoring system.

Instead, the project focuses on:

> **Fewer features, deeper understanding, reliable implementation, and clear explanation.**

Every major component should be understandable by the development team and explainable during a technical presentation or interview.

---

## License

This project is developed for educational and academic purposes.

Please review the license and attribution requirements of the original project and any third-party models, datasets, libraries, or code used as part of this system.
