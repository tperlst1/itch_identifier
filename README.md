
# Itch Identifier

## Overview

Itch Identifier is a computer vision application designed
to identify plant species and alert users to potentially
hazardous vegetation in their surroundings.

The application uses YOLO object detection and OpenCV
video processing to identify plants, assign hazard
classifications, and communicate potential environmental
risks to the user.

## Core Features

- Real-time plant species identification.
- Dangerous and non-dangerous plant classification.
- Species identification labels.
- 95% confidence threshold for confirmed classifications.
- Uncertain labels for unknown or low-confidence detections.
- Dangerous plant counting.
- Unsafe-area warning when more than 10 dangerous plants
  are detected in one frame.
- Proximity alert when a dangerous plant is less than
  one foot away.
- Live video capture using OpenCV.
- Pause and resume functionality.

## Repository Structure

```text
itch-identifier/
├── models/
├── dataset/
├── src/
│   ├── __init__.py
│   └── itch_identifier.py
├── tests/
│   ├── __init__.py
│   └── test_itch_identifier.py
├── train.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Technologies

- Python
- OpenCV
- Ultralytics YOLO
- PyTorch
- NumPy
- Pytest
- Roboflow

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/itch-identifier.git

cd itch-identifier
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The project uses the following Roboflow dataset:

https://universe.roboflow.com/ai-model-v2/object-detection-model-for-identification-of-poisonous-plants-in-the-chesapeake-bay-watershed-p8o4b

Download the dataset in YOLO format.

Place the exported dataset in the dataset directory.

Ensure that the dataset configuration is located at:

```text
dataset/data.yaml
```

Verify that the class names in data.yaml match the
species names in the classification dictionary.

## Model Training

Run:

```bash
python train.py
```

The trained model will be saved under:

```text
runs/itch_identifier/species_detection/weights/best.pt
```

Copy best.pt into:

```text
models/best.pt
```

## Launching the Application

Run:

```bash
python main.py
```

The default video input is the computer's webcam.

To use a prerecorded video, change:

```python
video_source = 0
```

to:

```python
video_source = "path/to/video.mp4"
```

## Keyboard Controls

| Key | Function |
|-----|----------|
| P | Pause/resume |
| Q | Quit |

## Classification System

| Label | Description |
|-------|-------------|
| Dangerous | Manually configured hazardous species |
| Non-dangerous | Manually configured non-hazardous species |
| Uncertain | Unknown species or confidence below 95% |

Unknown plants are never automatically assumed safe.

## Area Safety Monitoring

Itch Identifier counts dangerous plant detections
in each processed frame.

When more than 10 dangerous plants are detected,
the application displays an unsafe-area warning.

## Proximity Monitoring

The application supports a calibrated distance estimator.

When a dangerous plant is detected less than
0.3048 meters (one foot) away, the application
displays a warning to move away.

A calibrated depth camera or equivalent validated
distance-measurement system is required.

## Testing

Run the unit tests:

```bash
pytest tests/ -v
```

## Frontend Integration

The frontend is implemented in `src/frontend.py` and integrates
with the existing ItchIdentifier backend.

### Frontend Functions

| Function | Description |
|---|---|
| `initialize_interface()` | Initializes frontend display data. |
| `set_status()` | Generates user safety messages. |
| `count_plants()` | Counts dangerous plants in each frame. |
| `mark_up()` | Draws species labels and color-coded bounding boxes. |

### Display Classifications

| Classification | Color |
|---|---|
| Dangerous | Red |
| Non-dangerous | Green |
| Uncertain | Orange |

### Safety Alerts

- More than 10 dangerous plants:
  UNSAFE AREA - DO NOT PROCEED.

- Dangerous plant within one foot:
  DANGER: TOO CLOSE - MOVE AWAY.

- Uncertain plant identification:
  CAUTION: UNIDENTIFIED PLANTS DETECTED.

### Running the Application

```bash
python main.py
```

### Running Tests

```bash
pytest tests/ -v
```

### Frontend Controls

| Key | Function |
|---|---|
| P | Pause/resume video |
| Q | Quit application |


## Limitations

- Accuracy depends on training data quality.
- Only configured plant species receive confirmed
  hazard classifications.
- Confidence scores require calibration.
- Distance estimation requires calibrated hardware.
- Plant counts are per-frame detections, not unique
  tracked plants.
- The system cannot guarantee that an area is safe.

Users should avoid approaching unidentified or
potentially hazardous plants regardless of the
application's output.

## License

Add an appropriate license before distributing this
project. Verify the dataset's licensing terms separately.
