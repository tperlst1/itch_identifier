
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
- Optional calibrated distance estimation.
- Proximity alert when a dangerous plant is less than
  one foot away, provided valid calibration is available.
- Live video capture using OpenCV.
- Pause and resume functionality.

## Repository Structure

```text
itch_identifier/
├── dataset/
│   └── data.yaml
│
├── models/
│   ├── best.pt
│   └── ground_calibration.npz
│
├── src/
│   ├── __init__.py
│   ├── itch_identifier.py
│   ├── frontend.py
│   └── distance_estimator.py
│
├── tests/
│   ├── __init__.py
│   └── test_itch_identifier.py
│
├── runs/
│
├── train.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

The model and calibration files are optional repository
artifacts and may be stored locally rather than committed.

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
git clone https://github.com/tperlst1/itch_identifier.git

cd itch_identifier
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
python -m pip install -r requirements.txt
```

## Dataset

The project uses the following Roboflow dataset:

https://universe.roboflow.com/ai-model-v2/object-detection-model-for-identification-of-poisonous-plants-in-the-chesapeake-bay-watershed-p8o4b

Download the dataset in YOLO format.

Place the exported dataset in the location expected by
`dataset/data.yaml`.

Ensure the configuration is available at:

```text
dataset/data.yaml
```

The existing configuration defines four classes:

| ID | Species |
|---|---|
| 0 | Giant Hogweed |
| 1 | Mayapple |
| 2 | Poison Hemlock |
| 3 | Spotted Water Hemlock |

Preserve the original class ordering.

The dataset uses class names containing common names
and scientific-name suffixes.

The backend removes the scientific-name suffix before
performing its classification lookup.

Do not change class IDs without retraining the model.

## Model Training

Run:

```bash
python train.py
```

Optional arguments:

```bash
python train.py --epochs 100 --batch 16 --device cpu
```

For a compatible CUDA GPU:

```bash
python train.py --epochs 100 --batch 16 --device 0
```

The training process uses:

```text
dataset/data.yaml
```

Training outputs are stored under:

```text
runs/itch_identifier/species_detection/weights/
```

After successful training, the best model is automatically
copied to:

```text
models/best.pt
```

The application uses this file for inference.

## Launching the Application

Run:

```bash
python main.py
```

The default inference device is CPU.

To use a compatible CUDA GPU:

```bash
ITCH_DEVICE=0 python main.py
```

The default video input is the computer's webcam.

To use another camera:

```bash
ITCH_CAMERA=1 python main.py
```

To use a prerecorded video:

```bash
ITCH_CAMERA=videos/sample.mp4 python main.py
```

## Keyboard Controls

| Key | Function |
|---|---|
| P | Pause/resume |
| Q | Quit application |

## Classification System

| Label | Description |
|---|---|
| Dangerous | Species configured as hazardous |
| Non-dangerous | Species explicitly configured as non-hazardous |
| Uncertain | Unknown species or confidence below 95% |

Unknown plants are never automatically assumed safe.

The existing species classification dictionary is
maintained in:

```text
src/itch_identifier.py
```

Only species explicitly configured in that dictionary
receive confirmed hazard classifications.

## Area Safety Monitoring

Itch Identifier counts dangerous plant detections
in each processed frame.

When more than 10 dangerous plants are detected,
the application displays:

```text
UNSAFE AREA - DO NOT PROCEED
```

The threshold is configured in the backend:

```python
DANGEROUS_THRESHOLD = 10
```

The frontend receives the same threshold through
the interface dictionary.

## Proximity Monitoring

The application supports optional calibrated distance
estimation through:

```text
src/distance_estimator.py
```

The calibration file is:

```text
models/ground_calibration.npz
```

The calibration matrix must map image coordinates
to ground-plane coordinates expressed in meters.

To enable distance estimation:

```bash
ITCH_ENABLE_DISTANCE=1 python main.py
```

When a dangerous plant is estimated to be less than
0.3048 meters (one foot) away, the application displays:

```text
DANGER: TOO CLOSE - MOVE AWAY
```

Without valid calibration, plant detection continues
but distance-based proximity alerts remain disabled.

A ground-plane homography does not provide general
three-dimensional distance measurements.

Calibration accuracy depends on camera placement,
the ground plane, and the calibration reference points.

## Frontend Integration

The frontend is implemented in:

```text
src/frontend.py
```

It integrates with the backend through these functions:

| Function | Description |
|---|---|
| `initialize_interface()` | Initializes frontend display data |
| `count_plants()` | Counts dangerous plants |
| `set_status()` | Generates user safety messages |
| `mark_up()` | Draws labels and bounding boxes |

The backend provides:

- Video frame
- Plant detections
- Dangerous plant count
- Unsafe-area status
- Proximity alert data
- Distance measurements

The frontend produces:

- Annotated video frame
- Species labels
- Confidence scores
- Distance labels
- Dangerous plant count
- Safety status

### Display Classifications

| Classification | Color |
|---|---|
| Dangerous | Red |
| Non-dangerous | Green |
| Uncertain | Orange |

### Safety Alerts

- More than 10 dangerous plants:
  `UNSAFE AREA - DO NOT PROCEED`

- Dangerous plant within one foot:
  `DANGER: TOO CLOSE - MOVE AWAY`

- Uncertain plant identification:
  `CAUTION: UNIDENTIFIED PLANTS DETECTED`

## Testing

Run the unit tests:

```bash
pytest tests/ -v
```

Check Python syntax:

```bash
python -m compileall -q src main.py train.py tests
```

The tests cover:

- Species-name normalization.
- Hazard classification.
- Dangerous plant counting.
- Area safety thresholds.
- Optional distance estimation.
- Proximity alerts.
- Calibration validation.
- Frontend/backend integration.

## Limitations

- Accuracy depends on training data quality.
- Only configured plant species receive confirmed
  hazard classifications.
- Confidence scores require calibration.
- Distance estimation requires validated calibration.
- Plant counts are per-frame detections, not unique
  tracked plants.
- The system cannot guarantee that an area is safe.

Users should avoid approaching unidentified or
potentially hazardous plants regardless of the
application's output.

## License

Add an appropriate project license before distributing
this application.

Verify the Roboflow dataset licensing terms separately.
