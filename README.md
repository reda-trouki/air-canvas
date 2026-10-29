# Air Canvas — Real-Time Hand Gesture Drawing

Draw in the air using your webcam and hand tracking. The application detects hand landmarks, interprets simple finger gestures, and overlays persistent strokes on the live camera feed.

## Features

- Real-time hand landmark tracking using MediaPipe.
- Index finger only: draw.
- Index + middle fingers: hover and select; never draw.
- Hold the two-finger gesture over a toolbar color or eraser briefly to select it (no precise pinch required).
- Color palette, eraser, brush-size controls, clear canvas, save drawing.
- Mirrored webcam view, smoothed pointer, FPS indicator, and keyboard shortcuts.
- Local processing; the app does not save or upload camera frames. Saving a drawing is only triggered with the S key.

## Requirements

- Python 3.10 or 3.11 recommended.
- Webcam and a desktop environment with OpenCV GUI support.
- Windows, macOS, or Linux.

## Setup

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

If PowerShell blocks activation, use:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

## Gestures and controls

| Action | Gesture / key |
|---|---|
| Draw | Raise index finger only |
| Stop drawing / hover | Raise index and middle fingers |
| Select a color / eraser | Keep index and middle fingers raised over a toolbar tile for about 0.35 seconds |
| Pause | Lower index finger / show another gesture |
| Clear canvas | `C` |
| Save drawing as PNG | `S` |
| Increase brush size | `+` or `=` |
| Decrease brush size | `-` |
| Quit | `Q` or `Esc` |

**Gesture tip:** Keep your palm facing the camera and fingers separated clearly. To select a toolbar tile, point at it with the index fingertip, raise the middle finger, and hold still for a moment. Return to one raised finger to draw.

## Run options

```bash
python main.py --camera 1
python main.py --width 960 --height 540
python main.py --max-hands 2
```

## Architecture

- `main.py`: camera loop, landmark processing, gesture state, drawing layer, palette UI, keyboard input.
- MediaPipe Hands: 21 hand landmarks per detected hand.
- OpenCV: webcam capture, drawing overlays, compositing, display, and PNG export.
- NumPy: smoothing and canvas operations.

## Troubleshooting

- **Camera not found:** close apps using the webcam, check OS camera permissions, or try `--camera 1`.
- **OpenCV `imshow` error:** ensure `opencv-python` is installed, not `opencv-python-headless`. In the virtual environment, uninstall all OpenCV variants and reinstall `opencv-python`.
- **MediaPipe install error:** use Python 3.10 or 3.11 and create a fresh virtual environment.
- **Tracking is jittery:** improve lighting, keep the hand within the frame, and avoid a busy background.
- **Selection does not trigger:** aim at the center of a toolbar tile, keep index and middle fingers raised, and pinch thumb to index. The toolbar is at the top of the frame.

## Resume description

**Air Canvas — Real-Time Hand Gesture Drawing** — Built a Python computer-vision application using OpenCV, MediaPipe, and NumPy to track 21 hand landmarks, interpret finger gestures, and enable touchless drawing with color selection, brush controls, erasing, and PNG export.

## Limitations

This is a gesture-based computer-vision demo. Landmark and gesture recognition may be less reliable with poor lighting, occlusion, or unusual hand orientations.
