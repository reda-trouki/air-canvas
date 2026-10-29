from __future__ import annotations

import argparse
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


# ---------------------------- Configuration ---------------------------------
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
PALETTE = [
    ("RED", (40, 50, 235)),
    ("ORANGE", (0, 140, 255)),
    ("YELLOW", (0, 220, 255)),
    ("GREEN", (60, 190, 60)),
    ("CYAN", (220, 210, 30)),
    ("BLUE", (235, 100, 40)),
    ("PURPLE", (180, 60, 180)),
    ("WHITE", (245, 245, 245)),
]
ERASER_COLOR = (70, 70, 70)
CANVAS_ALPHA = 0.82
BRUSH_SIZES = [5, 10, 16, 24]


@dataclass
class HandState:
    index_tip: tuple[int, int]
    index_mcp: tuple[int, int]
    thumb_tip: tuple[int, int]
    index_up: bool
    middle_up: bool
    pinch_distance: float
    landmarks: object


def distance(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def landmark_point(landmarks, index: int, width: int, height: int) -> tuple[int, int]:
    point = landmarks.landmark[index]
    return (min(max(int(point.x * width), 0), width - 1),
            min(max(int(point.y * height), 0), height - 1))


def finger_is_up(landmarks, tip_id: int, pip_id: int) -> bool:
    # MediaPipe image coordinates increase downwards. This vertical test is
    # intended for an upright hand with the palm generally facing the camera.
    return landmarks.landmark[tip_id].y < landmarks.landmark[pip_id].y


def extract_hand(hand_landmarks, width: int, height: int) -> HandState:
    index_tip = landmark_point(hand_landmarks, 8, width, height)
    index_mcp = landmark_point(hand_landmarks, 5, width, height)
    thumb_tip = landmark_point(hand_landmarks, 4, width, height)
    middle_tip = landmark_point(hand_landmarks, 12, width, height)
    index_up = finger_is_up(hand_landmarks, 8, 6)
    middle_up = finger_is_up(hand_landmarks, 12, 10)
    return HandState(
        index_tip=index_tip,
        index_mcp=index_mcp,
        thumb_tip=thumb_tip,
        index_up=index_up,
        middle_up=middle_up,
        pinch_distance=distance(index_tip, thumb_tip),
        landmarks=hand_landmarks,
    )


def draw_text(image, text, position, scale=0.65, color=(245, 245, 245), thickness=2):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (15, 15, 15), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def build_palette(frame, selected_name: str, brush_size: int, eraser_selected: bool):
    height, width = frame.shape[:2]
    # Compact toolbar, leaving the central drawing area open.
    bar_h = 88
    cv2.rectangle(frame, (0, 0), (width, bar_h), (28, 30, 38), -1)
    cv2.putText(frame, "AIR CANVAS", (18, 27), cv2.FONT_HERSHEY_DUPLEX,
                0.68, (245, 245, 245), 1, cv2.LINE_AA)
    draw_text(frame, "1 finger: draw | 2 fingers: hover/select | Fist: pause",
              (18, 54), 0.48, (205, 210, 220), 1)
    draw_text(frame, "C: clear   S: save   +/-: brush   Q: quit",
              (18, 75), 0.42, (175, 185, 200), 1)

    # Palette tiles along the right side.
    tile_w, gap, start_x = 56, 8, max(430, width - (len(PALETTE) + 1) * 64 - 20)
    start_x = min(start_x, width - 64 * (len(PALETTE) + 1))
    y1, y2 = 16, 54
    for i, (name, color) in enumerate(PALETTE):
        x = start_x + i * (tile_w + gap)
        if x + tile_w >= width:
            break
        cv2.rectangle(frame, (x, y1), (x + tile_w, y2), color, -1)
        if name == selected_name and not eraser_selected:
            cv2.rectangle(frame, (x - 3, y1 - 3), (x + tile_w + 3, y2 + 3),
                          (255, 255, 255), 2)
        cv2.putText(frame, name[:3], (x + 5, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (230, 230, 230), 1, cv2.LINE_AA)

    eraser_x = min(width - 62, start_x + (len(PALETTE)) * (tile_w + gap))
    cv2.rectangle(frame, (eraser_x, y1), (eraser_x + tile_w, y2), ERASER_COLOR, -1)
    if eraser_selected:
        cv2.rectangle(frame, (eraser_x - 3, y1 - 3),
                      (eraser_x + tile_w + 3, y2 + 3), (255, 255, 255), 2)
    cv2.putText(frame, "ERASE", (eraser_x + 4, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (235, 235, 235), 1, cv2.LINE_AA)
    return start_x, tile_w, gap, eraser_x


def choose_toolbar_action(point, start_x, tile_w, gap, eraser_x, y_max=76):
    x, y = point
    if y > y_max:
        return None
    for i, (name, color) in enumerate(PALETTE):
        tile_x = start_x + i * (tile_w + gap)
        if tile_x <= x <= tile_x + tile_w:
            return ("color", name, color)
    if eraser_x <= x <= eraser_x + tile_w:
        return ("eraser", "ERASER", ERASER_COLOR)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Draw in the air using hand tracking and finger gestures."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=CAMERA_WIDTH)
    parser.add_argument("--height", type=int, default=CAMERA_HEIGHT)
    parser.add_argument("--max-hands", type=int, choices=[1, 2], default=1)
    parser.add_argument("--model-complexity", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not camera.isOpened():
        print("Error: Could not open webcam. Check camera permissions or try --camera 1.")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_hands,
        model_complexity=args.model_complexity,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.60,
    )

    canvas = None
    previous_point: Optional[tuple[int, int]] = None
    selected_name = "BLUE"
    selected_color = dict(PALETTE)["BLUE"]
    eraser_selected = False
    brush_index = 1
    status = "Show your hand to begin"
    fps_samples = deque(maxlen=20)
    previous_time = time.time()
    smoothed_point = None
    toolbar_candidate = None
    toolbar_candidate_since = 0.0
    last_toolbar_action = 0.0

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Error: Failed to read webcam frame.")
                break
            frame = cv2.flip(frame, 1)  # mirror view for natural interaction
            height, width = frame.shape[:2]
            if canvas is None or canvas.shape[:2] != frame.shape[:2]:
                canvas = np.zeros_like(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = hands.process(rgb)
            rgb.flags.writeable = True

            toolbar = build_palette(frame, selected_name, BRUSH_SIZES[brush_index],
                                    eraser_selected)
            start_x, tile_w, gap, eraser_x = toolbar
            active_point = None
            draw_mode = False
            gesture_label = "NO HAND"

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                state = extract_hand(hand_landmarks, width, height)
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(100, 220, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(210, 210, 210), thickness=2),
                )

                # Exponential smoothing reduces small landmark jitters.
                point = state.index_tip
                if smoothed_point is None:
                    smoothed_point = np.array(point, dtype=np.float32)
                else:
                    smoothed_point = 0.68 * smoothed_point + 0.32 * np.array(point)
                active_point = (int(smoothed_point[0]), int(smoothed_point[1]))

                # One index finger = draw. Index + middle finger = hover/select.
                # Closed/other hand shapes = pause drawing.
                if state.index_up and not state.middle_up:
                    draw_mode = True
                    gesture_label = "DRAW"
                    status = "Drawing with index finger"
                elif state.index_up and state.middle_up:
                    gesture_label = "HOVER / SELECT"
                    status = "Two fingers: move to a color, then pinch to select"
                else:
                    gesture_label = "PAUSED"
                    status = "Drawing paused - raise index finger only"

                cv2.circle(frame, active_point, 9,
                           selected_color if not eraser_selected else ERASER_COLOR, -1)
                cv2.circle(frame, active_point, 13, (245, 245, 245), 2)

                # Two raised fingers over a toolbar tile select it after a short
                # dwell. This is easier to use than requiring a precise pinch.
                # Two-finger mode never draws.
                if state.index_up and state.middle_up:
                    action = choose_toolbar_action(active_point, start_x, tile_w, gap, eraser_x)
                    candidate_key = (action[0], action[1]) if action else None
                    if candidate_key != toolbar_candidate:
                        toolbar_candidate = candidate_key
                        toolbar_candidate_since = time.time()
                    elif action and time.time() - toolbar_candidate_since >= 0.35 and time.time() - last_toolbar_action >= 0.65:
                        kind, name, color = action
                        if kind == "color":
                            selected_name, selected_color = name, color
                            eraser_selected = False
                        else:
                            selected_name, eraser_selected = name, True
                            selected_color = ERASER_COLOR
                        status = f"Selected {name.lower()}"
                        last_toolbar_action = time.time()
                else:
                    toolbar_candidate = None

                if draw_mode and active_point[1] > 88:
                    if previous_point is not None:
                        cv2.line(
                            canvas, previous_point, active_point,
                            ERASER_COLOR if eraser_selected else selected_color,
                            42 if eraser_selected else BRUSH_SIZES[brush_index],
                            cv2.LINE_AA,
                        )
                    previous_point = active_point
                else:
                    previous_point = None
            else:
                previous_point = None
                smoothed_point = None
                toolbar_candidate = None
                status = "Show your hand to begin"

            # Alpha blend persistent strokes with the live camera.
            if canvas is not None:
                mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                drawn = mask > 0
                frame[drawn] = cv2.addWeighted(
                    frame, 1.0 - CANVAS_ALPHA, canvas, CANVAS_ALPHA, 0
                )[drawn]

            # Redraw toolbar so canvas blending never dims the UI.
            build_palette(frame, selected_name, BRUSH_SIZES[brush_index], eraser_selected)
            now = time.time()
            fps_samples.append(1.0 / max(now - previous_time, 1e-6))
            previous_time = now
            fps = sum(fps_samples) / len(fps_samples)

            cv2.rectangle(frame, (10, height - 76), (360, height - 12), (28, 30, 38), -1)
            draw_text(frame, f"Mode: {gesture_label}", (20, height - 50), 0.55,
                      (80, 235, 130) if draw_mode else (220, 220, 220), 1)
            draw_text(frame, f"Color: {selected_name} | Brush: {BRUSH_SIZES[brush_index]} px",
                      (20, height - 27), 0.48, (230, 230, 230), 1)
            draw_text(frame, f"{fps:.1f} FPS", (width - 100, height - 20),
                      0.48, (235, 235, 235), 1)

            cv2.imshow("Air Canvas - Hand Gesture Drawing", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key in (ord("c"), ord("C")):
                canvas[:] = 0
                previous_point = None
                status = "Canvas cleared"
            elif key in (ord("s"), ord("S")):
                filename = f"air_canvas_{time.strftime('%Y%m%d_%H%M%S')}.png"
                if cv2.imwrite(filename, canvas):
                    print(f"Saved drawing: {os.path.abspath(filename)}")
            elif key in (ord("+"), ord("=")):
                brush_index = min(brush_index + 1, len(BRUSH_SIZES) - 1)
            elif key in (ord("-"), ord("_")):
                brush_index = max(brush_index - 1, 0)

    finally:
        camera.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import os
    main()
