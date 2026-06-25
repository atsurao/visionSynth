import os
import time
import urllib.request
import math

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# 21 hand landmark connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")

def calculate_angle(a, b, c):
    """
    Returns angle ABC in degrees.
    """

    ba = (
        a.x - b.x,
        a.y - b.y,
    )

    bc = (
        c.x - b.x,
        c.y - b.y,
    )

    dot = (
        ba[0] * bc[0]
        +
        ba[1] * bc[1]
    )

    mag_ba = math.hypot(
        ba[0],
        ba[1],
    )

    mag_bc = math.hypot(
        bc[0],
        bc[1],
    )

    if mag_ba == 0 or mag_bc == 0:
        return 0

    cosine = dot / (mag_ba * mag_bc)

    cosine = max(
        -1,
        min(1, cosine),
    )

    return math.degrees(
        math.acos(cosine)
    )

def get_palm_orientation(hand_landmarks):
    """
    Returns:
        "Front" -> palm facing camera
        "Back"  -> back of hand facing camera
    """

    wrist = hand_landmarks[0]
    index_mcp = hand_landmarks[5]
    pinky_mcp = hand_landmarks[17]

    v1 = (
        index_mcp.x - wrist.x,
        index_mcp.y - wrist.y,
    )

    v2 = (
        pinky_mcp.x - wrist.x,
        pinky_mcp.y - wrist.y,
    )

    cross_z = (
        v1[0] * v2[1]
        -
        v1[1] * v2[0]
    )

    if cross_z > 0:
        return "Front"
    else:
        return "Back"

def get_finger_state(hand_landmarks, handedness):
    """
    Returns:
    (thumb, index, middle, ring, pinky)
    """

    thumb_angle = calculate_angle(
        hand_landmarks[1],
        hand_landmarks[2],
        hand_landmarks[4],
    )

    index_angle = calculate_angle(
        hand_landmarks[5],
        hand_landmarks[6],
        hand_landmarks[8],
    )

    middle_angle = calculate_angle(
        hand_landmarks[9],
        hand_landmarks[10],
        hand_landmarks[12],
    )

    ring_angle = calculate_angle(
        hand_landmarks[13],
        hand_landmarks[14],
        hand_landmarks[16],
    )

    pinky_angle = calculate_angle(
        hand_landmarks[17],
        hand_landmarks[18],
        hand_landmarks[20],
    )

    thumb = thumb_angle > 140

    index = index_angle > 150

    middle = middle_angle > 150

    ring = ring_angle > 150

    pinky = pinky_angle > 150

    return (
        thumb,
        index,
        middle,
        ring,
        pinky,
    )


def hand_shape_index(finger_state):
    """
    Converts finger states into a number from 0-31.
    """

    thumb, index, middle, ring, pinky = finger_state

    value = 0

    if thumb:
        value |= 1

    if index:
        value |= 2

    if middle:
        value |= 4

    if ring:
        value |= 8

    if pinky:
        value |= 16

    return value


def shape_string(finger_state):
    names = ["T", "I", "M", "R", "P"]

    return "".join(
        name
        for name, active in zip(names, finger_state)
        if active
    ) or "None"


def draw_landmarks(frame, hand_landmarks_list, handedness_list):
    h, w, _ = frame.shape

    for hand_landmarks, handedness in zip(
        hand_landmarks_list,
        handedness_list,
    ):
        points = [
            (int(lm.x * w), int(lm.y * h))
            for lm in hand_landmarks
        ]

        # Draw skeleton
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(
                frame,
                points[start_idx],
                points[end_idx],
                (255, 255, 255),
                2,
            )

        # Draw landmarks
        for x, y in points:
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        label = handedness[0].category_name

        # Determine hand shape
        finger_state = get_finger_state(
            hand_landmarks,
            handedness,
        )

        shape_index = hand_shape_index(
            finger_state,
        )

        shape_name = shape_string(
            finger_state,
        )

        palm_orientation = get_palm_orientation(
            hand_landmarks
        )

        x_min = min(p[0] for p in points)
        y_min = min(p[1] for p in points)

        # Hand label
        cv2.putText(
            frame,
            label,
            (x_min, y_min - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        # Shape index
        cv2.putText(
            frame,
            f"Index: {shape_index}",
            (x_min, y_min - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        display_text = (
            f"{shape_name} | {palm_orientation}"
        )

        # Finger state string
        cv2.putText(
            frame,
            display_text,
            (x_min, y_min + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )


def main():
    ensure_model()

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    start_time = time.time()

    with vision.HandLandmarker.create_from_options(options) as landmarker:

        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            frame_timestamp_ms = int(
                (time.time() - start_time) * 1000
            )

            result = landmarker.detect_for_video(
                mp_image,
                frame_timestamp_ms,
            )

            if result.hand_landmarks:
                draw_landmarks(
                    frame,
                    result.hand_landmarks,
                    result.handedness,
                )

            cv2.imshow(
                "VisionSynth Hand Tracking",
                frame,
            )

            key = cv2.waitKey(5) & 0xFF

            if key == 27 or key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()