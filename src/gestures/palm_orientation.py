import os
import time
import urllib.request
import math

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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