import os
import time
import urllib.request
import math

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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

