"""Robot joint names, motor specs, and shared constants."""

SINGLE_ARM_JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

BI_ARM_JOINT_NAMES = [
    "left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
    "left_wrist_flex", "left_wrist_roll", "left_gripper",
    "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
    "right_wrist_flex", "right_wrist_roll", "right_gripper",
]

# USD joint limits (degrees)
SO101_USD_JOINT_LIMITS = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10.0, 100.0),
}

# Real motor limits (normalized range)
SO101_MOTOR_LIMITS = {
    "shoulder_pan": (-100.0, 100.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 100.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-100.0, 100.0),
    "gripper": (0.0, 100.0),
}

# Rest pose range (degrees)
SO101_REST_POSE_RANGE = {
    "shoulder_pan": (-30.0, 30.0),
    "shoulder_lift": (-130.0, -70.0),
    "elbow_flex": (60.0, 120.0),
    "wrist_flex": (20.0, 80.0),
    "wrist_roll": (-30.0, 30.0),
    "gripper": (-40.0, 20.0),
}

# STS3215 servo real specs
STS3215_EFFORT_LIMIT_NM = 3.5
STS3215_VELOCITY_LIMIT_RADS = 6.3  # ~60 RPM
