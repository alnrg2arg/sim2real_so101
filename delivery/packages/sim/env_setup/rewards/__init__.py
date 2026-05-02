"""Reward functions for staged robotic manipulation."""
from .reach import reach_stages_30
from .open import gripper_open_stages_10
from .align import align_stages_30
from .close import close_stages_10
from .grasp import grasp_start, grasp_enough_continuous, grasp_contact_verified
from .lift import lift_progressive, lift_hold_60
from .penalties import (
    contact_force_penalty, penalty_push_object, penalty_once,
)
