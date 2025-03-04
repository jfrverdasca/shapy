from copy import deepcopy

from .keypoints import get_part_idxs, kp_connections


def mirror_hand(hand_keypoint_names, source="right", target="left"):
    return [name.replace(source, target) for name in hand_keypoint_names]


__all__ = [
    "KEYPOINT_PARTS",
    "KEYPOINT_CONNECTIONS",
    "KEYPOINT_NAMES_DICT",
    "KEYPOINT_CONNECTIONS_DICT",
    "KEYPOINT_PARTS_DICT",
    "PART_NAMES",
    "KEYPOINT_PART_CONNECTION_DICTS",
]

KEYPOINT_PARTS = {
    "pelvis": "body,torso",
    "left_hip": "body,torso",
    "right_hip": "body,torso",
    "spine1": "body,torso",
    "left_knee": "body",
    "right_knee": "body",
    "spine2": "body,torso,upper",
    "left_ankle": "body",
    "right_ankle": "body",
    "spine3": "body,torso,upper",
    "left_foot": "body",
    "right_foot": "body",
    "neck": "body,head,face,torso,upper",
    "left_collar": "body,torso,upper",
    "right_collar": "body,torso,upper",
    "head": "body,torso,upper,head",
    "left_shoulder": "body,torso,upper",
    "right_shoulder": "body,torso,upper",
    "left_elbow": "body,torso,upper",
    "right_elbow": "body,torso,upper",
    "left_wrist": "body,hand",
    "right_wrist": "body,hand",
    "jaw": "body,torso,upper,head",
    "left_eye_smplx": "body,torso,upper,head",
    "right_eye_smplx": "body,torso,upper,head",
    "left_index1": "hand",
    "left_index2": "hand",
    "left_index3": "hand",
    "left_middle1": "hand",
    "left_middle2": "hand",
    "left_middle3": "hand",
    "left_pinky1": "hand",
    "left_pinky2": "hand",
    "left_pinky3": "hand",
    "left_ring1": "hand",
    "left_ring2": "hand",
    "left_ring3": "hand",
    "left_thumb1": "hand",
    "left_thumb2": "hand",
    "left_thumb3": "hand",
    "right_index1": "hand",
    "right_index2": "hand",
    "right_index3": "hand",
    "right_middle1": "hand",
    "right_middle2": "hand",
    "right_middle3": "hand",
    "right_pinky1": "hand",
    "right_pinky2": "hand",
    "right_pinky3": "hand",
    "right_ring1": "hand",
    "right_ring2": "hand",
    "right_ring3": "hand",
    "right_thumb1": "hand",
    "right_thumb2": "hand",
    "right_thumb3": "hand",
    "nose": "body,torso,upper,head",
    "right_eye": "body,torso,upper,head",
    "left_eye": "body,torso,upper,head",
    "right_ear": "body,torso,upper,head",
    "left_ear": "body,torso,upper,head",
    "left_big_toe": "body",
    "left_small_toe": "body",
    "left_heel": "body",
    "right_big_toe": "body",
    "right_small_toe": "body",
    "right_heel": "body",
    "left_thumb": "hand",
    "left_index": "hand",
    "left_middle": "hand",
    "left_ring": "hand",
    "left_pinky": "hand",
    "right_thumb": "hand",
    "right_index": "hand",
    "right_middle": "hand",
    "right_ring": "hand",
    "right_pinky": "hand",
    "right_eye_brow1": "face,torso,upper,head",
    "right_eye_brow2": "face,torso,upper,head",
    "right_eye_brow3": "face,torso,upper,head",
    "right_eye_brow4": "face,torso,upper,head",
    "right_eye_brow5": "face,torso,upper,head",
    "left_eye_brow5": "face,torso,upper,head",
    "left_eye_brow4": "face,torso,upper,head",
    "left_eye_brow3": "face,torso,upper,head",
    "left_eye_brow2": "face,torso,upper,head",
    "left_eye_brow1": "face,torso,upper,head",
    "nose1": "face,torso,upper,head",
    "nose2": "face,torso,upper,head",
    "nose3": "face,torso,upper,head",
    "nose4": "face,torso,upper,head",
    "right_nose_2": "face,torso,upper,head",
    "right_nose_1": "face,torso,upper,head",
    "nose_middle": "face,torso,upper,head",
    "left_nose_1": "face,torso,upper,head",
    "left_nose_2": "face,torso,upper,head",
    "right_eye1": "face,torso,upper,head",
    "right_eye2": "face,torso,upper,head",
    "right_eye3": "face,torso,upper,head",
    "right_eye4": "face,torso,upper,head",
    "right_eye5": "face,torso,upper,head",
    "right_eye6": "face,torso,upper,head",
    "left_eye4": "face,torso,upper,head",
    "left_eye3": "face,torso,upper,head",
    "left_eye2": "face,torso,upper,head",
    "left_eye1": "face,torso,upper,head",
    "left_eye6": "face,torso,upper,head",
    "left_eye5": "face,torso,upper,head",
    "right_mouth_1": "face,torso,upper,head",
    "right_mouth_2": "face,torso,upper,head",
    "right_mouth_3": "face,torso,upper,head",
    "mouth_top": "face,torso,upper,head",
    "left_mouth_3": "face,torso,upper,head",
    "left_mouth_2": "face,torso,upper,head",
    "left_mouth_1": "face,torso,upper,head",
    "left_mouth_5": "face,torso,upper,head",
    "left_mouth_4": "face,torso,upper,head",
    "mouth_bottom": "face,torso,upper,head",
    "right_mouth_4": "face,torso,upper,head",
    "right_mouth_5": "face,torso,upper,head",
    "right_lip_1": "face,torso,upper,head",
    "right_lip_2": "face,torso,upper,head",
    "lip_top": "face,torso,upper,head",
    "left_lip_2": "face,torso,upper,head",
    "left_lip_1": "face,torso,upper,head",
    "left_lip_3": "face,torso,upper,head",
    "lip_bottom": "face,torso,upper,head",
    "right_lip_3": "face,torso,upper,head",
    "right_contour_1": "face,torso,upper,head",
    "right_contour_2": "face,torso,upper,head",
    "right_contour_3": "face,torso,upper,head",
    "right_contour_4": "face,torso,upper,head",
    "right_contour_5": "face,torso,upper,head",
    "right_contour_6": "face,torso,upper,head",
    "right_contour_7": "face,torso,upper,head",
    "right_contour_8": "face,torso,upper,head",
    "contour_middle": "face,torso,upper,head",
    "left_contour_8": "face,torso,upper,head",
    "left_contour_7": "face,torso,upper,head",
    "left_contour_6": "face,torso,upper,head",
    "left_contour_5": "face,torso,upper,head",
    "left_contour_4": "face,torso,upper,head",
    "left_contour_3": "face,torso,upper,head",
    "left_contour_2": "face,torso,upper,head",
    "left_contour_1": "face,torso,upper,head",
    "head_top": "body,head,torso,upper",
}

PART_NAMES = {
    "body",
    "left_hand",
    "right_hand",
    "face",
    "head",
    "upper",
    "torso",
}

KEYPOINT_CONNECTIONS = [
    ["pelvis", "spine1"],
    ["spine1", "spine2"],
    ["spine2", "spine3"],
    ["spine3", "left_collar"],
    ["spine3", "right_collar"],
    ["left_collar", "left_shoulder"],
    ["right_collar", "right_shoulder"],
    ["spine3", "neck"],
    ["neck", "head"],
    ["head", "head_top"],
    ["left_eye", "nose"],
    ["right_eye", "nose"],
    ["right_eye", "right_ear"],
    ["left_eye", "left_ear"],
    ["right_shoulder", "right_elbow"],
    ["right_elbow", "right_wrist"],
    ["left_wrist", "left_hand"],
    ["right_wrist", "right_hand"],
    # Left Hand
    ["left_shoulder", "left_elbow"],
    ["left_elbow", "left_wrist"],
    # Left Thumb
    ["left_wrist", "left_thumb1"],
    ["left_thumb1", "left_thumb2"],
    ["left_thumb2", "left_thumb3"],
    ["left_thumb3", "left_thumb"],
    # Left Index
    ["left_wrist", "left_index1"],
    ["left_index1", "left_index2"],
    ["left_index2", "left_index3"],
    ["left_index3", "left_index"],
    # Left Middle
    ["left_wrist", "left_middle1"],
    ["left_middle1", "left_middle2"],
    ["left_middle2", "left_middle3"],
    ["left_middle3", "left_middle"],
    # Left Ring
    ["left_wrist", "left_ring1"],
    ["left_ring1", "left_ring2"],
    ["left_ring2", "left_ring3"],
    ["left_ring3", "left_ring"],
    # Left Pinky
    ["left_wrist", "left_pinky1"],
    ["left_pinky1", "left_pinky2"],
    ["left_pinky2", "left_pinky3"],
    ["left_pinky3", "left_pinky"],
    # Right Thumb
    ["right_wrist", "right_thumb1"],
    ["right_thumb1", "right_thumb2"],
    ["right_thumb2", "right_thumb3"],
    ["right_thumb3", "right_thumb"],
    # Right Index
    ["right_wrist", "right_index1"],
    ["right_index1", "right_index2"],
    ["right_index2", "right_index3"],
    ["right_index3", "right_index"],
    # Right Middle
    ["right_wrist", "right_middle1"],
    ["right_middle1", "right_middle2"],
    ["right_middle2", "right_middle3"],
    ["right_middle3", "right_middle"],
    # Right Ring
    ["right_wrist", "right_ring1"],
    ["right_ring1", "right_ring2"],
    ["right_ring2", "right_ring3"],
    ["right_ring3", "right_ring"],
    # Right Pinky
    ["right_wrist", "right_pinky1"],
    ["right_pinky1", "right_pinky2"],
    ["right_pinky2", "right_pinky3"],
    ["right_pinky3", "right_pinky"],
    # Right Foot
    ["right_hip", "right_knee"],
    ["right_knee", "right_ankle"],
    ["right_ankle", "right_heel"],
    ["right_ankle", "right_big_toe"],
    ["right_ankle", "right_small_toe"],
    ["left_hip", "left_knee"],
    ["left_knee", "left_ankle"],
    ["left_ankle", "left_heel"],
    ["left_ankle", "left_big_toe"],
    ["left_ankle", "left_small_toe"],
    ["neck", "right_shoulder"],
    ["neck", "left_shoulder"],
    ["neck", "nose"],
    #  ['pelvis', 'neck'],
    ["pelvis", "left_hip"],
    ["pelvis", "right_hip"],
    # Left Eye brow
    ["left_eye_brow1", "left_eye_brow2"],
    ["left_eye_brow2", "left_eye_brow3"],
    ["left_eye_brow3", "left_eye_brow4"],
    ["left_eye_brow4", "left_eye_brow5"],
    # Right Eye brow
    ["right_eye_brow1", "right_eye_brow2"],
    ["right_eye_brow2", "right_eye_brow3"],
    ["right_eye_brow3", "right_eye_brow4"],
    ["right_eye_brow4", "right_eye_brow5"],
    # Left Eye
    ["left_eye1", "left_eye2"],
    ["left_eye2", "left_eye3"],
    ["left_eye3", "left_eye4"],
    ["left_eye4", "left_eye5"],
    ["left_eye5", "left_eye6"],
    # Right Eye
    ["right_eye1", "right_eye2"],
    ["right_eye2", "right_eye3"],
    ["right_eye3", "right_eye4"],
    ["right_eye4", "right_eye5"],
    ["right_eye5", "right_eye6"],
    # Nose Vertical
    ["nose1", "nose2"],
    ["nose2", "nose3"],
    ["nose3", "nose4"],
    # Nose Horizontal
    ["left_nose_1", "left_nose_2"],
    ["left_nose_2", "nose_middle"],
    ["nose_middle", "right_nose_2"],
    ["right_nose_2", "right_nose_1"],
    # Mouth
    ["left_mouth_1", "left_mouth_2"],
    ["left_mouth_2", "left_mouth_3"],
    ["left_mouth_3", "mouth_top"],
    ["mouth_top", "right_mouth_3"],
    ["right_mouth_3", "right_mouth_2"],
    ["right_mouth_2", "right_mouth_1"],
    ["right_mouth_1", "right_mouth_4"],
    ["right_mouth_4", "right_mouth_5"],
    ["right_mouth_5", "mouth_bottom"],
    ["mouth_bottom", "left_mouth_4"],
    ["left_mouth_4", "left_mouth_5"],
    ["left_mouth_5", "left_mouth_1"],
    # Lips
    ["left_lip_1", "left_lip_2"],
    ["left_lip_2", "lip_top"],
    ["lip_top", "right_lip_2"],
    ["right_lip_2", "right_lip_1"],
    ["right_lip_1", "right_lip_3"],
    ["right_lip_3", "lip_bottom"],
    ["lip_bottom", "left_lip_3"],
    ["left_lip_3", "left_lip_1"],
    # Contour
    ["left_contour_1", "left_contour_2"],
    ["left_contour_2", "left_contour_3"],
    ["left_contour_3", "left_contour_4"],
    ["left_contour_4", "left_contour_5"],
    ["left_contour_5", "left_contour_6"],
    ["left_contour_6", "left_contour_7"],
    ["left_contour_7", "left_contour_8"],
    ["left_contour_8", "contour_middle"],
    ["contour_middle", "right_contour_8"],
    ["right_contour_8", "right_contour_7"],
    ["right_contour_7", "right_contour_6"],
    ["right_contour_6", "right_contour_5"],
    ["right_contour_5", "right_contour_4"],
    ["right_contour_4", "right_contour_3"],
    ["right_contour_3", "right_contour_2"],
    ["right_contour_2", "right_contour_1"],
]


FACIAL_LANDMARKS = [
    "right_eye_brow1",
    "right_eye_brow2",
    "right_eye_brow3",
    "right_eye_brow4",
    "right_eye_brow5",
    "left_eye_brow5",
    "left_eye_brow4",
    "left_eye_brow3",
    "left_eye_brow2",
    "left_eye_brow1",
    "nose1",
    "nose2",
    "nose3",
    "nose4",
    "right_nose_2",
    "right_nose_1",
    "nose_middle",
    "left_nose_1",
    "left_nose_2",
    "right_eye1",
    "right_eye2",
    "right_eye3",
    "right_eye4",
    "right_eye5",
    "right_eye6",
    "left_eye4",
    "left_eye3",
    "left_eye2",
    "left_eye1",
    "left_eye6",
    "left_eye5",
    "right_mouth_1",
    "right_mouth_2",
    "right_mouth_3",
    "mouth_top",
    "left_mouth_3",
    "left_mouth_2",
    "left_mouth_1",
    "left_mouth_5",  # 59 in OpenPose output
    "left_mouth_4",  # 58 in OpenPose output
    "mouth_bottom",
    "right_mouth_4",
    "right_mouth_5",
    "right_lip_1",
    "right_lip_2",
    "lip_top",
    "left_lip_2",
    "left_lip_1",
    "left_lip_3",
    "lip_bottom",
    "right_lip_3",
    # Face contour
    "right_contour_1",
    "right_contour_2",
    "right_contour_3",
    "right_contour_4",
    "right_contour_5",
    "right_contour_6",
    "right_contour_7",
    "right_contour_8",
    "contour_middle",
    "left_contour_8",
    "left_contour_7",
    "left_contour_6",
    "left_contour_5",
    "left_contour_4",
    "left_contour_3",
    "left_contour_2",
    "left_contour_1",
]

SMPL_KEYPOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]

SMPLH_KEYPOINT_NAMES = SMPL_KEYPOINT_NAMES[:-2] + [
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
]
assert len(SMPLH_KEYPOINT_NAMES) == 22 + 2 * 15


SMPLX_KEYPOINT_NAMES = (
    SMPL_KEYPOINT_NAMES[:-2]
    + ["jaw", "left_eye_smplx", "right_eye_smplx"]
    + SMPLH_KEYPOINT_NAMES[22:]
    + FACIAL_LANDMARKS
)

MANO_NAMES = [
    "wrist",
    "index1",
    "index2",
    "index3",
    "middle1",
    "middle2",
    "middle3",
    "pinky1",
    "pinky2",
    "pinky3",
    "ring1",
    "ring2",
    "ring3",
    "thumb1",
    "thumb2",
    "thumb3",
]
HO3D_NAMES = MANO_NAMES + ["thumb", "index", "middle", "ring", "pinky"]


INTERHAND26M_RIGHT = [
    "right_thumb",
    "right_thumb3",
    "right_thumb2",
    "right_thumb1",
    "right_index",
    "right_index3",
    "right_index2",
    "right_index1",
    "right_middle",
    "right_middle3",
    "right_middle2",
    "right_middle1",
    "right_ring",
    "right_ring3",
    "right_ring2",
    "right_ring1",
    "right_pinky",
    "right_pinky3",
    "right_pinky2",
    "right_pinky1",
    "right_wrist",
]
INTERHAND26M_LEFT = mirror_hand(INTERHAND26M_RIGHT, "right", "left")

AGORA_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
    "right_eye_brow1",
    "right_eye_brow2",
    "right_eye_brow3",
    "right_eye_brow4",
    "right_eye_brow5",
    "left_eye_brow5",
    "left_eye_brow4",
    "left_eye_brow3",
    "left_eye_brow2",
    "left_eye_brow1",
    "nose1",
    "nose2",
    "nose3",
    "nose4",
    "right_nose_2",
    "right_nose_1",
    "nose_middle",
    "left_nose_1",
    "left_nose_2",
    "right_eye1",
    "right_eye2",
    "right_eye3",
    "right_eye4",
    "right_eye5",
    "right_eye6",
    "left_eye4",
    "left_eye3",
    "left_eye2",
    "left_eye1",
    "left_eye6",
    "left_eye5",
    "right_mouth_1",
    "right_mouth_2",
    "right_mouth_3",
    "mouth_top",
    "left_mouth_3",
    "left_mouth_2",
    "left_mouth_1",
    "left_mouth_5",  # 59 in OpenPose output
    "left_mouth_4",  # 58 in OpenPose output
    "mouth_bottom",
    "right_mouth_4",
    "right_mouth_5",
    "right_lip_1",
    "right_lip_2",
    "lip_top",
    "left_lip_2",
    "left_lip_1",
    "left_lip_3",
    "lip_bottom",
    "right_lip_3",
    # Face contour
    #  'right_contour_1',
    #  'right_contour_2',
    #  'right_contour_3',
    #  'right_contour_4',
    #  'right_contour_5',
    #  'right_contour_6',
    #  'right_contour_7',
    #  'right_contour_8',
    #  'contour_middle',
    #  'left_contour_8',
    #  'left_contour_7',
    #  'left_contour_6',
    #  'left_contour_5',
    #  'left_contour_4',
    #  'left_contour_3',
    #  'left_contour_2',
    #  'left_contour_1',
]


EHF_KEYPOINTS = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplx",
    "right_eye_smplx",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
    "right_eye_brow1",
    "right_eye_brow2",
    "right_eye_brow3",
    "right_eye_brow4",
    "right_eye_brow5",
    "left_eye_brow5",
    "left_eye_brow4",
    "left_eye_brow3",
    "left_eye_brow2",
    "left_eye_brow1",
    "nose1",
    "nose2",
    "nose3",
    "nose4",
    "right_nose_2",
    "right_nose_1",
    "nose_middle",
    "left_nose_1",
    "left_nose_2",
    "right_eye1",
    "right_eye2",
    "right_eye3",
    "right_eye4",
    "right_eye5",
    "right_eye6",
    "left_eye4",
    "left_eye3",
    "left_eye2",
    "left_eye1",
    "left_eye6",
    "left_eye5",
    "right_mouth_1",
    "right_mouth_2",
    "right_mouth_3",
    "mouth_top",
    "left_mouth_3",
    "left_mouth_2",
    "left_mouth_1",
    "left_mouth_5",  # 59 in OpenPose output
    "left_mouth_4",  # 58 in OpenPose output
    "mouth_bottom",
    "right_mouth_4",
    "right_mouth_5",
    "right_lip_1",
    "right_lip_2",
    "lip_top",
    "left_lip_2",
    "left_lip_1",
    "left_lip_3",
    "lip_bottom",
    "right_lip_3",
    # Face contour
    "right_contour_1",
    "right_contour_2",
    "right_contour_3",
    "right_contour_4",
    "right_contour_5",
    "right_contour_6",
    "right_contour_7",
    "right_contour_8",
    "contour_middle",
    "left_contour_8",
    "left_contour_7",
    "left_contour_6",
    "left_contour_5",
    "left_contour_4",
    "left_contour_3",
    "left_contour_2",
    "left_contour_1",
]

OPENPOSE18_KEYPOINT_NAMES_v1 = [
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
]


OPENPOSE19_KEYPOINT_NAMES_v1 = [
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_wrist",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "left_thumb",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_index",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_middle",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_ring",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_pinky",
    "right_wrist",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "right_thumb",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_index",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_middle",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_ring",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_pinky",
    # Face contour
    "right_contour_1",
    "right_contour_2",
    "right_contour_3",
    "right_contour_4",
    "right_contour_5",
    "right_contour_6",
    "right_contour_7",
    "right_contour_8",
    "contour_middle",
    "left_contour_8",
    "left_contour_7",
    "left_contour_6",
    "left_contour_5",
    "left_contour_4",
    "left_contour_3",
    "left_contour_2",
    "left_contour_1",
    # Eye brows
    "right_eye_brow1",
    "right_eye_brow2",
    "right_eye_brow3",
    "right_eye_brow4",
    "right_eye_brow5",
    "left_eye_brow5",
    "left_eye_brow4",
    "left_eye_brow3",
    "left_eye_brow2",
    "left_eye_brow1",
    "nose1",
    "nose2",
    "nose3",
    "nose4",
    "right_nose_2",
    "right_nose_1",
    "nose_middle",
    "left_nose_1",
    "left_nose_2",
    "right_eye1",
    "right_eye2",
    "right_eye3",
    "right_eye4",
    "right_eye5",
    "right_eye6",
    "left_eye4",
    "left_eye3",
    "left_eye2",
    "left_eye1",
    "left_eye6",
    "left_eye5",
    "right_mouth_1",
    "right_mouth_2",
    "right_mouth_3",
    "mouth_top",
    "left_mouth_3",
    "left_mouth_2",
    "left_mouth_1",
    "left_mouth_5",  # 59 in OpenPose output
    "left_mouth_4",  # 58 in OpenPose output
    "mouth_bottom",
    "right_mouth_4",
    "right_mouth_5",
    "right_lip_1",
    "right_lip_2",
    "lip_top",
    "left_lip_2",
    "left_lip_1",
    "left_lip_3",
    "lip_bottom",
    "right_lip_3",
]

FEET_KEYPS_NAMES = [
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
]
OPENPOSE25_KEYPOINT_NAMES_V1 = deepcopy(OPENPOSE19_KEYPOINT_NAMES_v1)
start = 19
for feet_name in FEET_KEYPS_NAMES:
    OPENPOSE25_KEYPOINT_NAMES_V1.insert(start, feet_name)
    start += 1

MPII_KEYPOINT_NAMES = [
    "right_ankle",
    "right_knee",
    "right_hip",
    "left_hip",
    "left_knee",
    "left_ankle",
    "pelvis",
    "thorax",
    "upper_neck",
    "head_top",
    "right_wrist",
    "right_elbow",
    "right_shoulder",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    # Hand joints
    "left_wrist",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "left_thumb",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_index",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_middle",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_ring",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_pinky",
    "right_wrist",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "right_thumb",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_index",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_middle",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_ring",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_pinky",
]

FLAME_KEYPOINT_NAMES = [
    "global",
    "neck",
    "jaw",
    "left_eye",
    "right_eye",
] + FACIAL_LANDMARKS
FFHQ_KEYPOINTS = FLAME_KEYPOINT_NAMES

VGGFACE2_NAMES = FACIAL_LANDMARKS[-17:] + FACIAL_LANDMARKS[:-17]
ETHNICITY_NAMES = FACIAL_LANDMARKS[-17:] + FACIAL_LANDMARKS[:-17]


COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
COCO_WHOLE_BODY_KEYPOINTS = (
    COCO_KEYPOINTS
    + FEET_KEYPS_NAMES
    + [
        "left_wrist",
        "left_thumb1",
        "left_thumb2",
        "left_thumb3",
        "left_thumb",
        "left_index1",
        "left_index2",
        "left_index3",
        "left_index",
        "left_middle1",
        "left_middle2",
        "left_middle3",
        "left_middle",
        "left_ring1",
        "left_ring2",
        "left_ring3",
        "left_ring",
        "left_pinky1",
        "left_pinky2",
        "left_pinky3",
        "left_pinky",
        "right_wrist",
        "right_thumb1",
        "right_thumb2",
        "right_thumb3",
        "right_thumb",
        "right_index1",
        "right_index2",
        "right_index3",
        "right_index",
        "right_middle1",
        "right_middle2",
        "right_middle3",
        "right_middle",
        "right_ring1",
        "right_ring2",
        "right_ring3",
        "right_ring",
        "right_pinky1",
        "right_pinky2",
        "right_pinky3",
        "right_pinky",
    ]
    + FACIAL_LANDMARKS[-17:]
    + FACIAL_LANDMARKS[:-17]
)

THREEDPW_KEYPOINT_NAMES = [
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
]


POSETRACK_KEYPOINT_NAMES = [
    "nose",
    "neck",
    "head_top",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "pelvis",
]

AICH_KEYPOINT_NAMES = [
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "head_top",
    "neck",
    "pelvis",
]


SPIN_KEYPOINT_NAMES = [
    "right_ankle",
    "right_knee",
    "right_hip",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_wrist",
    "right_elbow",
    "right_shoulder",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "neck",
    "head_top",
    "pelvis",
    "thorax",
    "spine",
    "h36m_jaw",
    "h36m_head",
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
]

SPINX_KEYPOINT_NAMES = [
    "right_ankle",
    "right_knee",
    "right_hip",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_wrist",
    "right_elbow",
    "right_shoulder",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "neck",
    "head_top",
    "pelvis",
    "thorax",
    "spine",
    "h36m_jaw",
    "h36m_head",
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
] + OPENPOSE25_KEYPOINT_NAMES_V1[25:]


PANOPTIC_KEYPOINT_NAMES = [
    "neck",
    "nose",
    "pelvis",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_eye",
    "left_ear",
    "right_eye",
    "right_ear",
]
PANOPTIC_KEYPOINT_NAMES += (
    OPENPOSE19_KEYPOINT_NAMES_v1[19 : 19 + 2 * 21]
    + OPENPOSE19_KEYPOINT_NAMES_v1[19 + 2 * 21 + 17 :]
    + OPENPOSE19_KEYPOINT_NAMES_v1[19 + 2 * 21 : 19 + 2 * 21 + 17]
)

FREIHAND_NAMES_RIGHT = [
    "right_wrist",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "right_thumb",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_index",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_middle",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_ring",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_pinky",
]
FREIHAND_NAMES_LEFT = mirror_hand(FREIHAND_NAMES_RIGHT, "right", "left")

LSP_NAMES = [
    "right_ankle",
    "right_knee",
    "right_hip",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_wrist",
    "right_elbow",
    "right_shoulder",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "neck",
    "head_top",
]


RAW_H36M_NAMES = [
    "pelvis",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_hip",
    "right_knee",
    "right_ankle",
    "spine",
    "neck",  # 'thorax',
    "neck/nose",
    "head",  # 'head_h36m',
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
]

H36M_NAMES = [
    "right_ankle",
    "right_knee",
    "right_hip",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_wrist",
    "right_elbow",
    "right_shoulder",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "neck",
    "head_top",
    "pelvis_(mpii)",
    "thorax_(mpii)",
    "spine_(h36m)",
    "jaw_(h36m)",
    "head",
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
]

PANOPTIC_HAND_KEYPOINT_NAMES_RIGHT = [
    "right_wrist",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "right_thumb",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_index",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_middle",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_ring",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_pinky",
]
PANOPTIC_HAND_KEYPOINT_NAMES_LEFT = mirror_hand(
    PANOPTIC_HAND_KEYPOINT_NAMES_RIGHT, "right", "left"
)

YOUTUBE3D_HANDS_RIGHT = [
    "right_wrist",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
]
YOUTUBE3D_HANDS_LEFT = mirror_hand(YOUTUBE3D_HANDS_RIGHT, "right", "left")

KEYPOINT_NAMES_DICT = {
    "smpl": SMPL_KEYPOINT_NAMES,
    "smplh": SMPLH_KEYPOINT_NAMES,
    "smplx": SMPLX_KEYPOINT_NAMES,
    "mano": MANO_NAMES,
    "mano-from-smplx": SMPLX_KEYPOINT_NAMES,
    "flame-from-smplx": SMPLX_KEYPOINT_NAMES,
    "flame": FLAME_KEYPOINT_NAMES,
    "openpose18_v1": OPENPOSE18_KEYPOINT_NAMES_v1,
    "openpose19_v1": OPENPOSE19_KEYPOINT_NAMES_v1,
    "openpose25_v1": OPENPOSE25_KEYPOINT_NAMES_V1,
    "mpii": MPII_KEYPOINT_NAMES,
    "ffhq": FFHQ_KEYPOINTS,
    "ehf": EHF_KEYPOINTS,
    "coco": COCO_KEYPOINTS,
    "whole-coco": COCO_WHOLE_BODY_KEYPOINTS,
    "3dpw": THREEDPW_KEYPOINT_NAMES,
    "posetrack": POSETRACK_KEYPOINT_NAMES,
    "aich": AICH_KEYPOINT_NAMES,
    "spin": SPIN_KEYPOINT_NAMES,
    "spinx": SPINX_KEYPOINT_NAMES,
    "panoptic": PANOPTIC_KEYPOINT_NAMES,
    "freihand-left": FREIHAND_NAMES_LEFT,
    "freihand-right": FREIHAND_NAMES_RIGHT,
    "lsp": LSP_NAMES,
    "raw_h36m": RAW_H36M_NAMES,
    "h36m": H36M_NAMES,
    "mtc-right": PANOPTIC_HAND_KEYPOINT_NAMES_RIGHT,
    "mtc-left": PANOPTIC_HAND_KEYPOINT_NAMES_LEFT,
    "ho3d": HO3D_NAMES,
    "vggface2": VGGFACE2_NAMES,
    "ethnicity": ETHNICITY_NAMES,
    "youtube3d-hand-right": YOUTUBE3D_HANDS_RIGHT,
    "youtube3d-hand-left": YOUTUBE3D_HANDS_LEFT,
    "interhand26m-right": INTERHAND26M_RIGHT,
    "interhand26m-left": INTERHAND26M_LEFT,
    "agora": AGORA_NAMES,
}

KEYPOINT_CONNECTIONS_DICT = {}
for dset_name, keypoint_names in KEYPOINT_NAMES_DICT.items():
    KEYPOINT_CONNECTIONS_DICT[dset_name] = kp_connections(
        keypoint_names, KEYPOINT_CONNECTIONS
    )

KEYPOINT_PARTS_DICT = {}
for dset_name, keypoint_names in KEYPOINT_NAMES_DICT.items():
    KEYPOINT_PARTS_DICT[dset_name] = get_part_idxs(keypoint_names, KEYPOINT_PARTS)

# Stores for each part all edge pairs
KEYPOINT_PART_CONNECTION_DICTS = {}
for dset_name, keypoint_names in KEYPOINT_NAMES_DICT.items():
    KEYPOINT_PART_CONNECTION_DICTS[dset_name] = {}
    for part_name in PART_NAMES:
        KEYPOINT_PART_CONNECTION_DICTS[dset_name][part_name] = kp_connections(
            keypoint_names,
            KEYPOINT_CONNECTIONS,
            part=part_name,
            keypoint_parts=KEYPOINT_PARTS,
        )
