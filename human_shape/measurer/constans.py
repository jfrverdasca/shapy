# Landmarks
SMPLX_LANDMARK_INDICES = {
    "HEAD_TOP": 8976,
    "HEAD_LEFT_TEMPLE": 1980,
    "NECK_ADAM_APPLE": 8940,
    "LEFT_HEEL": 8847,
    "RIGHT_HEEL": 8635,
    "LEFT_NIPPLE": 3572,
    "RIGHT_NIPPLE": 8340,
    "SHOULDER_TOP": 5616,
    "INSEAM_POINT": 5601,
    "BELLY_BUTTON": 5939,
    "BACK_BELLY_BUTTON": 5941,
    "CROTCH": 3797,
    "PUBIC_BONE": 5949,
    "RIGHT_WRIST": 7449,
    "LEFT_WRIST": 4823,
    "RIGHT_BICEP": 6788,
    "RIGHT_FOREARM": 7266,
    "LEFT_SHOULDER": 4442,
    "RIGHT_SHOULDER": 7218,
    "LOW_LEFT_HIP": 4112,
    "LEFT_THIGH": 3577,
    "LEFT_CALF": 3732,
    "LEFT_ANKLE": 5880,
}

SMPLX_LANDMARK_INDICES["HEELS"] = (
    SMPLX_LANDMARK_INDICES["LEFT_HEEL"],
    SMPLX_LANDMARK_INDICES["RIGHT_HEEL"],
)

# Joints
SMPLX_NUM_JOINTS = 55
SMPLX_IND2JOINT = {
    0: "pelvis",
    1: "left_hip",
    2: "right_hip",
    3: "spine1",
    4: "left_knee",
    5: "right_knee",
    6: "spine2",
    7: "left_ankle",
    8: "right_ankle",
    9: "spine3",
    10: "left_foot",
    11: "right_foot",
    12: "neck",
    13: "left_collar",
    14: "right_collar",
    15: "head",
    16: "left_shoulder",
    17: "right_shoulder",
    18: "left_elbow",
    19: "right_elbow",
    20: "left_wrist",
    21: "right_wrist",
    22: "jaw",
    23: "left_eye",
    24: "right_eye",
    25: "left_index1",
    26: "left_index2",
    27: "left_index3",
    28: "left_middle1",
    29: "left_middle2",
    30: "left_middle3",
    31: "left_pinky1",
    32: "left_pinky2",
    33: "left_pinky3",
    34: "left_ring1",
    35: "left_ring2",
    36: "left_ring3",
    37: "left_thumb1",
    38: "left_thumb2",
    39: "left_thumb3",
    40: "right_index1",
    41: "right_index2",
    42: "right_index3",
    43: "right_middle1",
    44: "right_middle2",
    45: "right_middle3",
    46: "right_pinky1",
    47: "right_pinky2",
    48: "right_pinky3",
    49: "right_ring1",
    50: "right_ring2",
    51: "right_ring3",
    52: "right_thumb1",
    53: "right_thumb2",
    54: "right_thumb3",
}
SMPLX_JOINT2IND = {name: ind for ind, name in SMPLX_IND2JOINT.items()}

# Measurements
STANDARD_LABELS = {
    "A": "head circumference",
    "B": "neck circumference",
    "C": "shoulder to crotch height",
    "D": "chest circumference",
    "E": "waist circumference",
    "F": "hip circumference",
    "G": "wrist right circumference",
    "H": "bicep right circumference",
    "I": "forearm right circumference",
    "J": "arm right length",
    "K": "arm left length",
    "L": "inside leg height",
    "M": "thigh left circumference",
    "N": "calf left circumference",
    "O": "ankle left circumference",
    "P": "shoulder breadth",
    "Q": "height",
}


class MeasurementType:
    CIRCUMFERENCE = "circumference"
    LENGTH = "length"


MEASUREMENT_TYPES = {
    "height": MeasurementType.LENGTH,
    "head circumference": MeasurementType.CIRCUMFERENCE,
    "neck circumference": MeasurementType.CIRCUMFERENCE,
    "shoulder to crotch height": MeasurementType.LENGTH,
    "chest circumference": MeasurementType.CIRCUMFERENCE,
    "waist circumference": MeasurementType.CIRCUMFERENCE,
    "hip circumference": MeasurementType.CIRCUMFERENCE,
    "wrist right circumference": MeasurementType.CIRCUMFERENCE,
    "bicep right circumference": MeasurementType.CIRCUMFERENCE,
    "forearm right circumference": MeasurementType.CIRCUMFERENCE,
    "arm right length": MeasurementType.LENGTH,
    "arm left length": MeasurementType.LENGTH,
    "inside leg height": MeasurementType.LENGTH,
    "thigh left circumference": MeasurementType.CIRCUMFERENCE,
    "calf left circumference": MeasurementType.CIRCUMFERENCE,
    "ankle left circumference": MeasurementType.CIRCUMFERENCE,
    "shoulder breadth": MeasurementType.LENGTH,
    "arm length (shoulder to elbow)": MeasurementType.LENGTH,
    "arm length (spine to wrist)": MeasurementType.LENGTH,
    "crotch height": MeasurementType.LENGTH,
    "Hip circumference max height": MeasurementType.LENGTH,
}


class SMPLXMeasurementDefinitions:
    """
    Definition of SMPLX measurements.

    To add a new measurement:
    1. add it to the measurement_types dict and set the type:
       LENGTH or CIRCUMFERENCE
    2. depending on the type, define the measurement in LENGTHS or
       CIRCUMFERENCES dict
       - LENGTHS are defined using 2 landmarks - the measurement is
                found with distance between landmarks
       - CIRCUMFERENCES are defined with landmarks and joints - the
                measurement is found by cutting the SMPLX model with the
                plane defined by a point (landmark point) and normal (
                vector connecting the two joints)
    3. If the body part is a CIRCUMFERENCE, a possible issue that arises is
       that the plane cutting results in multiple body part slices. To alleviate
       that, define the body part where the measurement should be located in
       CIRCUMFERENCE_TO_BODYPARTS dict. This way, only slice in that body part is
       used for finding the measurement. The body parts are defined by the SMPL
       face segmentation.
    """

    LENGTHS = {
        "height": (SMPLX_LANDMARK_INDICES["HEAD_TOP"], SMPLX_LANDMARK_INDICES["HEELS"]),
        "shoulder to crotch height": (
            SMPLX_LANDMARK_INDICES["SHOULDER_TOP"],
            SMPLX_LANDMARK_INDICES["INSEAM_POINT"],
        ),
        "arm left length": (
            SMPLX_LANDMARK_INDICES["LEFT_SHOULDER"],
            SMPLX_LANDMARK_INDICES["LEFT_WRIST"],
        ),
        "arm right length": (
            SMPLX_LANDMARK_INDICES["RIGHT_SHOULDER"],
            SMPLX_LANDMARK_INDICES["RIGHT_WRIST"],
        ),
        "inside leg height": (
            SMPLX_LANDMARK_INDICES["LOW_LEFT_HIP"],
            SMPLX_LANDMARK_INDICES["LEFT_ANKLE"],
        ),
        "shoulder breadth": (
            SMPLX_LANDMARK_INDICES["LEFT_SHOULDER"],
            SMPLX_LANDMARK_INDICES["RIGHT_SHOULDER"],
        ),
    }

    # defined with landmarks and joints
    # landmarks are defined with indices of the smpl model points
    # normals are defined with joint names of the smpl model
    CIRCUMFERENCES = {
        "head circumference": {
            "LANDMARKS": ["HEAD_LEFT_TEMPLE"],
            "JOINTS": ["pelvis", "spine3"],
        },
        "neck circumference": {
            "LANDMARKS": ["NECK_ADAM_APPLE"],
            "JOINTS": ["spine1", "spine3"],
        },
        "chest circumference": {
            "LANDMARKS": ["LEFT_NIPPLE", "RIGHT_NIPPLE"],
            "JOINTS": ["pelvis", "spine3"],
        },
        "waist circumference": {
            "LANDMARKS": ["BELLY_BUTTON", "BACK_BELLY_BUTTON"],
            "JOINTS": ["pelvis", "spine3"],
        },
        "hip circumference": {
            "LANDMARKS": ["PUBIC_BONE"],
            "JOINTS": ["pelvis", "spine3"],
        },
        "wrist right circumference": {
            "LANDMARKS": ["RIGHT_WRIST"],
            "JOINTS": ["right_wrist", "right_elbow"],
        },  # different from SMPL
        "bicep right circumference": {
            "LANDMARKS": ["RIGHT_BICEP"],
            "JOINTS": ["right_shoulder", "right_elbow"],
        },
        "forearm right circumference": {
            "LANDMARKS": ["RIGHT_FOREARM"],
            "JOINTS": ["right_elbow", "right_wrist"],
        },
        "thigh left circumference": {
            "LANDMARKS": ["LEFT_THIGH"],
            "JOINTS": ["pelvis", "spine3"],
        },
        "calf left circumference": {
            "LANDMARKS": ["LEFT_CALF"],
            "JOINTS": ["pelvis", "spine3"],
        },
        "ankle left circumference": {
            "LANDMARKS": ["LEFT_ANKLE"],
            "JOINTS": ["pelvis", "spine3"],
        },
    }

    possible_measurements = list(LENGTHS.keys()) + list(CIRCUMFERENCES.keys())

    CIRCUMFERENCE_TO_BODYPARTS = {
        "head circumference": "head",
        "neck circumference": "neck",
        "chest circumference": ["spine1", "spine2"],
        "waist circumference": ["hips", "spine"],
        "hip circumference": "hips",
        "wrist right circumference": ["rightHand", "rightForeArm"],
        "bicep right circumference": "rightArm",
        "forearm right circumference": "rightForeArm",
        "thigh left circumference": "leftUpLeg",
        "calf left circumference": "leftLeg",
        "ankle left circumference": "leftLeg",
    }
