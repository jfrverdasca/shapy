from typing import List, NewType, Tuple, Union

from .abstract_structure import AbstractStructure
from .bbox import BoundingBox
from .betas import Betas
from .body_pose import BodyPose
from .expression import Expression
from .global_rot import GlobalRot
from .hand_pose import HandPose
from .image_list import ImageList, ImageListPacked, to_image_list
from .jaw_pose import JawPose
from .joints import Joints
from .keypoints import Keypoints2D, Keypoints3D
from .points_2d import Points2D
from .vertices import Vertices

StructureList = NewType("StructureList", List[AbstractStructure])
