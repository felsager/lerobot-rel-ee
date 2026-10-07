from enum import StrEnum, auto


class RotationRepresentation(StrEnum):
    quaternion = auto()
    euler_angles = auto()
    rot6d = auto()
    axis_angle_3d = auto()
    axis_angle_4d = auto()
    riemannian_fancy_stuff = auto()


ROT_REPR_DIM = {
    RotationRepresentation.quaternion: 4,
    RotationRepresentation.euler_angles: 3,
    RotationRepresentation.rot6d: 6,
    RotationRepresentation.axis_angle_3d: 3,
    RotationRepresentation.axis_angle_4d: 4,
    RotationRepresentation.riemannian_fancy_stuff: None,
}
"""Number of components in each rotation representation, denoted by 'm' in pose dimensions."""
