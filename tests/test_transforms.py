"""Tests for the SO(3)/SE(3) transform utilities.

All random samples come from a single seeded generator (``SEED``) so the
suite is deterministic. MuJoCo is used here only as an independent reference
for quaternion conventions; the library module itself does not import it.

Expected singular behavior (see ``rotation_to_rpy``): at gimbal lock
(``pitch = +/- pi/2``) roll and yaw are not unique, and this module pins
``roll = 0`` while absorbing the remaining freedom into ``yaw``. The tests
below record that convention.
"""

import mujoco
import numpy as np
import pytest

from so101_classical_control.kinematics import transforms as tf

SEED = 20261114
SAMPLES = 60
ATOL = 1e-10


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Return a uniformly distributed proper rotation (QR-based, no transforms)."""
    matrix, triangular = np.linalg.qr(rng.normal(size=(3, 3)))
    matrix = matrix @ np.diag(np.sign(np.diag(triangular)))
    if np.linalg.det(matrix) < 0.0:
        matrix[:, 0] = -matrix[:, 0]
    return matrix


def _random_translation(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=3)


def _random_pose(rng: np.random.Generator) -> np.ndarray:
    return tf.pose(_random_rotation(rng), _random_translation(rng))


def _random_quaternion(rng: np.random.Generator) -> np.ndarray:
    quaternion = rng.normal(size=4)
    return quaternion / np.linalg.norm(quaternion)


def _wrap(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


# ---------------------------------------------------------------------------
# Elementary rotations and basic SO(3) properties
# ---------------------------------------------------------------------------


def test_elementary_rotations_match_lecture_matrices() -> None:
    quarter_turn = np.pi / 2.0
    np.testing.assert_allclose(
        tf.rx(quarter_turn), [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], atol=1e-12
    )
    np.testing.assert_allclose(
        tf.ry(quarter_turn), [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], atol=1e-12
    )
    np.testing.assert_allclose(
        tf.rz(quarter_turn), [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], atol=1e-12
    )
    # Ry has -sin in the (3, 1) entry, i.e. index [2, 0].
    assert tf.ry(quarter_turn)[2, 0] < 0.0
    np.testing.assert_allclose(tf.rx(0.0), tf.identity_so3(), atol=0.0)
    np.testing.assert_allclose(tf.rz(np.pi), np.diag([-1.0, -1.0, 1.0]), atol=1e-15)


def test_inverse_and_identity_so3() -> None:
    rng = _rng()
    for _ in range(SAMPLES):
        rotation = _random_rotation(rng)
        assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12)
        inverse = tf.inverse_so3(rotation)
        np.testing.assert_allclose(inverse, rotation.T, atol=0.0)
        np.testing.assert_allclose(inverse @ rotation, tf.identity_so3(), atol=ATOL)
        np.testing.assert_allclose(rotation @ inverse, tf.identity_so3(), atol=ATOL)
        np.testing.assert_allclose(tf.inverse_so3(inverse), rotation, atol=ATOL)


def test_composition_associativity_so3() -> None:
    rng = _rng()
    for _ in range(SAMPLES):
        a = _random_rotation(rng)
        b = _random_rotation(rng)
        c = _random_rotation(rng)
        np.testing.assert_allclose(
            tf.compose_so3_fixed(tf.compose_so3_fixed(a, b), c),
            tf.compose_so3_fixed(a, tf.compose_so3_fixed(b, c)),
            atol=ATOL,
        )
        np.testing.assert_allclose(
            tf.compose_so3_body(tf.compose_so3_body(a, b), c),
            tf.compose_so3_body(a, tf.compose_so3_body(b, c)),
            atol=ATOL,
        )


def test_composition_frame_conventions() -> None:
    # Non-commuting pair: 90 deg about X then 90 deg about Y.
    a = tf.rx(np.pi / 2.0)
    b = tf.ry(np.pi / 2.0)

    # Both helpers are matrix products of their arguments.
    np.testing.assert_allclose(tf.compose_so3_fixed(a, b), a @ b, atol=1e-15)
    np.testing.assert_allclose(tf.compose_so3_body(a, b), a @ b, atol=1e-15)
    assert not np.allclose(tf.compose_so3_fixed(a, b), tf.compose_so3_fixed(b, a), atol=1e-6)

    # Applying a body-frame increment B to pose A is A @ B ...
    np.testing.assert_allclose(tf.compose_so3_body(a, b), a @ b, atol=ATOL)
    # ... while applying a fixed-frame increment B to pose A is B @ A.
    np.testing.assert_allclose(tf.compose_so3_fixed(b, a), b @ a, atol=ATOL)
    assert not np.allclose(tf.compose_so3_body(a, b), tf.compose_so3_fixed(b, a), atol=1e-6)

    ta = tf.pose(a, [1.0, 2.0, 3.0])
    tb = tf.pose(b, [0.5, -1.0, 2.0])
    np.testing.assert_allclose(tf.compose_se3_fixed(ta, tb), ta @ tb, atol=1e-15)
    np.testing.assert_allclose(tf.compose_se3_body(ta, tb), ta @ tb, atol=1e-15)
    assert not np.allclose(tf.compose_se3_fixed(ta, tb), tf.compose_se3_fixed(tb, ta), atol=1e-6)
    np.testing.assert_allclose(tf.compose_se3_body(ta, tb), ta @ tb, atol=ATOL)
    np.testing.assert_allclose(tf.compose_se3_fixed(tb, ta), tb @ ta, atol=ATOL)


# ---------------------------------------------------------------------------
# Quaternions
# ---------------------------------------------------------------------------


def test_quaternion_unit_norm_and_canonical_sign() -> None:
    rng = _rng()
    for _ in range(SAMPLES):
        rotation = _random_rotation(rng)
        quaternion = tf.rotation_to_quaternion(rotation)
        assert quaternion.shape == (4,)
        assert np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-12)
        assert quaternion[0] >= 0.0

        raw = _random_quaternion(rng)
        for candidate in (raw, -raw):
            normalized = tf.normalize_quaternion(candidate)
            assert np.isclose(np.linalg.norm(normalized), 1.0, atol=1e-12)
            assert normalized[0] >= 0.0
            np.testing.assert_allclose(normalized, tf.normalize_quaternion(candidate), atol=0.0)

        inverse = tf.inverse_quaternion(raw)
        assert np.isclose(np.linalg.norm(inverse), 1.0, atol=1e-12)
        assert inverse[0] >= 0.0
        np.testing.assert_allclose(
            tf.compose_quaternion_fixed(inverse, raw), [1.0, 0.0, 0.0, 0.0], atol=ATOL
        )

    # w == 0: the sign is chosen so the first nonzero of (x, y, z) is positive.
    np.testing.assert_allclose(
        tf.normalize_quaternion([0.0, -1.0, 0.0, 0.0]), [0.0, 1.0, 0.0, 0.0], atol=0.0
    )


def test_quaternion_sign_does_not_change_rotation() -> None:
    rng = _rng()
    for _ in range(SAMPLES):
        quaternion = _random_quaternion(rng)
        np.testing.assert_allclose(
            tf.quaternion_to_rotation(quaternion),
            tf.quaternion_to_rotation(-quaternion),
            atol=ATOL,
        )


def test_rotation_quaternion_roundtrip() -> None:
    rng = _rng()
    for _ in range(SAMPLES):
        rotation = _random_rotation(rng)
        quaternion = tf.rotation_to_quaternion(rotation)
        np.testing.assert_allclose(tf.quaternion_to_rotation(quaternion), rotation, atol=ATOL)

        canonical = tf.normalize_quaternion(quaternion)
        np.testing.assert_allclose(
            tf.rotation_to_quaternion(tf.quaternion_to_rotation(canonical)), canonical, atol=ATOL
        )


def test_quaternion_composition_matches_so3() -> None:
    rng = _rng()
    for _ in range(SAMPLES):
        qa = _random_quaternion(rng)
        qb = _random_quaternion(rng)
        ra = tf.quaternion_to_rotation(qa)
        rb = tf.quaternion_to_rotation(qb)
        np.testing.assert_allclose(
            tf.quaternion_to_rotation(tf.compose_quaternion_fixed(qa, qb)),
            tf.compose_so3_fixed(ra, rb),
            atol=ATOL,
        )
        np.testing.assert_allclose(
            tf.quaternion_to_rotation(tf.compose_quaternion_body(qa, qb)),
            tf.compose_so3_body(ra, rb),
            atol=ATOL,
        )


# ---------------------------------------------------------------------------
# Axis-angle
# ---------------------------------------------------------------------------


def test_rotation_axis_angle_roundtrip() -> None:
    rng = _rng()
    for _ in range(SAMPLES):
        rotation = _random_rotation(rng)
        axis, angle = tf.rotation_to_axis_angle(rotation)
        assert axis.shape == (3,)
        assert np.isclose(np.linalg.norm(axis), 1.0, atol=1e-12)
        assert 0.0 <= angle <= np.pi
        np.testing.assert_allclose(tf.axis_angle_to_rotation(axis, angle), rotation, atol=ATOL)


def test_axis_angle_at_pi_does_not_raise() -> None:
    axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0),
        np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8]),
        np.array([-1.0, -1.0, -1.0]) / np.sqrt(3.0),
    ]
    for axis in axes:
        rotation = tf.axis_angle_to_rotation(axis, np.pi)
        recovered_axis, recovered_angle = tf.rotation_to_axis_angle(rotation)
        assert np.isclose(recovered_angle, np.pi, atol=1e-9)
        first_nonzero = recovered_axis[np.flatnonzero(recovered_axis != 0.0)[0]]
        assert first_nonzero > 0.0
        np.testing.assert_allclose(
            tf.axis_angle_to_rotation(recovered_axis, recovered_angle), rotation, atol=1e-9
        )


def test_axis_angle_at_pi_with_mixed_sign_y_axis() -> None:
    """At pi the axis comes from the symmetric part even for a mixed-sign axis.

    The largest component of this axis is Y, and the recovered axis follows the
    module's conventional sign (first nonzero component positive).
    """
    axis = np.array([-0.3, 0.8, -0.5])
    axis = axis / np.linalg.norm(axis)
    assert np.argmax(np.abs(axis)) == 1

    rotation = tf.axis_angle_to_rotation(axis, np.pi)
    recovered_axis, recovered_angle = tf.rotation_to_axis_angle(rotation)

    assert np.isclose(recovered_angle, np.pi, atol=1e-12)
    first_nonzero = recovered_axis[np.flatnonzero(recovered_axis != 0.0)[0]]
    assert first_nonzero > 0.0
    np.testing.assert_allclose(
        tf.axis_angle_to_rotation(recovered_axis, recovered_angle), rotation, atol=1e-9
    )


def test_axis_angle_zero_uses_conventional_axis() -> None:
    axis, angle = tf.rotation_to_axis_angle(tf.identity_so3())
    np.testing.assert_allclose(axis, [1.0, 0.0, 0.0], atol=0.0)
    assert angle == 0.0


# ---------------------------------------------------------------------------
# RPY and ZYZ
# ---------------------------------------------------------------------------


def test_rpy_roundtrip() -> None:
    rng = _rng()
    for _ in range(SAMPLES):
        roll, pitch, yaw = rng.uniform(-np.pi, np.pi, size=3)
        pitch = rng.uniform(-1.4, 1.4)
        rotation = tf.rpy_to_rotation(roll, pitch, yaw)

        # RPY is defined as Rz(yaw) @ Ry(pitch) @ Rx(roll).
        np.testing.assert_allclose(rotation, tf.rz(yaw) @ tf.ry(pitch) @ tf.rx(roll), atol=0.0)

        recovered = tf.rotation_to_rpy(rotation)
        assert -np.pi / 2.0 <= recovered[1] <= np.pi / 2.0
        np.testing.assert_allclose(tf.rpy_to_rotation(*recovered), rotation, atol=ATOL)


def test_zyz_roundtrip() -> None:
    rng = _rng()
    for _ in range(SAMPLES):
        phi, psi = rng.uniform(-np.pi, np.pi, size=2)
        theta = rng.uniform(0.2, np.pi - 0.2)
        rotation = tf.zyz_to_rotation(phi, theta, psi)
        np.testing.assert_allclose(rotation, tf.rz(phi) @ tf.ry(theta) @ tf.rz(psi), atol=0.0)

        recovered_phi, recovered_theta, recovered_psi = tf.rotation_to_zyz(rotation)
        assert 0.0 <= recovered_theta <= np.pi
        np.testing.assert_allclose(
            tf.zyz_to_rotation(recovered_phi, recovered_theta, recovered_psi),
            rotation,
            atol=ATOL,
        )


@pytest.mark.parametrize("pitch", [np.pi / 2.0, -np.pi / 2.0])
def test_gimbal_lock_pins_roll_to_zero(pitch: float) -> None:
    """At pitch = +/- pi/2, roll is pinned to 0 and yaw absorbs the freedom."""
    rng = _rng()
    for _ in range(SAMPLES):
        roll, yaw = rng.uniform(-np.pi, np.pi, size=2)
        rotation = tf.rpy_to_rotation(roll, pitch, yaw)

        recovered_roll, recovered_pitch, recovered_yaw = tf.rotation_to_rpy(rotation)
        assert recovered_roll == 0.0
        assert np.isclose(recovered_pitch, pitch, atol=1e-12)
        np.testing.assert_allclose(
            tf.rpy_to_rotation(recovered_roll, recovered_pitch, recovered_yaw),
            rotation,
            atol=ATOL,
        )

        # The remaining freedom lands in yaw: yaw - roll for +pi/2, yaw + roll
        # for -pi/2, both wrapped to the principal range.
        expected_yaw = yaw - roll if pitch > 0.0 else yaw + roll
        assert np.isclose(_wrap(recovered_yaw), _wrap(expected_yaw), atol=1e-9)


@pytest.mark.parametrize("pitch", [np.pi / 2.0, -np.pi / 2.0])
def test_gimbal_lock_on_directly_built_matrix(pitch: float) -> None:
    """A locked pose assembled from matrices also reports roll = 0."""
    rotation = tf.rz(0.37) @ tf.ry(pitch) @ tf.rx(0.91)
    recovered_roll, recovered_pitch, recovered_yaw = tf.rotation_to_rpy(rotation)
    assert recovered_roll == 0.0
    assert np.isclose(recovered_pitch, pitch, atol=1e-12)
    np.testing.assert_allclose(
        tf.rpy_to_rotation(recovered_roll, recovered_pitch, recovered_yaw), rotation, atol=ATOL
    )


# ---------------------------------------------------------------------------
# SE(3)
# ---------------------------------------------------------------------------


def test_se3_identity_and_inverse() -> None:
    rng = _rng()
    for _ in range(SAMPLES):
        transform = _random_pose(rng)
        inverse = tf.inverse_se3(transform)

        np.testing.assert_allclose(inverse @ transform, tf.identity_se3(), atol=ATOL)
        np.testing.assert_allclose(transform @ inverse, tf.identity_se3(), atol=ATOL)
        np.testing.assert_allclose(tf.inverse_se3(inverse), transform, atol=ATOL)

        # Block form [R.T, -R.T @ p], not a numeric inversion.
        rotation = tf.rotation_of(transform)
        translation = tf.translation_of(transform)
        np.testing.assert_allclose(tf.rotation_of(inverse), rotation.T, atol=0.0)
        np.testing.assert_allclose(tf.translation_of(inverse), -rotation.T @ translation, atol=ATOL)


def test_se3_composition_associativity() -> None:
    rng = _rng()
    for _ in range(SAMPLES):
        a = _random_pose(rng)
        b = _random_pose(rng)
        c = _random_pose(rng)
        np.testing.assert_allclose(
            tf.compose_se3_fixed(tf.compose_se3_fixed(a, b), c),
            tf.compose_se3_fixed(a, tf.compose_se3_fixed(b, c)),
            atol=ATOL,
        )
        np.testing.assert_allclose(
            tf.compose_se3_body(tf.compose_se3_body(a, b), c),
            tf.compose_se3_body(a, tf.compose_se3_body(b, c)),
            atol=ATOL,
        )


def test_pose_and_transform_point() -> None:
    transform = tf.pose(tf.rz(np.pi / 2.0), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(tf.rotation_of(transform), tf.rz(np.pi / 2.0), atol=0.0)
    np.testing.assert_allclose(tf.translation_of(transform), [1.0, 2.0, 3.0], atol=0.0)
    np.testing.assert_allclose(tf.transform_point(transform, [1.0, 0.0, 0.0]), [1.0, 3.0, 3.0],
                               atol=1e-15)

    rng = _rng()
    for _ in range(SAMPLES):
        pose = _random_pose(rng)
        point = _random_translation(rng)
        np.testing.assert_allclose(
            tf.transform_point(pose, point),
            tf.rotation_of(pose) @ point + tf.translation_of(pose),
            atol=ATOL,
        )


# ---------------------------------------------------------------------------
# MuJoCo cross-checks
# ---------------------------------------------------------------------------


def test_mujoco_quaternion_conventions() -> None:
    rng = _rng()
    for _ in range(SAMPLES):
        rotation = _random_rotation(rng)
        quaternion = tf.rotation_to_quaternion(rotation)

        mujoco_matrix = np.zeros(9)
        mujoco.mju_quat2Mat(mujoco_matrix, quaternion)
        np.testing.assert_allclose(mujoco_matrix.reshape(3, 3), rotation, atol=ATOL)

        mujoco_quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(mujoco_quaternion, rotation.reshape(9))
        # MuJoCo may return the opposite sign; canonicalize before comparing.
        np.testing.assert_allclose(
            tf.normalize_quaternion(mujoco_quaternion), quaternion, atol=1e-9
        )
        np.testing.assert_allclose(
            tf.quaternion_to_rotation(mujoco_quaternion), rotation, atol=ATOL
        )


def test_mujoco_axis_angle_and_quaternion_product() -> None:
    rng = _rng()
    for _ in range(SAMPLES):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(0.0, np.pi)

        mujoco_quaternion = np.zeros(4)
        mujoco.mju_axisAngle2Quat(mujoco_quaternion, axis, angle)
        np.testing.assert_allclose(
            tf.quaternion_to_rotation(mujoco_quaternion),
            tf.axis_angle_to_rotation(axis, angle),
            atol=ATOL,
        )

        qa = _random_quaternion(rng)
        qb = _random_quaternion(rng)
        mujoco_product = np.zeros(4)
        mujoco.mju_mulQuat(mujoco_product, qa, qb)
        np.testing.assert_allclose(
            tf.compose_quaternion_fixed(qa, qb),
            tf.normalize_quaternion(mujoco_product),
            atol=1e-9,
        )


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda: tf.inverse_so3(np.eye(2)),
        lambda: tf.inverse_so3(np.zeros((3, 3))),
        lambda: tf.inverse_so3(np.full((3, 3), np.nan)),
        lambda: tf.rx(np.nan),
        lambda: tf.rz(np.inf),
        lambda: tf.quaternion_to_rotation(np.zeros(3)),
        lambda: tf.quaternion_to_rotation(np.zeros(4)),
        lambda: tf.normalize_quaternion(np.zeros(4)),
        lambda: tf.rotation_to_quaternion(np.zeros((3, 3))),
        lambda: tf.rotation_to_quaternion(np.eye(2)),
        lambda: tf.axis_angle_to_rotation(np.zeros(3), 1.0),
        lambda: tf.axis_angle_to_rotation(np.eye(2), 1.0),
        lambda: tf.axis_angle_to_rotation([1.0, 0.0, 0.0], np.nan),
        lambda: tf.rotation_to_axis_angle(np.eye(3) * 2.0),
        lambda: tf.rotation_to_rpy(np.eye(2)),
        lambda: tf.rotation_to_zyz(np.zeros((3, 3))),
        lambda: tf.rpy_to_rotation(np.nan, 0.0, 0.0),
        lambda: tf.zyz_to_rotation(0.0, np.inf, 0.0),
        lambda: tf.pose(np.eye(2), np.zeros(3)),
        lambda: tf.pose(np.eye(3), np.zeros(4)),
        lambda: tf.pose(np.zeros((3, 3)), np.zeros(3)),
        lambda: tf.inverse_se3(np.eye(3)),
        lambda: tf.inverse_se3(np.eye(4) * 2.0),
        lambda: tf.rotation_of(np.eye(3)),
        lambda: tf.translation_of(np.eye(3)),
        lambda: tf.transform_point(np.eye(3), np.zeros(3)),
        lambda: tf.transform_point(np.eye(4), np.zeros(4)),
    ],
)
def test_invalid_inputs_raise_value_error(call) -> None:
    with pytest.raises(ValueError):
        call()
