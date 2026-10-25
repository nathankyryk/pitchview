import cv2
import numpy as np

from pitchview.gsr import Box
from pitchview.pitch import PITCH_KEYPOINTS, fit_homography, keypoint_labels, oracle_homography, to_image, to_pitch

# A plausible broadcast camera: pitch metres -> 1920x1080 image pixels.
PITCH_CORNERS = np.array([[-52.5, -34], [52.5, -34], [52.5, 34], [-52.5, 34]], dtype=np.float64)
IMAGE_CORNERS = np.array([[300, 150], [1650, 150], [2600, 1300], [-700, 1300]], dtype=np.float64)
H_TRUE = cv2.getPerspectiveTransform(PITCH_CORNERS.astype(np.float32), IMAGE_CORNERS.astype(np.float32)).astype(np.float64)
H_IMAGE_TO_PITCH = np.linalg.inv(H_TRUE)


def test_template_is_a_fifa_pitch_in_gsr_coordinates():
    assert PITCH_KEYPOINTS.shape == (32, 2)
    np.testing.assert_allclose(PITCH_KEYPOINTS[0], [-52.5, -34.0])  # far-left corner
    np.testing.assert_allclose(PITCH_KEYPOINTS[8], [-41.5, 0.0])  # left penalty spot, 11 m out
    np.testing.assert_allclose(PITCH_KEYPOINTS[9], [-36.0, -20.16])  # penalty box, 16.5 m deep
    np.testing.assert_allclose(PITCH_KEYPOINTS[30:], [[-9.15, 0.0], [9.15, 0.0]])  # centre circle
    np.testing.assert_allclose(PITCH_KEYPOINTS.sum(axis=0), [0.0, 0.0], atol=1e-9)  # symmetric


def test_oracle_homography_recovers_the_camera():
    rng = np.random.default_rng(0)
    pitch = np.column_stack([rng.uniform(-40, 40, 12), rng.uniform(-30, 30, 12)])
    feet = to_pitch(H_TRUE, pitch)  # H_TRUE maps pitch -> image
    people = [
        Box(frame=1, track_id=i, category_id=1, role="player", team=None, jersey=None,
            x=fx - 10, y=fy - 40, w=20, h=40, pitch_x=px, pitch_y=py)
        for i, ((fx, fy), (px, py)) in enumerate(zip(feet, pitch))
    ]  # fmt: skip
    H = oracle_homography(people)
    np.testing.assert_allclose(to_pitch(H, feet), pitch, atol=1e-6)
    assert oracle_homography(people[:5]) is None  # too few people to trust


def test_keypoint_labels_round_trip_and_fit():
    xy, visible = keypoint_labels(H_IMAGE_TO_PITCH, 1920, 1080)
    np.testing.assert_allclose(to_pitch(H_IMAGE_TO_PITCH, xy), PITCH_KEYPOINTS, atol=1e-6)
    assert visible.any() and not visible.all()
    # perfect keypoints for the visible points give back the same mapping
    conf = visible.astype(float)
    H = fit_homography(xy, conf)
    np.testing.assert_allclose(to_pitch(H, xy[visible]), PITCH_KEYPOINTS[visible], atol=1e-3)
    np.testing.assert_allclose(to_image(H, PITCH_KEYPOINTS[:1]), xy[:1], atol=1e-2)


def test_fit_homography_needs_four_points():
    xy, _ = keypoint_labels(H_IMAGE_TO_PITCH, 1920, 1080)
    conf = np.zeros(32)
    conf[:3] = 1.0
    assert fit_homography(xy, conf) is None
