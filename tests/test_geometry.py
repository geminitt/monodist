import numpy as np

from monodist import geometry

F, CX, CY, W, H = 720.0, 610.0, 173.0, 1242, 375


def box_for(Z, height, cam_height, u=610.0, width=1.0):
    """The image box of an upright object standing on a flat road under a level pinhole camera."""
    v2 = CY + F * cam_height / Z
    v1 = CY + F * (cam_height - height) / Z
    half = F * width / Z / 2
    return np.array([[u - half, v1, u + half, v2]])


def test_size_distance_inverts_the_projection():
    for Z in (5.0, 17.0, 60.0):
        b = box_for(Z, 1.5, 1.65)
        assert np.allclose(geometry.size_distance(b, np.array([0]), F, [1.5, 1.75]), Z)


def test_ground_distance_inverts_the_projection():
    for Z in (8.0, 30.0, 70.0):
        b = box_for(Z, 1.5, 1.65)
        assert np.allclose(geometry.ground_distance(b, F, CY, 1.65, H), Z)


def test_ground_distance_refuses_cut_boxes_and_boxes_above_the_horizon():
    cut = np.array([[500, 200, 700, H - 1]])
    above = np.array([[500, 100, 700, CY - 5]])
    assert np.isnan(geometry.ground_distance(cut, F, CY, 1.65, H)).all()
    assert np.isnan(geometry.ground_distance(above, F, CY, 1.65, H)).all()
    z, fb = geometry.ground_or_size(np.vstack([cut, box_for(20, 1.5, 1.65)]), np.array([0, 0]), F, CY, 1.65, H, [1.5, 1.75])
    assert fb.tolist() == [True, False] and np.isclose(z[1], 20)


def test_features_are_camera_independent():
    """Same object seen by two cameras with different f: the angle features agree."""
    b1 = box_for(20, 1.5, 1.65)
    f2 = 700.0
    b2 = np.array([[CX + (b1[0, 0] - CX) * f2 / F, CY + (b1[0, 1] - CY) * f2 / F,
                    CX + (b1[0, 2] - CX) * f2 / F, CY + (b1[0, 3] - CY) * f2 / F]])
    x1 = geometry.features(b1, np.array([0]), F, CX, CY, W, H)
    x2 = geometry.features(b2, np.array([0]), f2, CX, CY, W, H)
    assert np.allclose(x1, x2)


def test_cut_flag():
    x = geometry.features(np.array([[0.5, 100, 50, 200], [100, 100, 200, 200]]), np.array([0, 1]), F, CX, CY, W, H)
    assert x[:, geometry.FEATURES.index("cut")].tolist() == [1.0, 0.0]
    assert x[:, geometry.FEATURES.index("pedestrian")].tolist() == [0.0, 1.0]


def test_iou():
    a = np.array([[0, 0, 10, 10]])
    assert geometry.iou(a, a)[0, 0] == 1
    assert np.isclose(geometry.iou(a, [[5, 0, 15, 10]])[0, 0], 1 / 3)
    assert geometry.iou(a, [[20, 20, 30, 30]])[0, 0] == 0
