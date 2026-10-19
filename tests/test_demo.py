"""Adapter and episode contracts. Learned perception is verified separately with the fixed real bundle."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from monodist.demo import Camera, Session, observations
from monodist.simulation import scenario


def test_control_observations_use_predictions_and_camera_only():
    predicted = [{'track':9,'filtered_depth_m':20.,'closing_speed_mps':3.,
                  'box':[660,160,800,240],'class':'Car'}]
    obs = observations(predicted,(700.,600.,180.))[0]
    assert obs.id == 9 and obs.depth == 20 and obs.closing_speed == 3
    assert obs.x == pytest.approx(130*20/700)
    assert obs.width == pytest.approx(4.)
    assert observations([], (700.,600.,180.)) == []  # missing detections cannot acquire truth substitutes


def test_camera_changes_with_ego_position_and_clips_near_billboards(tmp_path):
    paths = [tmp_path/'000003.png',tmp_path/'000077.png']
    photo = np.zeros((370,1224,3),np.uint8)
    photo[179:215,496:606] = 255
    cv2.imwrite(str(paths[0]),photo)
    cv2.imwrite(str(paths[1]),np.zeros_like(photo))
    camera = Camera(SimpleNamespace(images=paths,camera=(700.,600.,180.),
                                   meta={'constants':{'class_height':[1.5]}}))
    world = scenario('stationary_obstacle')
    first = camera.render(world)
    world.position = 20
    assert np.sum(camera.render(world)) > np.sum(first)
    world.position = 40.9
    assert camera.render(world).shape == first.shape  # near edge clipping does not fail
    world.position = 43
    assert not camera.render(world).any()  # passed obstacle is behind camera


def test_reset_discards_preview_tracking_state_and_episode_is_bounded():
    class Pipeline:
        def __init__(self): self.frame = 100
        def reset(self): self.frame = 0
    class FixtureSession(Session):
        def capture(self,advance=True):
            assert advance is False
            self.pipeline.frame += 1
            return {'preview':True}
    session = FixtureSession.__new__(FixtureSession)
    session.pipeline = Pipeline()
    assert session.reset('stationary_obstacle',True) == {'preview':True}
    assert session.pipeline.frame == 0 and session.steps == 0
    assert session.world.obstacles[0].width == 4.6
    for name, control in [('bogus',True),('stationary_obstacle','true')]:
        with pytest.raises(ValueError): session.reset(name,control)
    session.world = scenario('stationary_obstacle')
    session.steps = 120
    with pytest.raises(ValueError,match='finished'):
        Session.capture(session)  # no additional model inference beyond the episode deadline
