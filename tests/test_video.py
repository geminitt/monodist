from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from monodist.video import Video


class FixturePipeline:
    """Empty predictions test decoding/timestamp/reset contracts, not learned perception quality."""
    def __init__(self): self.frame=0;self.shapes=[]
    def reset(self): self.frame=0
    def __call__(self,images,camera):
        self.frame+=1;self.shapes.append(images[0].shape[:2])
        return [{'cls':np.empty(0,int),'box':np.empty((0,4)),'score':np.empty(0),
                 'z':np.empty(0),'range':np.empty(0)}]


def fixture(tmp_path,fps=10,source_hw=(60,100),encoded_hw=None):
    sample=tmp_path/'sample.png';cv2.imwrite(str(sample),np.zeros((*source_hw,3),np.uint8))
    pipeline=FixturePipeline()
    bundle=SimpleNamespace(images=[sample],camera=(70.,50.,30.),pipeline=lambda device:pipeline)
    h,w=encoded_hw or source_hw
    path=tmp_path/'clip.mp4';writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    assert writer.isOpened()
    for value in [30,70,100,160]:writer.write(np.full((h,w,3),value,np.uint8))
    writer.release()
    return bundle,path,pipeline


def test_video_consecutive_timestamps_eof_and_reset(tmp_path):
    bundle,path,pipe=fixture(tmp_path)
    clip=Video(bundle,path)
    try:
        rows=[clip.next() for _ in range(4)]
        assert [row['media_time_s'] for row in rows]==[0,.1,.2,.3]
        assert [row['done'] for row in rows]==[False,False,False,True]
        assert pipe.frame==4 and rows[-1]['mean_processing_fps']>0
        with pytest.raises(ValueError,match='finished'):clip.next()
        assert clip.reset()['index']==0 and pipe.frame==0
        assert clip.next()['index']==0 and pipe.frame==1
    finally:clip.close()


def test_video_retains_calibrated_odd_height_by_discarding_only_encoder_padding(tmp_path):
    bundle,path,pipe=fixture(tmp_path,source_hw=(61,100),encoded_hw=(62,100))
    clip=Video(bundle,path)
    try:
        info=clip.info();clip.next()
        assert info['height']==61 and info['encoded_height']==62
        assert pipe.shapes==[(61,100)]
    finally:clip.close()


@pytest.mark.parametrize('fps,encoded_hw,match',[(30,(60,100),'10 FPS'),(10,(60,120),'dimensions')])
def test_video_rejects_unknown_temporal_or_camera_geometry(tmp_path,fps,encoded_hw,match):
    bundle,path,pipe=fixture(tmp_path,fps=fps,encoded_hw=encoded_hw)
    with pytest.raises(ValueError,match=match):Video(bundle,path)
    assert not pipe.shapes
