"""Consecutive calibrated video frames through the fixed real pipeline, with wall-clock processing timing.

The saved temporal model assumes 10 Hz. This demo rejects other frame rates or image dimensions rather
than silently reporting incorrect closing speeds/TTC. Playback consumes every frame; no prediction cache.
"""
import argparse
import json
import math
from pathlib import Path
import time

import cv2

from .bundle import Bundle, objects, sha


def annotated(image, predictions):
    image = image.copy()
    for obj in predictions:
        x0,y0,x1,y1 = map(round,obj['box'])
        color = (80,220,100)
        cv2.rectangle(image,(x0,y0),(x1,y1),color,2)
        ttc = '--' if obj.get('ttc_s') is None else f"{obj['ttc_s']:.1f}s"
        text = f"{obj['class']} #{obj.get('track',-1)}  {obj.get('filtered_depth_m',obj['depth_m']):.1f}m  TTC {ttc}"
        cv2.putText(image,text,(max(0,x0),max(16,y0-7)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
    return image


def jpeg(image):
    import base64
    ok, data = cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    return 'data:image/jpeg;base64,'+base64.b64encode(data).decode()


class Video:
    def __init__(self,bundle,path,device='cpu'):
        self.path = Path(path).resolve()
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise ValueError(f'cannot open video: {self.path}')
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.encoded_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.encoded_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reference = cv2.imread(str(bundle.images[0]))
        if not math.isfinite(self.fps) or abs(self.fps-10.0) > 0.01 or self.count < 1:
            self.capture.release()
            raise ValueError('saved tracker/TTC requires a finite 10 FPS video')
        if reference is None:
            self.capture.release()
            raise ValueError('cannot read the calibrated reference image')
        self.height,self.width=reference.shape[:2]
        # Video encoders require even dimensions. Only optional one-pixel bottom/right padding is allowed;
        # discard it before inference to retain the exact calibrated image size and principal point.
        allowed={(self.height,self.width),(self.height+self.height%2,self.width+self.width%2)}
        if (self.encoded_height,self.encoded_width) not in allowed:
            self.capture.release()
            raise ValueError('video dimensions must match the bundle camera/sample images')
        self.bundle,self.device = bundle,device
        self.pipeline = bundle.pipeline(device)
        self.digest = sha(self.path)
        self.rewind()

    def rewind(self):
        if not self.capture.set(cv2.CAP_PROP_POS_FRAMES,0):
            raise RuntimeError('video cannot rewind')
        self.pipeline.reset()
        self.index,self.total_model_ms,self.total_process_ms = 0,0.0,0.0

    def info(self):
        return {'name':self.path.name,'sha256':self.digest,'native_fps':self.fps,'frames':self.count,
                'width':self.width,'height':self.height,'encoded_width':self.encoded_width,'encoded_height':self.encoded_height,'duration_s':self.count/self.fps,'device':self.device,
                'sensor':'Actual YOLO26n + saved distance ensemble + tracker/Kalman/TTC'}

    def reset(self):
        self.rewind()
        preview=self.next()
        preview['preview']=True
        self.rewind()  # preview does not enter the timed playback/tracker sequence
        return preview

    def next(self):
        if self.index >= self.count:
            raise ValueError('video finished; reset to replay')
        started=time.perf_counter()
        ok,image=self.capture.read()
        if not ok:
            raise RuntimeError(f'video decode failed at frame {self.index} of {self.count}')
        image=image[:self.height,:self.width]
        inference=time.perf_counter()
        predictions=objects(self.pipeline([image],self.bundle.camera)[0])
        model_ms=(time.perf_counter()-inference)*1000
        frame=jpeg(annotated(image,predictions))
        process_ms=(time.perf_counter()-started)*1000
        self.total_model_ms+=model_ms
        self.total_process_ms+=process_ms
        index=self.index
        self.index+=1
        return {'info':self.info(),'index':index,'media_time_s':index/self.fps,'done':self.index>=self.count,
                'objects':predictions,'image':frame,'model_ms':model_ms,'processing_ms':process_ms,
                'processed_frames':self.index,'mean_processing_fps':self.index*1000/self.total_process_ms,
                'mean_model_fps':self.index*1000/self.total_model_ms}

    def close(self):
        self.capture.release()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--video',type=Path,required=True)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--frames',type=int)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():
        raise FileExistsError('choose a fresh timing output')
    import torch
    torch.set_num_threads(2)
    clip=Video(Bundle(args.bundle),args.video,args.device)
    target=clip.count if args.frames is None else args.frames
    if not 1<=target<=clip.count:
        raise ValueError('frame count must fit the clip')
    rows=[]
    started=time.perf_counter()
    try:
        for _ in range(target):
            row=clip.next();row.pop('image');rows.append(row)
        seconds=time.perf_counter()-started
        result={'info':clip.info(),'seconds':seconds,'frames':len(rows),'wall_processing_fps':len(rows)/seconds,
                'keeps_native_pace':len(rows)/seconds>=clip.fps,'includes':'decode + actual model + annotation/JPEG + Python result collection; browser/network excluded',
                'rows':rows}
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=1,allow_nan=False))
        print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    finally:clip.close()


if __name__=='__main__':main()
