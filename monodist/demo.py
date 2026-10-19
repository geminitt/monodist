"""Browser-local 2.5D simulation with actual YOLO and saved distance predictions in control.

Run: python -m monodist.demo --bundle /path/to/monodist-demo
World truth enters the camera renderer and diagnostics, never the observation adapter or controller.
"""
import argparse
import base64
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

import cv2
import numpy as np

from .bundle import Bundle, child, objects
from .simulation import Controller, Decision, Observation, scenario

SCENARIOS = {'stationary_obstacle': 'Stationary vehicle', 'lead_brakes': 'Lead vehicle brakes',
             'adjacent_lane': 'Vehicle outside corridor', 'late_obstacle': 'Late obstacle · failure case'}
ASSETS = Path(__file__).parent / 'web/dist'


class Camera:
    """Photo billboards projected by a pinhole camera; fixed background, no 3D lighting/occlusion model."""
    def __init__(self, bundle):
        by_name = {path.name: path for path in bundle.images}
        required = ('000003.png', '000077.png')
        if any(name not in by_name for name in required):
            raise ValueError('the simulation needs KITTI 0012 sample frames 000003.png and 000077.png; build 78 frames')
        self.background = cv2.imread(str(by_name[required[1]]))
        photo = cv2.imread(str(by_name[required[0]]))
        if self.background is None or photo is None:
            raise ValueError('cannot decode simulation assets')
        self.sprite = photo[179:215, 496:606].copy()
        self.camera = bundle.camera
        self.height = float(bundle.meta['constants']['class_height'][0])
        self.camera_height = 1.4

    def render(self, world):
        image = self.background.copy()
        f, cx, cy = self.camera
        ih, iw = image.shape[:2]
        for obj in sorted(world.obstacles, key=lambda obj: obj.z-world.position, reverse=True):
            depth = obj.z-world.position
            if depth < 1.0:
                continue
            h = max(2, round(f*self.height/depth))
            w = max(2, round(h*self.sprite.shape[1]/self.sprite.shape[0]))
            x0 = round(cx + f*obj.x/depth - w/2)
            y1 = round(cy + f*self.camera_height/depth)
            y0 = y1-h
            left, top, right, bottom = max(0,x0), max(0,y0), min(iw,x0+w), min(ih,y1)
            if right <= left or bottom <= top:
                continue
            sprite = cv2.resize(self.sprite,(w,h),interpolation=cv2.INTER_LINEAR)
            image[top:bottom,left:right] = sprite[top-y0:bottom-y0,left-x0:right-x0]
        return image


def observations(predictions, camera):
    """Only model outputs and camera calibration cross this boundary. No world input is accepted."""
    f, cx, _ = camera
    result = []
    for obj in predictions:
        z = obj['filtered_depth_m']
        box = obj['box']
        result.append(Observation(id=obj['track'], depth=z, closing_speed=obj['closing_speed_mps'],
                                  x=((box[0]+box[2])/2-cx)*z/f,
                                  width=max(0.1,(box[2]-box[0])*z/f), kind=obj['class']))
    return result


class Session:
    def __init__(self, bundle, device='cpu'):
        self.bundle = bundle
        self.camera = Camera(bundle)
        self.pipeline = bundle.pipeline(device)
        self.policy = Controller()
        self.reset('stationary_obstacle', True)

    def reset(self, name, control):
        if name not in SCENARIOS or type(control) is not bool:
            raise ValueError('choose a listed scenario and a boolean control setting')
        self.name, self.control = name, control
        self.world = scenario(name)
        # The photographic sprite shows a car side: its width across the camera is its long side.
        for obj in self.world.obstacles:
            obj.width, obj.length = 4.6, 1.8
        self.pipeline.reset()
        self.steps, self.history = 0, []
        preview = self.capture(advance=False)
        self.pipeline.reset()  # preview must not add a duplicate time-zero observation to the live tracker
        return preview

    def capture(self, advance=True):
        if advance and (self.world.collision or self.steps >= 120):
            raise ValueError('episode finished; reset to run again')
        before = self.world.snapshot()
        image = self.camera.render(self.world)
        start = time.perf_counter()
        predicted = objects(self.pipeline([image],self.bundle.camera)[0])
        model_ms = (time.perf_counter()-start)*1000
        seen = observations(predicted,self.bundle.camera)
        decision = self.policy.decide(self.world.speed,seen)
        if not self.control:
            decision = Decision('control off',0.0,decision.target_id,decision.ttc,'Constant ego speed; predictions are still computed')
        if advance:
            self.world.step(decision.acceleration)
            self.steps += 1
        after = self.world.snapshot()
        # Draw only after inference; labels cannot become model inputs.
        for obj in predicted:
            x0,y0,x1,y1 = map(round,obj['box'])
            color = (69,175,27) if obj['track'] != decision.target_id else (0,126,255)
            cv2.rectangle(image,(x0,y0),(x1,y1),color,2)
            label = f"{obj['class']} #{obj['track']}  {obj['filtered_depth_m']:.1f} m"
            cv2.putText(image,label,(max(0,x0),max(16,y0-7)),cv2.FONT_HERSHEY_SIMPLEX,.47,color,1,cv2.LINE_AA)
        ok, encoded = cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])
        if not ok:
            raise RuntimeError('camera JPEG encoding failed')
        row = {'time':before['time'],'speed':before['ego_speed'],
               'true_depth':min((obj['depth'] for obj in before['obstacles']),default=None),
               'predicted_depth':min((obj.depth for obj in seen if obj.id == decision.target_id),default=None),
               'acceleration':decision.acceleration}
        if advance:
            self.history.append(row)
        return {'scenario':self.name,'control':self.control,'capture':before,'after':after,
                'objects':predicted,'observations':[asdict(obj) for obj in seen], 'decision':asdict(decision),
                'image':'data:image/jpeg;base64,'+base64.b64encode(encoded).decode(),
                'model_ms':model_ms,'steps':self.steps,'done':after['collision'] or self.steps>=120,
                'history':self.history[-121:]}


def server(session, port, video=None):
    """Single shared local episode. Lock reset/step so two tabs cannot interleave tracker updates."""
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body, content_type='application/json'):
            payload = body if isinstance(body,bytes) else json.dumps(body,allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(payload)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/api/video/info':
                self.reply(200,video.info() if video else {'error':'Start the server with --video and a calibrated 10 FPS clip'})
                return
            if path == '/api/info':
                self.reply(200,{'scenarios':SCENARIOS,'dt':0.1,'episode_s':12,
                                'sensor':'YOLO26n + saved five-model distance ensemble + Kalman',
                                'shared_session':True})
                return
            files = {'/':('index.html','text/html; charset=utf-8'),
                     '/app.js':('app.js','text/javascript'),'/style.css':('style.css','text/css'),
                     '/video':('video.html','text/html; charset=utf-8'),'/video.js':('video.js','text/javascript')}
            if path not in files:
                self.reply(404,{'error':'Not found'})
                return
            filename, mime = files[path]
            self.reply(200,(ASSETS/filename).read_bytes(),mime)

        def do_POST(self):
            # Same-origin browser only. JSON + no CORS prevents other websites controlling the local episode.
            origin = self.headers.get('Origin')
            if origin and origin not in (f'http://127.0.0.1:{self.server.server_port}',
                                         f'http://localhost:{self.server.server_port}'):
                self.reply(403,{'error':'Use the local demo page'})
                return
            try:
                length = int(self.headers.get('Content-Length','0'))
                if not 0 < length <= 4096 or self.headers.get_content_type() != 'application/json':
                    raise ValueError('request must be a small JSON object')
                data = json.loads(self.rfile.read(length))
                if not isinstance(data,dict):
                    raise ValueError('request must be an object')
                with lock:
                    if self.path in ('/api/video/reset','/api/video/step'):
                        if video is None:
                            raise ValueError('video is not configured')
                        response=video.reset() if self.path.endswith('/reset') else video.next()
                    elif self.path == '/api/reset':
                        response = session.reset(data.get('scenario'),data.get('control'))
                    elif self.path == '/api/step':
                        response = session.capture()
                    else:
                        self.reply(404,{'error':'Not found'})
                        return
                self.reply(200,response)
            except (ValueError,TypeError) as exc:
                self.reply(400,{'error':str(exc)})
            except Exception:
                import traceback
                traceback.print_exc()
                self.reply(500,{'error':'Model/server error; inspect the terminal log'})

    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--port',type=int,default=8765)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--video',type=Path,help='calibrated 10 FPS clip matching bundle camera dimensions')
    args = ap.parse_args()
    import torch
    torch.set_num_threads(2)
    session = Session(Bundle(args.bundle),args.device)
    from .video import Video
    video_path=args.video or (child(session.bundle.root,session.bundle.manifest['video']) if session.bundle.manifest.get('video') else None)
    clip=Video(session.bundle,video_path,args.device) if video_path else None
    app = server(session,args.port,clip)
    print(f'Open http://127.0.0.1:{app.server_port} · actual model inference · Ctrl+C to stop',flush=True)
    try:
        app.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.server_close()
        if clip: clip.close()


if __name__ == '__main__':
    main()
