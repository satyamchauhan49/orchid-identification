"""Local-only photo upload and frozen model/resolver inference."""
import argparse
import io
import json
import warnings
from pathlib import Path
from http.server import HTTPServer,BaseHTTPRequestHandler
import numpy as np
import torch
from PIL import Image,ImageOps,UnidentifiedImageError
from torchvision import transforms
from explain.checkpoints import load_checkpoint
from models_other.preprocessing import image_transform
from bn_resolver.contracts import load_maps,digest,FEATURES
from bn_resolver.pipeline import predict
from bn_resolver.core import resolve

MAX_BYTES=10*1024*1024


def decode(data):
    if not data or len(data)>MAX_BYTES: raise ValueError('Choose a JPEG or PNG smaller than 10 MB.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as im:
                if im.format not in ('JPEG','PNG'): raise ValueError('Only JPEG and PNG images are supported.')
                if im.width*im.height>20_000_000: raise ValueError('Image must be at most 20 megapixels.')
                im.load()
                return im.convert('RGB')  # Match the checkpoint training dataset's pixel orientation.
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError,Image.DecompressionBombWarning) as e:
        raise ValueError('The file is not a valid supported image.') from e


class Predictor:
    def __init__(self,maps,checkpoint,artifact,bundle,species_names=None):
        torch.set_num_threads(2)
        self.maps=load_maps(maps)
        self.species_names={} if species_names is None else json.loads(Path(species_names).read_text())
        if species_names is not None and (not isinstance(self.species_names,dict) or any(not isinstance(self.species_names.get(k),str) for k in self.maps['species'])):
            raise ValueError('Species display-name map must cover every class ID')
        self.artifact=json.loads(Path(artifact).read_text())
        if self.artifact['label_maps_sha256']!=digest(maps): raise ValueError('Resolver class-map hash mismatch')
        with np.load(bundle,allow_pickle=False) as z:
            for key,value in [('label_maps_sha256',digest(maps)),('checkpoint_sha256',digest(checkpoint)),('manifest_sha256',self.artifact['manifest_sha256'])]:
                if key not in z or str(z[key].item())!=value: raise ValueError('Calibration bundle provenance mismatch: '+key)
            # Verify this checkpoint was the calibration source for the frozen head temperatures.
            expected=set(self.artifact['cal_model_ids'])|set(self.artifact['cal_policy_ids'])
            if set(z['ids'].astype(str))!=expected: raise ValueError('Supply matching calibration export, not validation/test')
        self.model,labels,cfg,*_=load_checkpoint(checkpoint)
        expected={'species':self.maps['species'],**self.maps['features']}
        if labels!=expected: raise ValueError('Checkpoint labels/order mismatch')
        self.demo=bool(cfg.get('dummy') or cfg.get('demo_data'))
        if not cfg.get('mean') or not cfg.get('std'): raise ValueError('Checkpoint lacks train preprocessing statistics')
        self.model.eval()
        self.preprocessing=cfg['preprocessing']
        self.transform=image_transform(cfg['mean'],cfg['std'],self.preprocessing,False)
        self.checkpoint_hash=digest(checkpoint)

    def infer(self,data):
        image=decode(data)
        with torch.inference_mode():
            logits,_=self.model(self.transform(image).unsqueeze(0))
        arrays={k:v.numpy() for k,v in logits.items()}
        d,b,used=predict(arrays,self.artifact)
        policy=self.artifact['policy']
        fused,proposal,accepted,stages=resolve(d,b,policy['weight'],policy['threshold'])
        order=np.argsort(-fused[0],kind='stable')[:5]
        return dict(demo=self.demo,accepted_name=self.species_names.get(self.maps['species'][int(proposal[0])]) if accepted[0] else None,accepted_species=self.maps['species'][int(proposal[0])] if accepted[0] else None,
            stage=str(stages[0]),confidence=float(fused[0].max()),
            candidates=[dict(species=self.maps['species'][i],display_name=self.species_names.get(self.maps['species'][i],self.maps['species'][i]),probability=float(fused[0,i])) for i in order],
            features_used={f:bool(used[f][0]) for f in FEATURES},
            note='Synthetic-trained checkpoint: these are demonstration labels, not real orchid identification.' if self.demo else 'Development model: not yet final-test validated. The nominal 1.5% accepted-error target was not met on validation. Candidates cover trained species only; non-orchids and unknown species are not detected.',
            preprocessing=self.preprocessing,checkpoint_sha256=self.checkpoint_hash)


def handler(predictor):
    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,data,content_type='application/json'):
            payload=json.dumps(data,allow_nan=False).encode() if content_type=='application/json' else data
            self.send_response(status);self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(payload)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' blob:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'")
            self.end_headers();self.wfile.write(payload)
        def do_GET(self):
            if self.path=='/':self.reply(200,Path(__file__).with_name('index.html').read_bytes(),'text/html; charset=utf-8')
            elif self.path=='/api/status':self.reply(200,dict(demo=predictor.demo))
            else:self.reply(404,dict(error='Not found'))
        def do_POST(self):
            if self.path!='/api/predict':return self.reply(404,dict(error='Not found'))
            # Reject cross-site submissions, including ordinary HTML forms.
            expected='http://'+self.headers.get('Host','')
            if self.headers.get('Origin')!=expected:return self.reply(403,dict(error='Open the local upload page first.'))
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=MAX_BYTES:return self.reply(413,dict(error='Choose an image smaller than 10 MB.'))
                if self.headers.get('Content-Type') not in ('image/jpeg','image/png'):raise ValueError('Choose JPEG or PNG.')
                self.connection.settimeout(20)
                self.reply(200,predictor.infer(self.rfile.read(size)))
            except ValueError as e:self.reply(400,dict(error=str(e)))
            except Exception:self.reply(500,dict(error='Prediction failed. Check the local server terminal.'))
        def log_message(self,*args):pass
    return Handler


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('maps','checkpoint','artifact','calibration-bundle'):p.add_argument('--'+key,required=True)
    p.add_argument('--port',type=int,default=8765)
    p.add_argument('--species-names',help='Optional JSON mapping class IDs to botanical display names')
    a=p.parse_args()
    model=Predictor(a.maps,a.checkpoint,a.artifact,a.calibration_bundle,a.species_names)
    server=HTTPServer(('127.0.0.1',a.port),handler(model))
    print(f'Image upload ready: http://127.0.0.1:{a.port} — '+('SYNTHETIC DEMO' if model.demo else 'trained model'),flush=True)
    server.serve_forever()

if __name__=='__main__':main()
