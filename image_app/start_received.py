"""Launch the verified received ConvNeXt development upload model locally."""
import argparse
from pathlib import Path
from http.server import HTTPServer
from .server import Predictor,handler


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['convnext_tiny_lr0.0001','efficientnet_b0_lr0.0003'],default='convnext_tiny_lr0.0001')
    p.add_argument('--port',type=int,default=8766)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    source=root/'work/sneha-received'/a.model;analysis=root/'work/sneha-analysis'/a.model
    model=Predictor(analysis/'audited/label_maps.json',source/'best.pt',analysis/'experiments/frozen_direct_soft.json',analysis/'audited/calib.npz',root/'outputs/species_id_to_name.json')
    server=HTTPServer(('127.0.0.1',a.port),handler(model))
    print(f'Real development model: {a.model}. Open http://127.0.0.1:{a.port}. Ctrl+C stops.',flush=True)
    server.serve_forever()

if __name__=='__main__':main()
