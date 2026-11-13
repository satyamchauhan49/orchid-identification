import argparse
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np

def phash_int(image):
    """Compute a 64-bit perceptual hash using an 8x8 low-frequency DCT."""
    import cv2
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)[:8, :8]
    values = dct.flatten()
    median = np.median(values[1:])
    bits = values > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset-root",required=True)
    p.add_argument("--out",default="outputs")
    p.add_argument("--threshold",type=int,default=4,help="Maximum pHash Hamming distance")
    a=p.parse_args()
    root,out=Path(a.dataset_root),Path(a.out); out.mkdir(parents=True,exist_ok=True)
    imgdir=root/"Orchid_Images"
    hashes=[]
    for path in tqdm(sorted(imgdir.iterdir()),desc="Hashing images"):
        if path.suffix.lower() not in {".jpg",".jpeg",".png",".jfif"}: continue
        try:
            with Image.open(path) as im: h=phash_int(im)
            hashes.append((path.name,h))
        except Exception as e:
            print(f"Skipping {path.name}: {e}")
    # Exact/near-duplicate comparisons. Dataset has 7k images; this simple O(n^2)
    # implementation is transparent but may take a few minutes.
    parsed=hashes
    rows=[]
    for i,(name_i,hash_i) in enumerate(tqdm(parsed,desc="Comparing hashes")):
        for name_j,hash_j in parsed[i+1:]:
            dist=(hash_i ^ hash_j).bit_count()
            if dist <= a.threshold:
                rows.append({"image_a":name_i,"image_b":name_j,"hamming_distance":int(dist)})
    pd.DataFrame(rows,columns=["image_a","image_b","hamming_distance"]).to_csv(out/"duplicate_pairs.csv",index=False)
    print(f"Images hashed: {len(parsed)} | candidate duplicate pairs: {len(rows)}")
    print(f"Saved {out/'duplicate_pairs.csv'}")
if __name__=="__main__": main()
