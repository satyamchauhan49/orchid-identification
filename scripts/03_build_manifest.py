import argparse, json
from pathlib import Path
import pandas as pd

FEATURES=["T","LC","In","NF","CL","CF"]

def load(path,split):
    df=pd.read_csv(path,sep=";")
    if "Texture" in df.columns and "T" not in df.columns: df=df.rename(columns={"Texture":"T"})
    if "T" in df.columns: df["T"]=df["T"].replace({"NoSpot":"Nospot"})
    df["split"]=split
    return df

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset-root",required=True)
    p.add_argument("--out",default="outputs")
    a=p.parse_args()
    root,out=Path(a.dataset_root),Path(a.out); out.mkdir(parents=True,exist_ok=True)
    d=root/"Multi-label Classification"
    frames=[
      load(d/"Multilabel_training.csv","train"),
      load(d/"Multilabel_validation.csv","val"),
      load(d/"Multilabel_testing.csv","test")]
    df=pd.concat(frames,ignore_index=True)
    df=df.rename(columns={"Image_Number":"image_id"})
    df["path"]=df["image_id"].map(lambda x: f"Orchid_Images/{x}")
    df["phash_cluster"]=""
    df=df[["image_id","path","Species"]+FEATURES+["phash_cluster","split"]]
    df=df.rename(columns={"Species":"species"})
    df.to_csv(out/"split_manifest.csv",index=False)
    maps={"species":sorted(df.species.dropna().unique().tolist())}
    for f in FEATURES: maps[f]=sorted(df[f].dropna().astype(str).unique().tolist())
    (out/"label_maps.json").write_text(json.dumps(maps,indent=2),encoding="utf-8")
    print(df.groupby("split").size().to_string())
    print(f"Species classes: {len(maps['species'])}")
    print("NOTE: this preserves the dataset's published splits. Review duplicate_pairs.csv before making a group-aware split.")
if __name__=="__main__": main()
