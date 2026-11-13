import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

FEATURES=["T","LC","In","NF","CL","CF"]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",default="outputs/split_manifest.csv")
    p.add_argument("--out",default="outputs/eda")
    a=p.parse_args()
    df=pd.read_csv(a.manifest); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    summary=[]
    for col in ["species"]+FEATURES:
        counts=df[col].value_counts(dropna=False)
        counts.rename_axis(col).reset_index(name="count").to_csv(out/f"{col}_counts.csv",index=False)
        if col!="species":
            plt.figure(figsize=(10,5))
            counts.sort_values().plot(kind="barh")
            plt.title(f"Class distribution: {col}")
            plt.xlabel("Number of images"); plt.tight_layout()
            plt.savefig(out/f"{col}_distribution.png",dpi=180); plt.close()
        summary.append({"label":col,"classes":int(df[col].nunique()),"missing":int(df[col].isna().sum()),"min_class_count":int(counts.min()),"max_class_count":int(counts.max())})
    pd.DataFrame(summary).to_csv(out/"eda_summary.csv",index=False)
    print(pd.DataFrame(summary).to_string(index=False))
if __name__=="__main__": main()
