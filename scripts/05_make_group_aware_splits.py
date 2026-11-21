"""Create leakage-aware train/calib/val/test splits using pHash connected components.

The 156 species are retained. Calibration is carved from the non-test, non-validation
portion and is approximately 10% of the remaining training pool.
"""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='outputs/split_manifest.csv')
    ap.add_argument('--duplicates', default='outputs/duplicate_pairs.csv')
    ap.add_argument('--out', default='outputs')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.manifest)
    df['image_id'] = df['image_id'].astype(str)
    df['original_split'] = df['split'].astype(str)

    # Union-find: transitive duplicate relationships become one indivisible group.
    parent = {x: x for x in df['image_id']}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    dup = pd.read_csv(args.duplicates)
    for r in dup.itertuples(index=False):
        union(str(r.image_a), str(r.image_b))

    roots = {x: find(x) for x in df['image_id']}
    # Stable compact cluster IDs; singleton images receive their own group.
    root_to_id = {r: f'phash_{i:05d}' for i, r in enumerate(sorted(set(roots.values())))}
    df['phash_cluster'] = df['image_id'].map(lambda x: root_to_id[roots[x]])
    df['split'] = 'unassigned'

    # Sequential stratified group holdouts: test (~1/9), validation (~1/5 of
    # remainder), calibration (~1/10 of remainder). Remaining samples train.
    def take_fold(frame, n_splits, fold_index, split_name):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed + fold_index)
        X = frame[['image_id']]
        y = frame['species'].astype(str)
        groups = frame['phash_cluster']
        folds = list(splitter.split(X, y, groups))
        chosen = folds[fold_index % n_splits]
        hold_idx = frame.index[chosen[1]]
        df.loc[hold_idx, 'split'] = split_name
        return frame.drop(index=hold_idx)

    pool = df.copy()
    pool = take_fold(pool, 9, 0, 'test')
    pool = take_fold(pool, 5, 1, 'val')
    pool = take_fold(pool, 10, 2, 'calib')
    df.loc[pool.index, 'split'] = 'train'

    # Hard assertions: every duplicate component must have exactly one split.
    cross = df.groupby('phash_cluster')['split'].nunique()
    assert int(cross.max()) == 1, 'A duplicate cluster crosses splits.'
    assert df['species'].nunique() == 156, f"Expected 156 species; found {df['species'].nunique()}"
    assert set(df['split']) == {'train', 'calib', 'val', 'test'}

    df.to_csv(out / 'split_manifest_grouped.csv', index=False)
    summary = df.groupby('split').agg(images=('image_id','size'), species=('species','nunique'), duplicate_groups=('phash_cluster','nunique'))
    summary.to_csv(out / 'grouped_split_summary.csv')
    print(summary.to_string())
    print('\nSpecies represented per split (species with fewer than 3 images may not occur everywhere):')
    for s, g in df.groupby('split'):
        print(f"{s}: {g['species'].nunique()} species")
    print(f"\nDuplicate groups: {df['phash_cluster'].nunique()} | cross-split groups: {int((cross > 1).sum())}")
    print(f"Saved: {out / 'split_manifest_grouped.csv'}")

if __name__ == '__main__':
    main()
