"""Adapt C's verified validation/test artifacts to B's explanation exchange format."""
import argparse
import json
from pathlib import Path
import numpy as np
from bn_resolver.contracts import digest, load_maps, read_manifest, load_bundle, FEATURES
from bn_resolver.pipeline import predict
from bn_resolver.core import softmax, resolve


def convert(maps_path, manifest_path, bundle_path, artifact_path, evaluation_path, output, checkpoint,
            limit=10, locked_test=False):
    if limit < 1: raise ValueError("limit must be positive")
    maps = load_maps(maps_path)
    rows = read_manifest(manifest_path, maps)
    artifact = json.loads(Path(artifact_path).read_text())
    evaluation = json.loads(Path(evaluation_path).read_text())
    split = evaluation["split"]
    if split == "test" and not locked_test:
        raise ValueError("Real/test artifact conversion requires --locked-test-approved after Sync 4")
    if split not in ("val", "test"): raise ValueError("Only validation/final test evaluations are supported")
    if artifact["label_maps_sha256"] != digest(maps_path) or artifact["manifest_sha256"] != digest(manifest_path):
        raise ValueError("Frozen maps/manifest hash mismatch")
    if evaluation["artifact_sha256"] != digest(artifact_path) or evaluation["bundle_sha256"] != digest(bundle_path):
        raise ValueError("Evaluation does not belong to this artifact and bundle")
    ids, arrays = load_bundle(bundle_path, rows, maps, split, digest(maps_path))
    with np.load(bundle_path, allow_pickle=False) as bundle:
        if "checkpoint_sha256" not in bundle or str(bundle["checkpoint_sha256"].item()) != digest(checkpoint):
            raise ValueError("Missing/mismatched checkpoint provenance: re-export from the matching model")
    dnn, bn, _ = predict(arrays, artifact)
    fused, proposed, accepted, stages = resolve(dnn, bn, artifact["policy"]["weight"], artifact["policy"]["threshold"])
    with np.load(str(evaluation_path)+".posteriors.npz", allow_pickle=False) as post:
        if str(post["label_maps_sha256"].item()) != digest(maps_path) or str(post["artifact_sha256"].item()) != digest(artifact_path):
            raise ValueError("Posterior provenance mismatch")
        post_ids = post["ids"].astype(str).tolist()
        if len(set(post_ids)) != len(post_ids) or set(post_ids) != set(ids): raise ValueError("Posterior IDs mismatch")
        order = [post_ids.index(str(i)) for i in ids]
        for key, expected in (("dnn", dnn), ("bn", bn), ("fused", fused)):
            if not np.allclose(post[key][order], expected, atol=1e-7):
                raise ValueError("Posterior values do not match the frozen policy")
    decisions = json.loads(Path(str(evaluation_path)+".decisions.json").read_text())
    by_id = {d["image_id"]: d for d in decisions}
    if len(by_id) != len(decisions) or set(by_id) != set(ids): raise ValueError("Decision IDs mismatch")
    lookup = {r["image_id"]: r for r in rows}
    q = {f: softmax(arrays[f]/artifact["temperatures"][f]["temperature"]) for f in FEATURES}
    stage_map = {"trusted_agreement": "trusted", "auto_resolved": "auto_resolved", "low_confidence": "deferred"}
    # Prefer a representative sample of available stages; never fabricate absent stages.
    selected = []
    for stage in stage_map:
        candidates = np.flatnonzero(stages == stage)
        if len(candidates): selected.append(int(candidates[0]))
    selected = (selected + [i for i in range(len(ids)) if i not in selected])[:limit]
    cases = []
    for j in range(len(ids)):
        decision = by_id[str(ids[j])]
        expected_label = maps["species"][int(proposed[j])] if accepted[j] else None
        if (decision["stage"] != str(stages[j]) or decision["accepted_species"] != expected_label or
            not np.isclose(decision["confidence"], fused[j].max())):
            raise ValueError("Decision differs from frozen policy")
        if j not in selected: continue
        row = lookup[str(ids[j])]
        cases.append({"image_id": str(ids[j]), "path": row["path"], "split": split,
                      "dnn_probs": dnn[j].tolist(), "bn_probs": bn[j].tolist(),
                      "feature_probs": {f: q[f][j].tolist() for f in FEATURES},
                      "stage": stage_map[str(stages[j])], "fused_confidence": float(fused[j].max()),
                      "final_species": expected_label,
                      "reason": f"Frozen C resolver stage {stages[j]}; weight={artifact['policy']['weight']}; threshold={artifact['policy']['threshold']}."})
    demo = all(s.startswith("SYNTHETIC_") for s in maps["species"])
    output = Path(output)
    if output.exists() and any(output.iterdir()): raise ValueError("Adapter output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"resolver": "c-resolver", "label_maps_sha256": digest(maps_path),
                  "artifact_sha256": digest(artifact_path), "bundle_sha256": digest(bundle_path),
                  "checkpoint_sha256": digest(checkpoint)}
    payload = {"mode": "demo" if demo else "real", "label_maps": maps, "cases": cases, "provenance": provenance}
    exchange = output / "exchange.json"
    exchange.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    taxonomy = {name: {f: dict(zip(maps["features"][f], artifact["theta"][f][i])) for f in FEATURES}
                for i, name in enumerate(maps["species"])}
    (output / "taxonomy.json").write_text(json.dumps(taxonomy, indent=2), encoding="utf-8")
    print(f"Converted {len(cases)} actual resolver decisions; present stages: {sorted(set(c['stage'] for c in cases))}")
    return exchange


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("maps", "manifest", "bundle", "artifact", "evaluation", "output", "checkpoint"):
        p.add_argument("--"+key, required=True)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--locked-test-approved", action="store_true")
    a = p.parse_args()
    convert(a.maps, a.manifest, a.bundle, a.artifact, a.evaluation, a.output, a.checkpoint, a.limit, a.locked_test_approved)
