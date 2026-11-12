"""Development-only E3/E4/E5 suite; accepts calibration and validation bundles.

No test bundle argument exists. Synthetic fixtures and real development inputs use
exactly the same contract. Figures/tables are labelled using the map provenance.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from bn_resolver.contracts import (FEATURES, digest, load_maps, read_manifest,
                                   load_bundle, truth, calibration_partition)
from bn_resolver.core import (softmax, fit_temperature, fit_policy, resolve,
                              fuse, reliability, risk_coverage)
from bn_resolver.pipeline import fit, predict, write_json
from .metrics import classification, grouped_bootstrap
from bn_resolver.binary import fit_binary_rates, binary_posterior

VARIANTS = ('hard_direct', 'hard_confusion', 'direct_soft', 'virtual_soft')


def is_synthetic(maps):
    return all(s.startswith('SYNTHETIC_') for s in maps['species'])


def save_table(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score(name, p, y, kind, accepted=None, **extra):
    return dict(name=name, data_kind=kind, **extra,
                **classification(p, y, accepted))


def divide_policy(rows, ids, policy_mask):
    """Keep duplicate groups together in independent fusion and threshold halves."""
    lookup = {r['image_id']: r for r in rows}
    half = np.array([int(hashlib.sha256(
        ('fusion-stage-42:' + lookup[str(i)]['phash_cluster']).encode()
    ).hexdigest(), 16) % 2 == 0 for i in ids])
    return policy_mask & half, policy_mask & ~half


def feature_metrics(arrays, ids, rows, maps, temperatures):
    out = []
    for f in FEATURES:
        y = truth(rows, ids, maps, f)
        keep = y >= 0
        for calibrated in (False, True):
            t = temperatures[f]['temperature'] if calibrated else 1.
            p = softmax(arrays[f] / t)
            pred = p.argmax(1)
            for k, label in enumerate(maps['features'][f]):
                tp = int(((pred == k) & (y == k) & keep).sum())
                fn = int(((pred != k) & (y == k) & keep).sum())
                tn = int(((pred != k) & (y != k) & keep).sum())
                fp = int(((pred == k) & (y != k) & keep).sum())
                out.append(dict(feature=f, value=label, calibrated=calibrated,
                    labelled=int(keep.sum()), positive_support=tp+fn,
                    tpr=tp/(tp+fn) if tp+fn else None,
                    tnr=tn/(tn+fp) if tn+fp else None,
                    tp=tp, fn=fn, tn=tn, fp=fp))
    return out


def explanations(ids, maps, arrays, artifact, dnn, bn, policy):
    """Taxonomic and decision records; no invented images or Grad-CAM maps."""
    fused, proposed, accepted, stages = resolve(
        dnn, bn, policy['weight'], policy['threshold'])
    _, _, used = predict(arrays, artifact)
    q = {f: softmax(arrays[f]/artifact['temperatures'][f]['temperature'])
         for f in FEATURES}
    records = []
    def top(p):
        order = np.argsort(-p, kind='stable')[:5]
        return [dict(species=maps['species'][i], probability=float(p[i])) for i in order]
    for j, image_id in enumerate(ids):
        # Prefer a DNN-versus-BN comparison; otherwise compare proposed and runner-up.
        a, b = int(dnn[j].argmax()), int(bn[j].argmax())
        if a == b:
            a = int(proposed[j])
            b = int(np.argsort(-fused[j], kind='stable')[1])
        features, contrasts = [], []
        for f in FEATURES:
            theta = np.asarray(artifact['theta'][f])
            prior = np.asarray(artifact['feature_priors'][f])
            # Explanation contract for the chosen direct-soft variant.
            likelihood = theta @ (q[f][j] / np.maximum(prior, 1e-12))
            active = bool(used[f][j])
            delta = float(np.log(max(likelihood[a], 1e-12)) -
                          np.log(max(likelihood[b], 1e-12))) if active else 0.
            value = int(q[f][j].argmax())
            features.append(dict(name=f, observed_value=maps['features'][f][value],
                probability=float(q[f][j, value]), used=active,
                expected_distribution=theta[int(proposed[j])].tolist(),
                expected_values=maps['features'][f],
                train_support=artifact['theta_support'][f][int(proposed[j])],
                source=artifact['provenance']))
            contrasts.append(dict(feature=f, log_evidence_delta=delta))
        records.append(dict(image_id=str(image_id), data_kind=(
            'synthetic_demo' if is_synthetic(maps) else 'development_validation'),
            dnn_top5=top(dnn[j]), bn_top5=top(bn[j]), fused_top5=top(fused[j]),
            features=features, contrast_candidates=[maps['species'][a],maps['species'][b]],
            contrast=sorted(contrasts, key=lambda v:-abs(v['log_evidence_delta'])),
            decision=dict(proposed=maps['species'][proposed[j]],
                accepted=maps['species'][proposed[j]] if accepted[j] else None,
                status='accepted' if accepted[j] else 'deferred', stage=str(stages[j]),
                confidence=float(fused[j].max()), weight=policy['weight'],
                threshold=policy['threshold']), visual_paths=[],
            warnings=['No visual explanation generated: image/checkpoint integration belongs to Sneha.',
                      'Feature evidence describes model inference, not botanical ground truth.']))
    # At most 10 unique records; include every observed decision stage first.
    selected = []
    for stage in ('trusted_agreement','auto_resolved','low_confidence'):
        selected.extend([r for r in records if r['decision']['stage']==stage][:3])
    seen = {r['image_id'] for r in selected}
    selected.extend(r for r in records if r['image_id'] not in seen)
    return selected[:10], {s: int((stages==s).sum()) for s in set(stages.tolist())}


def run(args):
    from .plots import render  # fail on missing plotting dependency before writing artifacts
    out = Path(args.output)
    if out.exists():
        raise ValueError('Output directory already exists; use a new run directory')
    maps = load_maps(args.maps)
    rows = read_manifest(args.manifest, maps)
    kind = 'synthetic_demo' if is_synthetic(maps) else 'development_validation'
    if args.synthetic_only and kind != 'synthetic_demo':
        raise ValueError('--synthetic-only requires SYNTHETIC_ species names')
    mh = digest(args.maps)
    cid, ca = load_bundle(args.calib, rows, maps, 'calib', mh)
    vid, va = load_bundle(args.val, rows, maps, 'val', mh)
    cy, vy = truth(rows,cid,maps,'species'), truth(rows,vid,maps,'species')
    if (cy<0).any() or (vy<0).any():
        raise ValueError('Complete development species labels are required')
    cm = calibration_partition(rows, cid)
    fw, ft = divide_policy(rows, cid, ~cm)
    if min(cm.sum(),fw.sum(),ft.sum()) < 20:
        raise ValueError('Need at least 20 rows in each grouped calibration partition')
    out.mkdir(parents=True)
    e3, e4, e5, artifacts = [], [], [], {}
    for variant in VARIANTS:
        path = out / (variant+'.json')
        fit(SimpleNamespace(maps=args.maps,manifest=args.manifest,bundle=args.calib,
                            output=str(path),variant=variant,risk=args.risk,minimum=args.minimum))
        artifact = json.loads(path.read_text())
        artifact['data_kind'] = kind
        write_json(path, artifact)
        artifacts[variant] = artifact
        for mask in (0., .4):
            a = copy.deepcopy(artifact)
            a['missing_confidence'] = mask
            _, p, _ = predict(va,a)
            e3.append(score(variant,p,vy,kind,missing_confidence=mask))
    # Binary selected-detector baseline: rates fitted on cal_model only.
    reference = artifacts['direct_soft']
    cq = {f:softmax(ca[f]/reference['temperatures'][f]['temperature']) for f in FEATURES}
    vq = {f:softmax(va[f]/reference['temperatures'][f]['temperature']) for f in FEATURES}
    binary_rates = {f:fit_binary_rates(cq[f][cm],truth(rows,cid,maps,f)[cm]) for f in FEATURES}
    write_json(out/'binary_rates.json',dict(data_kind=kind,rates=binary_rates,
        fit_ids=cid[cm].tolist(),semantics='Selected category detector=yes; other detectors unobserved',
        limitation='Paper binary CPT mechanism on simplified CLASS-to-feature graph; not full original graph reproduction'))
    for mask in (0.,.4):
        binary_p,_ = binary_posterior(reference['prior'],reference['theta'],vq,binary_rates,mask)
        e3.append(score('binary_tpr_tnr',binary_p,vy,kind,missing_confidence=mask))
    # Prespecified V3 is used downstream; E3 validation winners are not silently
    # retuned on those same validation labels.
    artifact = artifacts['direct_soft']
    dc, bc, _ = predict(ca,artifact)
    dv, bv, _ = predict(va,artifact)
    raw = copy.deepcopy(artifact)
    for temperature in raw['temperatures'].values():
        temperature['temperature']=1.
    _, raw_bn, _ = predict(va,raw)
    raw_dnn = softmax(va['species'])
    bn_t = fit_temperature(np.log(np.maximum(bc[cm],1e-12)),cy[cm])
    bct = softmax(np.log(np.maximum(bc,1e-12))/bn_t['temperature'])
    bvt = softmax(np.log(np.maximum(bv,1e-12))/bn_t['temperature'])
    # Weight/fusion-temperature fitting and threshold selection use disjoint groups.
    policy,_ = fit_policy(dc[fw],bct[fw],cy[fw],args.risk,args.minimum)
    fused_c = fuse(dc,bct,policy['weight'])
    fused_v = fuse(dv,bvt,policy['weight'])
    fusion_t = fit_temperature(np.log(np.maximum(fused_c[fw],1e-12)),cy[fw])
    for name,p in [('dnn_raw',raw_dnn),('dnn_temperature',dv),
                   ('bn_raw_heads',raw_bn),('bn_calibrated_heads',bv),
                   ('bn_temperature',bvt),('fusion_before_temperature',fused_v),
                   ('fusion_temperature',softmax(np.log(np.maximum(fused_v,1e-12))/fusion_t['temperature']))]:
        e4.append(score(name,p,vy,kind))
    # E5 uses head-calibrated DNN and BN (without extra BN/fusion temperature),
    # keeping the frozen base pipeline interoperable with its original CLI.
    policy,_ = fit_policy(dc[fw],bc[fw],cy[fw],args.risk,args.minimum)
    from bn_resolver.core import risk_coverage as curve_fn
    candidate_curve = curve_fn(fuse(dc[ft],bc[ft],policy['weight']),cy[ft])
    feasible = [r for r in candidate_curve if r['accepted']>=args.minimum and r['risk']<=args.risk]
    chosen = max(feasible,key=lambda r:r['accepted']) if feasible else None
    policy.update(threshold=chosen['threshold'] if chosen else None,
                  feasible=chosen is not None,calibration_selection=chosen,
                  limitation='Weight fit and threshold selection use disjoint duplicate groups; empirical selected risk is not guaranteed.')
    artifact['policy']=policy
    artifact['calibration_partitions']={'cal_model':cid[cm].tolist(),
        'cal_fusion':cid[fw].tolist(),'cal_threshold':cid[ft].tolist()}
    artifact['data_kind']=kind
    write_json(out/'frozen_direct_soft.json',artifact)
    f,pred,accepted,stages = resolve(dv,bv,policy['weight'],policy['threshold'])
    rank = (np.argsort(-bv,axis=1,kind='stable')[:,:5]==dv.argmax(1)[:,None]).any(1)
    dcurve = risk_coverage(dc[ft],cy[ft])
    ok = [r for r in dcurve if r['accepted']>=args.minimum and r['risk']<=args.risk]
    dt = max(ok,key=lambda r:r['accepted'])['threshold'] if ok else None
    da = np.zeros(len(vid),dtype=bool) if dt is None else dv.max(1)>=dt
    for name,p,a in [('dnn',dv,None),('bn',bv,None),('rank_only',dv,rank),
                     ('fusion_only',f,None),('fusion_selective',f,accepted),
                     ('dnn_selective',dv,da)]:
        e5.append(score(name,p,vy,kind,a))
    # Manifest may supply a broader independent source group than pHash clusters.
    group_column = getattr(args,'group_column','phash_cluster')
    lookup = {r['image_id']:r for r in rows}
    if any(not lookup[str(i)].get(group_column,'').strip() for i in vid):
        raise ValueError('Missing bootstrap group column: '+group_column)
    groups = np.array([lookup[str(i)][group_column] for i in vid])
    intervals = grouped_bootstrap(vy,dv.argmax(1),pred,groups,accepted)
    write_json(out/'grouped_confidence_intervals.json',dict(data_kind=kind,
        group_column=group_column,**intervals))
    feature_rows = feature_metrics(va,vid,rows,maps,artifact['temperatures'])
    for r in feature_rows:
        r['data_kind']=kind
    save_table(out/'E3_R3_bn_comparison.csv',e3)
    save_table(out/'E4_calibration.csv',e4)
    save_table(out/'E5_R4_resolver.csv',e5)
    save_table(out/'feature_value_metrics.csv',feature_rows)
    explain,counts = explanations(vid,maps,va,artifact,dv,bv,policy)
    write_json(out/'example_records.json',explain)
    write_json(out/'calibration_diagnostics.json',dict(bn=bn_t,fusion=fusion_t,
        warning='BN temperature reuses cal_model used for head temperatures and CPTs; fusion temperature reuses cal_fusion used for weight. Validation is independent. E4 is diagnostic, not the frozen E5 policy.'))
    plot_data = dict(data_kind=kind,
        reliability={n:reliability(p,vy) for n,p in [('DNN raw',raw_dnn),('DNN calibrated',dv),('BN calibrated heads',bv),('Fusion',f)]},
        risk_coverage={n:risk_coverage(p,vy) for n,p in [('DNN',dv),('BN',bv),('Fusion',f)]})
    write_json(out/'plot_data.json',plot_data)
    ledger = dict(data_kind=kind,seed=42,species_count=len(maps['species']),
        inputs={n:dict(path=str(p),sha256=digest(p)) for n,p in
                [('maps',args.maps),('manifest',args.manifest),('calib',args.calib),('val',args.val)]},
        partitions={k:len(v) for k,v in artifact['calibration_partitions'].items()},
        validation_n=len(vid),test_accessed=False,variant_selection='Prespecified direct_soft',
        policy=policy,decision_stages=counts,
        caveats=['Synthetic performance is not orchid research evidence.' if kind=='synthetic_demo' else 'Development analysis only; no test claims.',
                 'V1 is hard-direct, V2 is full categorical confusion; binary_tpr_tnr implements selected-detector virtual evidence on our simplified graph, not the full paper graph.',
                 'Feature metrics are R1-style diagnostics; E2 neural training ablations remain Agrim\'s work.'])
    write_json(out/'run_manifest.json',ledger)
    render(out)
    heading = 'SYNTHETIC DEMONSTRATION - NOT RESEARCH RESULTS' if kind=='synthetic_demo' else 'DEVELOPMENT VALIDATION - NOT FINAL TEST RESULTS'
    text = f'# {heading}\n\n156-class E3/E4/E5 pipeline completed. Test data was not opened.\n\n'
    text += '## Tables\n\n- E3_R3_bn_comparison.csv: four categorical/soft variants plus binary TPR/TNR, masking on/off.\n- E4_calibration.csv: DNN, BN and fusion calibration diagnostics.\n- E5_R4_resolver.csv: full and selective baselines.\n- feature_value_metrics.csv: supported per-value TPR/TNR.\n\n'
    text += 'Grouped paired bootstrap intervals are in grouped_confidence_intervals.json. Whole duplicate/source groups are sampled together; the frozen policy is not refitted.\n\n'
    text += '## Policy and limitations\n\n' + f'Frozen policy feasible: {policy["feasible"]}. Weight: {policy["weight"]}. Threshold: {policy["threshold"]}.\n\n'
    text += 'Temperatures/CPTs use cal_model. Weight uses cal_fusion. Threshold uses cal_threshold. All groups stay together. The E4 BN and fusion temperature diagnostics reuse fitting partitions; validation remains independent. Empirical threshold selection is not a risk guarantee.\n\n'
    text += 'Example records include botanical-distribution evidence, contrastive contributions, and decisions; visual maps await Sneha\'s image/checkpoint integration. See run_manifest.json for hashes and support.\n'
    (out/'RUN_REPORT.md').write_text(text)
    print(f'{heading}\nOutput: {out}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('maps','manifest','calib','val','output'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--synthetic-only',action='store_true')
    p.add_argument('--group-column',default='phash_cluster',help='Independent manifest group for bootstrap, e.g. source_group')
    p.add_argument('--risk',type=float,default=.015)
    p.add_argument('--minimum',type=int,default=50)
    a=p.parse_args()
    if not 0<=a.risk<=1 or a.minimum<1:
        p.error('Risk must be in [0,1] and minimum must be positive')
    run(a)


if __name__=='__main__':
    main()
