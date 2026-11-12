"""Generate a local results site, report section and Drive handoff inventory."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np


def build(comparison_path, records_path, output):
    comparison_path, records_path, output = Path(comparison_path), Path(records_path), Path(output)
    comparison = json.loads(comparison_path.read_text())
    records = json.loads(records_path.read_text())
    if comparison["mode"] != records["mode"]:
        raise ValueError("Do not mix demo and real artifacts")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Release output must be a new/empty directory")
    output.mkdir(parents=True)
    mode = comparison["mode"]
    shutil.copytree(records_path.parent, output / "records")
    shutil.copyfile(comparison_path.parent / "comparison.csv", output / "comparison.csv")
    from models_other.plots import create
    create(comparison_path, output / "figures")
    payload = {"mode": mode, "comparison": comparison, "records": records["records"], "provenance": records.get("provenance", {})}
    serialized = json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")
    template = Path("site/template.html").read_text(encoding="utf-8")
    figure_links = '<p><a href="figures/comparison.png">Comparison figure</a> · <a href="figures/learning_curves.png">Learning curves</a> · <a href="figures/confusion_CF.png">CF confusion matrix</a> · <a href="figures/confusion_CL.png">CL confusion matrix</a></p>'
    (output / "index.html").write_text(template.replace("__PAYLOAD__", serialized).replace("Agr im", "Agrim").replace('<div class="pending">', figure_links+'<div class="pending">'), encoding="utf-8")
    selected = comparison_path.parent / comparison["selected_run"]
    config = json.loads((selected / "run_config.json").read_text())
    report = ["# Sneha Malik: model comparison and explainability", "",
              "SYNTHETIC DEMONSTRATION ONLY. No real orchid conclusions." if mode == "demo" else "Validation results. Locked-test analysis remains a separate team stage.",
              "", "## Models and protocol", "EfficientNet-B0 and ConvNeXt-T share a 156-way species head and six feature heads (T, NF, In, LC, CF, CL). Missing labels are masked. Raw logits preserve the calibration interface.",
              f"Seed: {config['seed']}. Resolution: 224 x 224. Optimizer: AdamW. Epochs: {config['epochs']}. Batch size: {config['batch_size']}. Loss: {config['loss']}. Sampler: {config['sampler']}. Pretrained: {config['pretrained']}.",
              "Validation loss selects each run's checkpoint. The grid selects by mean feature macro-F1, breaking ties using validation loss. Final backbone selection still requires calibration and Agrim's ResNet comparison.",
              "", "## Validation comparison", "| Backbone | LR | Species macro-F1 | Feature macro-F1 | Top-1 | Val loss |", "|---|---:|---:|---:|---:|---:|"]
    for row in comparison["rows"]:
        report.append(f"| {row['backbone']} | {row['lr']} | {row['species_macro_f1']:.4f} | {row['feature_macro_f1']:.4f} | {row['species_top1']:.4f} | {row['best_val_loss']:.4f} |")
    report += ["", f"Selected validation run: `{comparison['selected_run']}`.", "", "## R1 contribution: per-value feature metrics",
               "TPR = TP/(TP+FN); TNR = TN/(TN+FP). Undefined rates are null. Macro-F1 averages labels present in predictions or ground truth.",
               "| Run | Feature | Value | Support | TPR | TNR |", "|---|---|---|---:|---:|---:|"]
    for detail in comparison["details"]:
        for feature in ("T", "NF", "In", "LC", "CF", "CL"):
            for value, metric in detail["metrics"][feature]["per_value"].items():
                fmt = lambda x: "undefined" if x is None else f"{x:.4f}"
                report.append(f"| {detail['run']} | {feature} | {value} | {metric['support']} | {fmt(metric['tpr'])} | {fmt(metric['tnr'])} |")
    report += ["", "## Explainability method", "Grad-CAM uses gradients from the predicted species/CF/CL logit at the selected final spatial layer. Heatmaps and the resized input use the same 224-pixel coordinates. Real data uses checkpoint-stored train-only mean/std. Softmax scores are uncalibrated.",
               "Taxonomic records compare observed features to expected feature distributions. Contrastive records rank feature differences by total variation and include candidate references. Features below 0.4 confidence are marked unobserved. Decision records retain the externally supplied stage, confidence and reason.",
               "", "## Example records", "| Image | Stage | DNN candidate | BN candidate |", "|---|---|---|---|"]
    for r in records["records"]:
        report.append(f"| {r['image_id']} | {r['decision']['stage']} | {r['dnn_candidate']} | {r['bn_candidate']} |")
    report += ["", "## Limitations and remaining team work", "Demo scores use artificial images, one epoch and no pretrained weights; they are software checks. Demo resolver cases are simulated independently of model predictions. They establish neither accuracy nor deferral performance.",
               "Real final work requires Agrim's audited splits, labels, final checkpoint and train-only normalization; Satyam's calibrated BN/resolver outputs; a suitable GPU; and human inspection of CL attention on the labellum. Missing references/taxonomy cannot be invented.",
               "Satyam supplies final evaluation JSON, calibration, uncertainty, BN/fusion tables and best-model justification. Review his R4, reliability diagrams and risk-coverage curve; review Agrim's R2. Real test explanations are produced only after Sync 4. Website publication and team merges remain separate actions.",
               "", "## Input provenance", json.dumps(comparison["input_sha256"], indent=2)]
    report_text = "\n".join(report)
    if records.get("provenance", {}).get("resolver") == "c-resolver":
        report_text = report_text.replace("Demo resolver cases are simulated independently of model predictions.",
            "Resolver cases come from the actual frozen C pipeline on B's synthetic image-model outputs. All decisions retain map, model, artifact and bundle provenance. Only observed stages are shown.")
    (output / "report.md").write_text(report_text, encoding="utf-8")
    # Only calib/val are opened here. Test files are inventoried without reading scores.
    inventory = []
    for row in comparison["rows"]:
        directory = comparison_path.parent / row["run"]
        for split in ("calib", "val"):
            with np.load(directory / f"{split}.npz", allow_pickle=False) as data:
                inventory.append({"run": row["run"], "split": split, "rows": len(data["ids"]), "path": str(directory / f"{split}.npz")})
        inventory.append({"run": row["run"], "split": "test", "rows": "locked: not inspected", "exists": (directory / "test.npz").is_file()})
    (output / "handoff_inventory.json").write_text(json.dumps({"mode": mode, "selected_run": comparison["selected_run"], "files": inventory}, indent=2), encoding="utf-8")
    print(f"Local release ready at {output / 'index.html'}")
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", required=True); p.add_argument("--records", required=True); p.add_argument("--output", required=True)
    a = p.parse_args(); build(a.comparison, a.records, a.output)
