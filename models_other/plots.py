"""Create validation-only comparison, learning-curve and feature-confusion figures."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def create(comparison_path, output):
    path, output = Path(comparison_path), Path(output)
    data = json.loads(path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 140})
    demo = "SYNTHETIC DEMO / " if data["mode"] == "demo" else ""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(data["rows"])), [r["feature_macro_f1"] for r in data["rows"]], color="#42795a")
    ax.set_xticks(range(len(data["rows"])), [r["backbone"]+"\n"+str(r["lr"]) for r in data["rows"]], fontsize=8)
    ax.set_ylabel("Mean feature macro-F1"); ax.set_ylim(0, 1)
    ax.set_title(demo+"Validation backbone comparison"); fig.tight_layout(); fig.savefig(output / "comparison.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    for row in data["rows"]:
        history = json.loads((path.parent / row["run"] / "history.json").read_text())
        ax.plot([h["epoch"] for h in history], [h["val_loss"] for h in history], marker="o", label=row["run"])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation loss"); ax.set_title(demo+"Checkpoint selection curves")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(output / "learning_curves.png"); plt.close(fig)
    metrics = next(d["metrics"] for d in data["details"] if d["run"] == data["selected_run"])
    for feature in ("T", "NF", "In", "LC", "CF", "CL"):
        item = metrics[feature]
        if not item["labelled_count"]: continue
        names = list(item["per_value"])
        matrix = np.asarray(item["confusion_matrix"])
        size = max(5, min(16, len(names)*0.55))
        fig, ax = plt.subplots(figsize=(size, size))
        im = ax.imshow(matrix, cmap="Greens"); fig.colorbar(im, ax=ax, shrink=0.7)
        ax.set_xticks(range(len(names)), names, rotation=75, ha="right", fontsize=7)
        ax.set_yticks(range(len(names)), names, fontsize=7)
        if len(names) <= 12:
            for (i, j), v in np.ndenumerate(matrix): ax.text(j, i, str(v), ha="center", va="center", color="black")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(demo+feature+" / selected validation run")
        fig.tight_layout(); fig.savefig(output / f"confusion_{feature}.png"); plt.close(fig)
