#!/usr/bin/env python3
# Phase 7: joins ab-CROWN sweep results with cheap structural features
# (computed directly from each sample's tensat .model file, no ab-CROWN or
# GPU needed) and writes a summary write-up.
#
# Usage: aggregate_sweep_results.py [sweep_results.jsonl] [out.md]
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from structural_signature import analyze

REPO_ROOT = Path(__file__).resolve().parent.parent
NNS = REPO_ROOT / "NNs"
TMP = REPO_ROOT / "tensat" / "tmp"

# (model, method, sample_id) -> .model file used for structural analysis.
# Each maps to the extraction that the corresponding ONNX in
# build_sweep_manifest.py's entries was reconstructed from.
MODEL_FILES = {
    ("inception_mnist", "unfused", "baseline"): TMP / "inception_mnist_prov_start.model",
    ("inception_mnist", "fused_v2", "handverified"): TMP / "inception_mnist_v2_optimized.model",
    ("inception_mnist", "fused_auto", "repvar1"): TMP / "repvar_1_optimized.model",
    ("mnist_cnn_a", "unfused", "baseline"): TMP / "mnist_cnn_a_prov_start.model",
    ("resnet2b", "unfused", "baseline"): TMP / "resnet2b_prov_start.model",
}


def load_results(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def structural_row(key):
    model_file = MODEL_FILES.get(key)
    if model_file is None or not model_file.exists():
        return None
    a = analyze(str(model_file))
    return {
        "node_count": a["node_count"],
        "op_counts": a["op_counts"],
        "max_depth": a["max_depth"],
        "max_branching_factor": a["max_branching_factor"],
        "has_concat_split": bool(a["concat_split_axes"]),
        "concat_split_axes": a["concat_split_axes"],
    }


def main():
    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else NNS / "sweep_results.jsonl"
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else NNS / "sweep_summary.md"

    rows = load_results(results_path)
    print(f"loaded {len(rows)} result rows from {results_path}")

    lines = ["# Structural diversity vs. verifiability -- sweep results\n"]
    by_key = defaultdict(list)
    for row in rows:
        key = (row["model"], row["method"], row["sample_id"])
        by_key[key].append(row)

    for key in sorted(by_key):
        model, method, sample_id = key
        struct = structural_row(key)
        lines.append(f"\n## {model} / {method} / {sample_id}\n")
        if struct:
            lines.append(f"- Structure: {struct['node_count']} nodes, depth {struct['max_depth']}, "
                          f"max branching {struct['max_branching_factor']}, "
                          f"Concat/Split present: {struct['has_concat_split']}")
            if struct["concat_split_axes"]:
                lines.append(f"  - axes: {struct['concat_split_axes']}")
            lines.append(f"  - op counts: {struct['op_counts']}")
        else:
            lines.append("- Structure: (no .model file mapping found)")

        lines.append("\n| epsilon | verified_acc% | verified/total | mean_time(s) | max_time(s) | wall_time(s) |")
        lines.append("|---|---|---|---|---|---|")
        for row in sorted(by_key[key], key=lambda r: r["epsilon"]):
            s = row["summary"]
            acc = s.get("final_verified_acc_pct", "?")
            vs = s.get("verified_safe", "?")
            tot = s.get("instances_count", "?")
            mean_t = s.get("mean_time_all_sec")
            max_t = s.get("max_time_all_sec")
            mean_str = f"{mean_t:.2f}" if mean_t is not None else "?"
            max_str = f"{max_t:.2f}" if max_t is not None else "?"
            lines.append(
                f"| {row['epsilon']} | {acc} | {vs}/{tot} | "
                f"{mean_str} | {max_str} | {row['wall_time_sec']:.1f} |"
            )

    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
