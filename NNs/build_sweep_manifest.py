#!/usr/bin/env python3
# Builds NNs/sweep_manifest.json for run_verification_sweep.py.
#
# Scope, revised down from the original "15 samples/method/model" after
# Phase 4's pre-flight check (see PROGRESS.md's 2026-08-24 entry): only
# InceptionMNIST has more than one genuinely distinct, verifiable
# structure (mnist_cnn_a and resnet2b never produce a Concat/Split under
# any setting tried -- confirmed twice each). With the sample count this
# small, the compute budget comfortably supports the FULL established
# 10-image range at every epsilon point (not the originally-planned
# 4-image reduced subset, which existed only to control cost under a much
# larger assumed sample count) -- worst case is roughly 2 hours, well
# under the ~8-12h approved budget.
#
# Entries:
#   - inception_mnist unfused / fused_v2 (the existing, hand-verified
#     pair from the earlier fused-vs-unfused comparison): full 5-point
#     MNIST-family epsilon grid x 10 images -- the main comparison.
#   - inception_mnist fused_auto (repvar_1, discovered by the NEW
#     automated diverse-sampling pipeline rather than hand-tuning): a
#     single epsilon x 10 images, as a consistency check that the
#     automated pipeline reproduces the same verifiability profile as the
#     hand-found fused_v2 (same structural signature -- same channel-axis
#     single Concat+Split -- so should give the same result; a
#     divergence here would indicate a reconstruction bug worth chasing).
#   - mnist_cnn_a / resnet2b unfused: single epsilon x 10 images each,
#     first-time descriptive baselines (no comparison structure exists).
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NNS = REPO_ROOT / "NNs"

MNIST_FAMILY_GRID = [0.02, 0.05, 0.1, 0.15, 0.2]

entries = [
    {
        "model": "inception_mnist", "method": "unfused", "sample_id": "baseline",
        "onnx_path": str(NNS / "inception_mnist_unfused_simplified.onnx"),
        "base_config": str(NNS / "abcrown_config_inception_mnist_unfused_randombranch.yaml"),
        "epsilons": MNIST_FAMILY_GRID, "start": 0, "end": 10,
    },
    {
        "model": "inception_mnist", "method": "fused_v2", "sample_id": "handverified",
        "onnx_path": str(NNS / "inception_mnist_fused_v2_simplified.onnx"),
        "base_config": str(NNS / "abcrown_config_inception_mnist_fused_v2.yaml"),
        "epsilons": MNIST_FAMILY_GRID, "start": 0, "end": 10,
    },
    {
        "model": "inception_mnist", "method": "fused_auto", "sample_id": "repvar1",
        "onnx_path": str(NNS / "inception_mnist_fused_auto_repvar1_simplified.onnx"),
        "base_config": str(NNS / "abcrown_config_inception_mnist_fused_v2.yaml"),
        "epsilons": [0.1], "start": 0, "end": 10,
    },
    {
        "model": "mnist_cnn_a", "method": "unfused", "sample_id": "baseline",
        "onnx_path": str(NNS / "mnist_cnn_a_regen_simplified.onnx"),
        "base_config": str(NNS / "abcrown_config_mnist_cnn_a_unfused.yaml"),
        "epsilons": [0.1], "start": 0, "end": 10,
    },
    {
        "model": "resnet2b", "method": "unfused", "sample_id": "baseline",
        "onnx_path": str(NNS / "resnet2b_regen_simplified.onnx"),
        "base_config": str(NNS / "abcrown_config_resnet2b_unfused.yaml"),
        "epsilons": [0.031], "start": 0, "end": 10,
    },
]


def main():
    missing = [e for e in entries if not Path(e["onnx_path"]).exists()]
    for e in missing:
        print(f"NOT YET READY: {e['model']}/{e['method']}/{e['sample_id']} -> {e['onnx_path']} (missing)")
    ready = [e for e in entries if e not in missing]
    out_path = NNS / "sweep_manifest.json"
    with open(out_path, "w") as f:
        json.dump(ready, f, indent=2)
    print(f"wrote {out_path}: {len(ready)}/{len(entries)} entries ready "
          f"({sum(len(e['epsilons']) for e in ready)} total runs planned)")


if __name__ == "__main__":
    main()
