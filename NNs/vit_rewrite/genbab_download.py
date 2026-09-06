#!/usr/bin/env python
"""Download the GenBaB (Shi et al., TACAS 2025) CIFAR-10 ViT benchmarks from HuggingFace: 4 PGD-trained ViTs (model.pth,
vnnlib specs, instances.csv, abcrown config) + the model definitions.  ~105 MB.  Target: genbab_benchmarks/ (git-ignored)."""
import json, urllib.request, os
base = "https://huggingface.co/api/datasets/zhouxingshi/GenBaB/tree/main/"; raw = "https://huggingface.co/datasets/zhouxingshi/GenBaB/resolve/main/"
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../genbab_benchmarks")
n = 0
for d in ["cifar/models", "cifar/vit_1_3", "cifar/vit_1_6", "cifar/vit_2_3", "cifar/vit_2_6"]:
    with urllib.request.urlopen(base + d + "?recursive=true", timeout=60) as r: entries = json.load(r)
    for e in entries:
        if e["type"] != "file": continue
        dst = os.path.join(out, e["path"]); os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst): urllib.request.urlretrieve(raw + e["path"], dst); n += 1
print(f"downloaded {n} new files into {out}")
