#!/usr/bin/env python3
# Batch driver for the structural-diversity-vs-verifiability sweep. Runs
# alpha-beta-CROWN once per (manifest entry, epsilon) combination, using
# CLI overrides on top of one base YAML per model (confirmed working:
# alpha-beta-CROWN/complete_verifier/arguments.py's parse_config loads the
# YAML first, then applies CLI args on top -- "commandline arguments have
# the highest priority"). Captures FULL stdout per run (the existing saved
# logs under NNs/ are truncated mid-run -- this driver must not repeat
# that), parses per-image verdicts and the final Summary block, and
# appends one JSON row per run to a results file. Resumable: reruns skip
# any (model, method, sample_id, epsilon, start, end) combination already
# present in the results file.
#
# Usage:
#   run_verification_sweep.py <manifest.json> <results.jsonl> [--dry-run]
#
# Manifest format: a JSON list of entries, each:
#   {"model": "...", "method": "...", "sample_id": "...",
#    "onnx_path": "<absolute path>", "base_config": "<absolute path to a
#    base YAML>", "epsilons": [<float>, ...], "start": 0, "end": 4}
# (epsilons/start/end can differ per entry -- e.g. a "headline" entry uses
# a single epsilon and the full 0:10 image range, while a sweep entry uses
# several epsilons over a 4-image subset.)
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ABCROWN_DIR = REPO_ROOT / "alpha-beta-CROWN" / "complete_verifier"
ABCROWN_CONFIG_DIR = ABCROWN_DIR / "exp_configs" / "beta_crown"
ABCROWN_PYTHON = REPO_ROOT / "alpha-beta-CROWN" / ".venv" / "bin" / "python"

RESULT_RE = re.compile(r"Result:\s+(\S.*?)\s+in\s+([\d.]+)\s+seconds")
SUMMARY_LINE_RES = {
    "final_verified_acc": re.compile(r"Final verified acc:\s+([\d.]+)%\s+\(total (\d+) examples\)"),
    "counts": re.compile(
        r"Problem instances count:\s+(\d+)\s*,\s*total verified \(safe/unsat\):\s+(\d+)\s*,\s*"
        r"total falsified \(unsafe/sat\):\s+(\d+)\s*,\s*timeout:\s+(\d+)"
    ),
    "mean_all": re.compile(r"mean time for ALL instances \(total (\d+)\):([\d.]+), max time: ([\d.]+)"),
}
CATEGORY_RE = re.compile(r"^(\S[\w\s+/-]*?)\s+\(total (\d+)\), index: (\[.*\])\s*$")


def ensure_config_copied(base_config: Path) -> str:
    """Copies base_config into exp_configs/beta_crown/ if not already
    there (idempotent), returns the relative --config path to pass."""
    ABCROWN_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    dest = ABCROWN_CONFIG_DIR / base_config.name
    if not dest.exists() or dest.read_bytes() != base_config.read_bytes():
        shutil.copy(base_config, dest)
    return f"exp_configs/beta_crown/{base_config.name}"


def run_one(config_rel: str, onnx_path: str, epsilon: float, start: int, end: int, log_path: Path) -> dict:
    cmd = [
        str(ABCROWN_PYTHON), "abcrown.py",
        "--config", config_rel,
        "--onnx_path", onnx_path,
        "--epsilon", str(epsilon),
        "--start", str(start),
        "--end", str(end),
    ]
    start_time = time.time()
    proc = subprocess.run(cmd, cwd=str(ABCROWN_DIR), capture_output=True, text=True)
    wall_time = time.time() - start_time
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"$ {' '.join(cmd)}\n(cwd={ABCROWN_DIR})\n\n=== STDOUT ===\n{proc.stdout}\n=== STDERR ===\n{proc.stderr}\n"
    )

    per_image = [{"status": m.group(1), "time_sec": float(m.group(2))} for m in RESULT_RE.finditer(proc.stdout)]

    summary = {}
    m = SUMMARY_LINE_RES["final_verified_acc"].search(proc.stdout)
    if m:
        summary["final_verified_acc_pct"] = float(m.group(1))
        summary["total_examples"] = int(m.group(2))
    m = SUMMARY_LINE_RES["counts"].search(proc.stdout)
    if m:
        summary["instances_count"] = int(m.group(1))
        summary["verified_safe"] = int(m.group(2))
        summary["falsified_unsafe"] = int(m.group(3))
        summary["timeout"] = int(m.group(4))
    m = SUMMARY_LINE_RES["mean_all"].search(proc.stdout)
    if m:
        summary["mean_time_all_sec"] = float(m.group(2))
        summary["max_time_all_sec"] = float(m.group(3))
    categories = {}
    for line in proc.stdout.splitlines():
        cm = CATEGORY_RE.match(line.strip())
        if cm:
            categories[cm.group(1).strip()] = {"total": int(cm.group(2)), "index": json.loads(cm.group(3))}
    if categories:
        summary["categories"] = categories

    return {
        "per_image": per_image,
        "summary": summary,
        "wall_time_sec": wall_time,
        "exit_code": proc.returncode,
        "log_path": str(log_path),
        "stdout_tail": proc.stdout[-2000:] if proc.returncode != 0 else None,
    }


def load_done_keys(results_path: Path) -> set:
    done = set()
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                done.add((row["model"], row["method"], row["sample_id"], row["epsilon"], row["start"], row["end"]))
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest_json")
    ap.add_argument("results_jsonl")
    ap.add_argument("--dry-run", action="store_true", help="Print the planned run list without executing")
    ap.add_argument("--log-dir", default=str(REPO_ROOT / "NNs" / "sweep_logs"))
    args = ap.parse_args()

    with open(args.manifest_json) as f:
        manifest = json.load(f)

    results_path = Path(args.results_jsonl)
    done = load_done_keys(results_path)
    log_dir = Path(args.log_dir)

    plan = []
    for entry in manifest:
        for eps in entry["epsilons"]:
            key = (entry["model"], entry["method"], entry["sample_id"], eps, entry["start"], entry["end"])
            if key in done:
                continue
            plan.append((entry, eps))

    print(f"{len(plan)} runs planned ({len(manifest)} manifest entries x their epsilons, "
          f"{len(done)} already done, skipped).")
    if args.dry_run:
        for entry, eps in plan:
            print(f"  {entry['model']:14s} {entry['method']:20s} {entry['sample_id']:10s} "
                  f"eps={eps} [{entry['start']}:{entry['end']}]  onnx={entry['onnx_path']}")
        return

    config_cache = {}
    with open(results_path, "a") as out:
        for i, (entry, eps) in enumerate(plan):
            base_config = Path(entry["base_config"])
            if base_config not in config_cache:
                config_cache[base_config] = ensure_config_copied(base_config)
            config_rel = config_cache[base_config]

            log_name = f"{entry['model']}_{entry['method']}_{entry['sample_id']}_eps{eps}_{entry['start']}-{entry['end']}.log"
            log_path = log_dir / log_name
            print(f"[{i+1}/{len(plan)}] {entry['model']} {entry['method']} {entry['sample_id']} "
                  f"eps={eps} [{entry['start']}:{entry['end']}] ...", flush=True)

            result = run_one(config_rel, entry["onnx_path"], eps, entry["start"], entry["end"], log_path)
            row = {
                "model": entry["model"],
                "method": entry["method"],
                "sample_id": entry["sample_id"],
                "epsilon": eps,
                "start": entry["start"],
                "end": entry["end"],
                "onnx_path": entry["onnx_path"],
                **result,
            }
            out.write(json.dumps(row) + "\n")
            out.flush()
            acc = result["summary"].get("final_verified_acc_pct")
            print(f"    -> exit={result['exit_code']} wall={result['wall_time_sec']:.1f}s "
                  f"verified_acc={acc}", flush=True)


if __name__ == "__main__":
    main()
