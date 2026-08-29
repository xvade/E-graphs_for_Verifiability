#!/usr/bin/env python3
# Baseline verification of scouted VNN-COMP models, UNMODIFIED (no tensat
# rewriting), on their bundled vnnlib specs. Manifest-driven, resumable.
# Each row: (model, onnx, vnnlib, base_config, timeout). Runs abcrown.py with
# --onnx_path/--vnnlib_path/--timeout overriding the per-model base config,
# captures full stdout to a log, parses the per-instance "Result: X in Y
# seconds" line, appends one JSON row to baselines_results.jsonl.
#
# Run (on a GPU node, from repo root):
#   srun ... alpha-beta-CROWN/.venv/bin/python NNs/run_vnncomp_baselines.py
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CM = REPO / "NNs" / "candidate_models"
ABC = REPO / "alpha-beta-CROWN" / "complete_verifier"
PY = REPO / "alpha-beta-CROWN" / ".venv" / "bin" / "python"
LOGDIR = REPO / "NNs" / "baseline_logs"
RESULTS = REPO / "NNs" / "baselines_results.jsonl"

# Resolved instances print "Result: <verdict> in <N> seconds"; timeout/unknown
# print a bare "Result: <verdict>" followed by a separate "Time: <N>" line. Match
# the verdict either way, and pull time from the "in N seconds" clause or the
# trailing Time: line.
RESULT_RE = re.compile(r"Result:\s+(\S+)")
RESULT_INSEC_RE = re.compile(r"Result:\s+\S+\s+in\s+([\d.]+)\s+seconds")
TIME_LINE_RE = re.compile(r"^Time:\s+([\d.]+)\s*$", re.M)

def manifest():
    rows = []
    # resnet_2b @ eps 0.008, resnet_4b @ eps 0.004 (official cifar10_resnet)
    for prop in sorted((CM / "cifar10_resnet2021" / "vnnlib_2b").glob("*.vnnlib")):
        rows.append(dict(model="resnet_2b", onnx=str(CM/"cifar10_resnet2021"/"onnx"/"resnet_2b.onnx"),
                         vnnlib=str(prop), cfg=str(CM/"cfg_resnet.yaml"), timeout=300))
    for prop in sorted((CM / "cifar10_resnet2021" / "vnnlib").glob("*.vnnlib")):
        rows.append(dict(model="resnet_4b", onnx=str(CM/"cifar10_resnet2021"/"onnx"/"resnet_4b.onnx"),
                         vnnlib=str(prop), cfg=str(CM/"cfg_resnet.yaml"), timeout=300))
    for prop in sorted((CM / "mnistfc2021").glob("prop_*.vnnlib")):
        rows.append(dict(model="mnist-net_256x2", onnx=str(CM/"mnistfc2021"/"mnist-net_256x2.onnx"),
                         vnnlib=str(prop), cfg=str(CM/"cfg_mnistfc.yaml"), timeout=300))
    for prop in sorted((CM / "eran2021_sigmoid" / "specs").glob("*.vnnlib")):
        rows.append(dict(model="ffnnSIGMOID_6x200", onnx=str(CM/"eran2021_sigmoid"/"ffnnSIGMOID_Point_6x200.onnx"),
                         vnnlib=str(prop), cfg=str(CM/"cfg_sigmoid.yaml"), timeout=300))
    return rows

def done_keys():
    keys = set()
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                keys.add((r["model"], r["vnnlib"]))
    return keys

def run_one(row):
    LOGDIR.mkdir(exist_ok=True)
    tag = f"{row['model']}__{Path(row['vnnlib']).stem}"
    log = LOGDIR / f"{tag}.log"
    cmd = [str(PY), "abcrown.py", "--config", row["cfg"],
           "--onnx_path", row["onnx"], "--vnnlib_path", row["vnnlib"],
           "--timeout", str(row["timeout"])]
    t0 = time.time()
    with open(log, "w") as f:
        p = subprocess.run(cmd, cwd=str(ABC), stdout=f, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    text = log.read_text()
    verdicts = RESULT_RE.findall(text)
    verdict = verdicts[-1] if verdicts else "NO_RESULT_LINE"
    tm = RESULT_INSEC_RE.search(text) or TIME_LINE_RE.search(text)
    vtime = float(tm.group(1)) if tm else None
    return dict(model=row["model"], vnnlib=row["vnnlib"], onnx=Path(row["onnx"]).name,
                verdict=verdict, verify_time=vtime, wall_time=round(wall, 1),
                exit_code=p.returncode, log=str(log.relative_to(REPO)))

def main():
    rows = manifest()
    done = done_keys()
    todo = [r for r in rows if (r["model"], r["vnnlib"]) not in done]
    print(f"manifest {len(rows)} instances, {len(done)} done, {len(todo)} to run", flush=True)
    with open(RESULTS, "a") as out:
        for i, row in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {row['model']} {Path(row['vnnlib']).name} ...", flush=True)
            res = run_one(row)
            out.write(json.dumps(res) + "\n"); out.flush()
            print(f"    -> {res['verdict']} ({res['verify_time']}s, wall {res['wall_time']}s, exit {res['exit_code']})", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
