import sys, time, torch, numpy as np, collections; sys.argv = ["x"]; sys.path.insert(0, ".")  # run from NNs/transformer_rewrite (see run_*.sh); import deept_gauge as g
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
dev = "cuda"; m, tok, net = g.build("sst_bert_small_3", dev); data = g.load_sst("test")
bylen = {}
for ex in data:
    e, toks = g.embed(m, tok, ex); bylen.setdefault(e.shape[1], (ex, e, toks))
n = 12; ex, e, toks = bylen[n]; i = g.positions(toks)[0]
lp = BoundedModule(net, torch.empty(1, n, 128, device=dev), bound_opts={"softmax": "lse"}, device=dev)
torch.cuda.memory._record_memory_history(max_entries=200000)
lb = g.crown_lb(lp, e, i, 0.03, ex["label"], dev, method="CROWN")
snap = torch.cuda.memory._snapshot(); torch.cuda.memory._record_memory_history(enabled=None)
# largest live/ever allocations by frames
agg = collections.Counter(); ex_frames = {}
for seg in snap["segments"]:
    for b in seg["blocks"]:
        if b["state"] != "active_allocated" and "frames" not in b: continue
        fr = b.get("frames", []); key = tuple(f"{f['filename'].split('/')[-1]}:{f['line']} {f['name']}" for f in fr if "auto_LiRPA" in f["filename"])[:5]
        agg[key] += b["size"]; ex_frames[key] = fr
print(f"peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB; lb {lb:+.4f}")
blocks = []
for seg in snap["segments"]:
    for b in seg["blocks"]:
        if "frames" in b: blocks.append((b["size"], b["state"], tuple(f"{f['filename'].split('/')[-1]}:{f['line']} {f['name']}" for f in b["frames"] if "auto_LiRPA" in f["filename"])[:6]))
blocks.sort(reverse=True)
print("LARGEST SINGLE BLOCKS:")
for sz, st, key in blocks[:6]: print(f"{sz/2**30:6.2f} GiB {st} <-", " | ".join(key))
# alloc history traces: largest single allocations ever
ev = [(t["size"], tuple(f"{f['filename'].split('/')[-1]}:{f['line']} {f['name']}" for f in t.get("frames", []) if "auto_LiRPA" in f["filename"])[:6]) for dt in snap.get("device_traces", []) for t in dt if t["action"] == "alloc"]
ev.sort(reverse=True); print("LARGEST ALLOC EVENTS:")
for sz, key in ev[:6]: print(f"{sz/2**30:6.2f} GiB <-", " | ".join(key))
for key, sz in agg.most_common(8): print(f"{sz/2**30:6.2f} GiB  <-", " | ".join(key))
# also: per-node output sizes (largest intermediate nodes)
sizes = sorted(((int(np.prod(nd.output_shape)) if getattr(nd, 'output_shape', None) is not None else 0, nm, type(nd).__name__) for nm, nd in lp._modules.items()), reverse=True)[:8]
print("largest node outputs:", sizes)
