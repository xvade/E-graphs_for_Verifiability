import sys, collections; sys.argv=["x"]; sys.path.insert(0, ".")  # run from NNs/transformer_rewrite (see run_*.sh); import deept_gauge as g
m, tok, net = g.build("sst_bert_small_3", "cpu")
for split in ["test", "dev"]:
    data = g.load_sst(split); S = g.short_instances(net, m, tok, data, 12); c = collections.Counter(e.shape[1] for _, _, e, _ in S); npos = collections.Counter()
    for _, _, e, t in S: npos[e.shape[1]] += len(g.positions(t))
    print(f"# {split}: {len(data)} sentences; correctly classified with <=12 tokens: {len(S)} ({sum(npos.values())} positions); by length: " + ", ".join(f"{n}:{c[n]}s/{npos[n]}p" for n in sorted(c)) + f"; <=8 tokens: {sum(c[n] for n in c if n <= 8)} sentences / {sum(npos[n] for n in npos if n <= 8)} positions", flush=True)
