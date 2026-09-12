"""Do two gauges share the metric G G^T (equivalently: differ by a rotation)? Scale-free relative Frobenius distance of the normalised metrics."""
import torch, numpy as np
G = "gauges/"
def load(p): g = torch.load(G + p); return g["qk"].double(), g["av"].double()
def met(A): M = A @ A.T; return M / M.norm()
def rel(A, B): return (met(A) - met(B)).norm().item()
def cmp(name, pa, pb):
    Ga, Gb = load(pa), load(pb); L, H, d, _ = Ga[0].shape; I = torch.eye(d, dtype=torch.float64)
    out = []
    for side, ia, ib in (("Q", Ga[0], Gb[0]), ("A", Ga[1], Gb[1])):
        out.append(side + " " + " ".join(f"L{l}:{np.mean([rel(ia[l,h], ib[l,h]) for h in range(H)]):.3f}" for l in range(L)))
    # reference scales: distance of each metric from the identity metric, and between the two random rotations of the same metric (=0)
    ref = []
    for side, ia, ib in (("Q", Ga[0], Gb[0]), ("A", Ga[1], Gb[1])):
        ref.append(side + " " + " ".join(f"L{l}:{np.mean([rel(ia[l,h], I) for h in range(H)]):.3f}/{np.mean([rel(ib[l,h], I) for h in range(H)]):.3f}" for l in range(L)))
    print(f"{name}\n   metric distance a vs b : {' | '.join(out)}\n   each vs identity (a/b) : {' | '.join(ref)}")
cmp("big_3 closed vs learned", "formula_sst_bert_big_3_svd_jacN_all_r3.pt", "deept_big3_seed0.pt")
cmp("big_3 manual(unified) vs learned", "formula_sst_bert_big_3_l1N_qk_av0_r3.pt", "deept_big3_seed0.pt")
cmp("big_3 closed vs manual(unified)", "formula_sst_bert_big_3_svd_jacN_all_r3.pt", "formula_sst_bert_big_3_l1N_qk_av0_r3.pt")
cmp("yelp3 closed vs learned", "formula_yelp_bert_small_3_svd_jacN_all_r3.pt", "deept_yelp3_seed0.pt")
cmp("yelp3 manual(l1N) vs learned", "formula_yelp_bert_small_3_l1N_r3.pt", "deept_yelp3_seed0.pt")
cmp("yelp3 closed vs manual(l1N)", "formula_yelp_bert_small_3_svd_jacN_all_r3.pt", "formula_yelp_bert_small_3_l1N_r3.pt")
cmp("small_6 closed vs learned", "formula_sst_bert_small_6_svd_jacN_all.pt", "deept_small6_seed0.pt")
cmp("small_6 manual(unified) vs learned", "formula_sst_bert_small_6_l1N_qk_av0_r4.pt", "deept_small6_seed0.pt")
cmp("small_6 closed vs manual(unified)", "formula_sst_bert_small_6_svd_jacN_all.pt", "formula_sst_bert_small_6_l1N_qk_av0_r4.pt")
# probe-seed dependence of the closed form on Yelp (seed 0 vs seed 1 builds): same metric, different rotation?
cmp("yelp3 closed seed0 (r5) vs seed1 (r5s1)", "formula_yelp_bert_small_3_svd_jacN_all_r5.pt", "formula_yelp_bert_small_3_svd_jacN_all_r5s1.pt")
cmp("yelp3 l1N seed0 (r5) vs seed1 (r5s1)", "formula_yelp_bert_small_3_l1N_r5.pt", "formula_yelp_bert_small_3_l1N_r5s1.pt")
# random-rotation reference: relF(D - I)/sqrt(d) and to-orthog for D = closed^-1 (closed @ Qrandom)
torch.manual_seed(0); d = 64; Qr = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))[0]; print(f"reference: relF(Qrandom - I)/sqrt(d) = {((Qr - torch.eye(d, dtype=torch.float64)).norm() / d ** 0.5).item():.3f} (d=64)")
