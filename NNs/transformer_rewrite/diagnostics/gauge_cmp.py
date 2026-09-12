"""Compare gauge matrices: D = G_a^{-1} G_b per layer/head/side. Diagonal D is CROWN-neutral, so the off-diagonal part is what matters."""
import torch, numpy as np, sys
G = "gauges/"
def load(p): g = torch.load(G + p); return g["qk"].double(), g["av"].double()
def stats(A):
    s = torch.linalg.svdvals(A); return s.min().item(), s.max().item(), (s.max() / s.min()).item()
def cmp(name, pa, pb, la="manual", lb="learned"):
    Ga, Gb = load(pa), load(pb); L, H, d, _ = Ga[0].shape; I = torch.eye(d, dtype=torch.float64)
    print(f"\n=== {name}: D = {la}^-1 {lb}   ({pa} vs {pb}); L={L} H={H} dh={d}")
    print(f"  layer side | relF(D-I) | offdiag/diag | to scalar | to diag | to orthog | sv(D) min/max cond | cond {la} | cond {lb}   (means over heads; [max over heads] for offdiag/diag)")
    tot = {}
    for side, ia, ib in (("Q", Ga[0], Gb[0]), ("A", Ga[1], Gb[1])):
        for l in range(L):
            rows = []
            for h in range(H):
                A, B = ia[l, h], ib[l, h]; D = torch.linalg.solve(A, B)
                relF = ((D - I).norm() / d ** 0.5).item()
                diag = torch.diag(torch.diag(D)); off = ((D - diag).norm() / diag.norm()).item()
                alpha = torch.trace(D) / d; sc = ((D - alpha * I).norm() / D.norm()).item()
                dg = ((D - diag).norm() / D.norm()).item()
                U, S, Vh = torch.linalg.svd(D); Q = U @ Vh; orth = ((D - Q).norm() / D.norm()).item()
                # off-diagonal energy after normalising D to unit diagonal (row scaling is a diagonal gauge, neutral)
                Dn = D @ torch.diag(1.0 / torch.diag(D)); offn = ((Dn - I).norm() / d ** 0.5).item()
                smin, smax, cd = stats(D); ca = stats(A)[2]; cb = stats(B)[2]
                rows.append((relF, off, sc, dg, orth, smin, smax, cd, ca, cb, offn))
            r = np.array(rows); m = r.mean(0); mx = r.max(0)
            print(f"  L{l} {side}   | {m[0]:8.3f} | {m[1]:6.3f} [{mx[1]:6.3f}] | {m[2]:6.3f} | {m[3]:6.3f} | {m[4]:6.3f} | {m[5]:6.3f}/{m[6]:6.3f} {m[7]:7.2f} | {m[8]:6.2f} | {m[9]:6.2f}   unit-diag offdiag relF {m[10]:.3f}")
            tot[(l, side)] = m[1]
    top = sorted(tot.items(), key=lambda kv: -kv[1])[:4]; print("  largest offdiag/diag (layer, side):", [(f"L{l}{s}", round(v, 3)) for (l, s), v in top])
    return tot
# per model
cmp("big_3 manual(unified) vs learned", "formula_sst_bert_big_3_l1N_qk_av0_r3.pt", "deept_big3_seed0.pt")
cmp("big_3 closed form vs manual(unified)", "formula_sst_bert_big_3_svd_jacN_all_r3.pt", "formula_sst_bert_big_3_l1N_qk_av0_r3.pt", "closed", "manual")
cmp("big_3 closed form vs learned", "formula_sst_bert_big_3_svd_jacN_all_r3.pt", "deept_big3_seed0.pt", "closed", "learned")
cmp("yelp3 manual(l1N) vs learned", "formula_yelp_bert_small_3_l1N_r3.pt", "deept_yelp3_seed0.pt")
cmp("yelp3 closed form vs manual(l1N)", "formula_yelp_bert_small_3_svd_jacN_all_r3.pt", "formula_yelp_bert_small_3_l1N_r3.pt", "closed", "manual")
cmp("yelp3 closed form vs learned", "formula_yelp_bert_small_3_svd_jacN_all_r3.pt", "deept_yelp3_seed0.pt", "closed", "learned")
cmp("small_6 manual(unified r4) vs learned", "formula_sst_bert_small_6_l1N_qk_av0_r4.pt", "deept_small6_seed0.pt")
cmp("small_6 closed form (24-box, paired) vs manual(unified r4)", "formula_sst_bert_small_6_svd_jacN_all.pt", "formula_sst_bert_small_6_l1N_qk_av0_r4.pt", "closed", "manual")
cmp("small_6 closed form (24-box, paired) vs learned", "formula_sst_bert_small_6_svd_jacN_all.pt", "deept_small6_seed0.pt", "closed", "learned")
# sanity: identity-like comparison of two closed-form builds (r3 vs r4 = same seed, should be ~equal)
cmp("small_6 closed form .pt vs _r4.pt (same seed build)", "formula_sst_bert_small_6_svd_jacN_all.pt", "formula_sst_bert_small_6_svd_jacN_all_r4.pt", "closed", "closed_r4")
# how far is each gauge from the identity (the stock net)?
print("\n=== distance from identity (relF(G-I) mean over heads, per layer, Q / A) and cond")
for name, p in (("big_3 learned", "deept_big3_seed0.pt"), ("big_3 manual", "formula_sst_bert_big_3_l1N_qk_av0_r3.pt"), ("yelp3 learned", "deept_yelp3_seed0.pt"), ("yelp3 manual", "formula_yelp_bert_small_3_l1N_r3.pt"), ("small_6 learned", "deept_small6_seed0.pt"), ("small_6 manual", "formula_sst_bert_small_6_l1N_qk_av0_r4.pt"), ("small_6 closed", "formula_sst_bert_small_6_svd_jacN_all.pt")):
    Gq, Ga = load(p); L, H, d, _ = Gq.shape; I = torch.eye(d, dtype=torch.float64)
    q = [((Gq[l] - I).norm(dim=(1, 2)) / d ** 0.5).mean().item() for l in range(L)]; a = [((Ga[l] - I).norm(dim=(1, 2)) / d ** 0.5).mean().item() for l in range(L)]
    cq = [np.mean([stats(Gq[l, h])[2] for h in range(H)]) for l in range(L)]; ca = [np.mean([stats(Ga[l, h])[2] for h in range(H)]) for l in range(L)]
    print(f"  {name:16s} Q {['%.3f' % x for x in q]}  A {['%.3f' % x for x in a]}  condQ {['%.1f' % x for x in cq]} condA {['%.1f' % x for x in ca]}")
