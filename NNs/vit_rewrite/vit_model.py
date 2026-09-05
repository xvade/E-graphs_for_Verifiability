#!/usr/bin/env python
"""
Faithful PyTorch reimplementation of the VNN-COMP'23 ViT benchmark models (pgd_2_3_16, ibp_3_3_8),
loading the stock ONNX weights unchanged, with SWITCHABLE EXACT REWRITES of the attention block.
(2026-09-04)

The stock model is a real, competition-standard, PGD/IBP-trained vision transformer. Every rewrite
variant is FUNCTION-IDENTICAL to it (float64 gate in gate_variants()); the rewrites change only the
GRAPH that a bound propagator (auto_LiRPA CROWN / alpha-CROWN / IBP) has to relax.

Softmax:
  softmax="native"      torch.softmax -> auto_LiRPA BoundSoftmax ('lse' or 'complex' mode)
  softmax="shift_const" R1. softmax(s)=exp(s-c)/sum(exp(s-c)) for a FIXED constant c (shift-invariance
                        => exact for ANY c; c may be a per-(layer,head,row) constant vector). Mirrors the
                        'complex' decomposition minus its perturbed ReduceMax/Sub nodes.
QK^T product (per head, r=dh bilinear products per score):
  qk="qk"               (X Wq + bq)(X Wk + bk)^T   stock factorization
  qk="xm_x" / "x_mx"    R2. reassociate onto X: (X M)X^T / X(M X^T), M_h=Wq_h Wk_h^T, + exact bias cross-terms
  qk_gauge="svd"        R4. GAUGE rewrite: (XU)(XV)^T = (XUG)(XVG^-T)^T for invertible G in GL(dh). Exact for
                        any G; changes the per-coordinate operand widths the McCormick relaxation sees
                        (diagonal G is provably neutral, only mixing matters). "svd" = closed-form balanced
                        factorization of M_h (Wq_h' = U sqrt(S), Wk_h' = V sqrt(S)); biases transform with G.
Attn.V product:
  av="av"               A (X Wv + bv)               stock
  av="ax_w"             R3. (A X) Wv + bv   (rows of A sum to 1)
  av_gauge="svd"        R5. gauge on (Wv_h, Wo_h): Wv_h G, bv_h G, G^-1 Wo_h; "svd" = balanced factorization of
                        N_h = Wv_h Wo_h.
Diagnostics (NOT exact; slack attribution only): diag flags linQK / linSM / linAV replace exactly ONE
nonlinearity by its first-order linearization at a chosen point x0 (set_diag_constants), keeping the graph
connected. The CROWN width drop measures how much relaxation slack that nonlinearity contributes.
"""
import re, torch, torch.nn as nn, numpy as np, onnx
from onnx import numpy_helper

def load_onnx(path):
    m = onnx.load(path); g = m.graph
    W = {x.name: torch.tensor(numpy_helper.to_array(x).copy()) for x in g.initializer}
    const = {n.output[0]: numpy_helper.to_array(onnx.helper.get_attribute_value(n.attribute[0])) for n in g.node if n.op_type == "Constant"}
    mm = {n.name: W[n.input[1]] for n in g.node if n.op_type == "MatMul" and n.input[1] in W}
    blk0 = [n for n in g.node if n.op_type == "Concat" and n.name.endswith("/fn/fn.1/Concat")][0]
    H = int(const[blk0.input[2]].item()); dh = int(const[blk0.input[3]].item())
    return W, mm, H, dh

def gauge_qk(Wq, Wk, bq, bk, H, dh, mode):
    """Exact re-factorization of each head's bilinear form: Wq_h<-Wq_h G, bq_h<-bq_h G, Wk_h<-Wk_h G^-T, bk_h<-bk_h G^-T."""
    Wq, Wk, bq, bk = Wq.clone(), Wk.clone(), bq.clone(), bk.clone(); info = []
    for h in range(H):
        sl = slice(h * dh, (h + 1) * dh); Wqh, Wkh = Wq[:, sl], Wk[:, sl]
        if mode == "svd":
            M = Wqh @ Wkh.T; U, S, Vh = torch.linalg.svd(M); U, S = U[:, :dh], S[:dh]
            G = torch.linalg.pinv(Wqh) @ (U * S.sqrt())            # Wq_h G = U sqrt(S)  (=> Wk_h G^-T = V sqrt(S))
            info.append(f"h{h}: sv_min/max={S.min().item():.2e}/{S.max().item():.2e} cond(G)={torch.linalg.cond(G).item():.1e}")
        elif isinstance(mode, torch.Tensor):
            G = mode[h].double()
        elif isinstance(mode, str) and mode.startswith("rand"):   # random ORTHOGONAL gauge (probe of bound sensitivity to G)
            g = torch.Generator().manual_seed(int(mode[4:]) * 100 + h); G = torch.linalg.qr(torch.randn(dh, dh, generator=g, dtype=torch.float64))[0]
        else: raise ValueError(mode)
        Gi = torch.linalg.inv(G)
        Wq[:, sl] = Wqh @ G; bq[sl] = bq[sl] @ G; Wk[:, sl] = Wkh @ Gi.T; bk[sl] = bk[sl] @ Gi.T
    return Wq, Wk, bq, bk, info

def gauge_av(Wv, bv, Wo, H, dh, mode):
    """Exact: Wv_h<-Wv_h G, bv_h<-bv_h G, Wo_h<-G^-1 Wo_h (Wo_h = rows of Wo for head h)."""
    Wv, bv, Wo = Wv.clone(), bv.clone(), Wo.clone(); info = []
    for h in range(H):
        sl = slice(h * dh, (h + 1) * dh); Wvh, Woh = Wv[:, sl], Wo[sl, :]
        if mode == "svd":
            N = Wvh @ Woh; U, S, Vh = torch.linalg.svd(N); U, S = U[:, :dh], S[:dh]
            G = torch.linalg.pinv(Wvh) @ (U * S.sqrt())
            info.append(f"h{h}: sv_min/max={S.min().item():.2e}/{S.max().item():.2e} cond(G)={torch.linalg.cond(G).item():.1e}")
        elif isinstance(mode, torch.Tensor):
            G = mode[h].double()
        else: raise ValueError(mode)
        Gi = torch.linalg.inv(G)
        Wv[:, sl] = Wvh @ G; bv[sl] = bv[sl] @ G; Wo[sl, :] = Gi @ Woh
    return Wv, bv, Wo, info

def stock_attn_weights(onnx_path):
    """Per-layer stock attention weights in float64: list of (Wq,Wk,Wv,Wo,bq,bk,bv,bo), plus (H,dh,L)."""
    W, mm, H, dh = load_onnx(onnx_path); out = []
    L = len([k for k in W if re.match(r"1\.\d+\.0\.fn\.1\.query\.bias", k)])
    for l in range(L):
        p = f"/1/1.{l}/1.{l}.0/fn/fn.1"; b = f"1.{l}.0.fn.1"
        out.append(tuple(mm[f"{p}/{r}/MatMul"].double() for r in ("query", "key", "value", "out")) + tuple(W[f"{b}.{r}.bias"].double() for r in ("query", "key", "value", "out")))
    return out, H, dh, L

def svd_gauges(onnx_path):
    """The closed-form SVD-balanced gauges as tensors (L,H,dh,dh): (G_qk, G_av). Used to initialize learned gauges."""
    ws, H, dh, L = stock_attn_weights(onnx_path)
    Gq = torch.zeros(L, H, dh, dh, dtype=torch.float64); Ga = torch.zeros_like(Gq)
    for l, (Wq, Wk, Wv, Wo, *_r) in enumerate(ws):
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh)
            U, S, _ = torch.linalg.svd(Wq[:, sl] @ Wk[:, sl].T); Gq[l, h] = torch.linalg.pinv(Wq[:, sl]) @ (U[:, :dh] * S[:dh].sqrt())
            U, S, _ = torch.linalg.svd(Wv[:, sl] @ Wo[sl, :]); Ga[l, h] = torch.linalg.pinv(Wv[:, sl]) @ (U[:, :dh] * S[:dh].sqrt())
    return Gq, Ga

class BNTok(nn.Module):
    """BatchNorm over the feature dim of (B,T,D) exactly as the ONNX does it: transpose -> BN1d -> transpose."""
    def __init__(self, W, p, eps=1e-5):
        super().__init__()
        D = W[p + ".weight"].numel()
        self.bn = nn.BatchNorm1d(D, eps=eps); self.bn.weight.data = W[p + ".weight"].clone(); self.bn.bias.data = W[p + ".bias"].clone()
        self.bn.running_mean.data = W[p + ".running_mean"].clone(); self.bn.running_var.data = W[p + ".running_var"].clone(); self.bn.eval()
    def forward(self, x): return self.bn(x.transpose(1, 2)).transpose(1, 2)

class Attn(nn.Module):
    def __init__(self, W, mm, layer, H, dh, softmax="native", qk="qk", av="av", shift_c=0.0, qk_gauge=None, av_gauge=None):
        super().__init__()
        p = f"/1/1.{layer}/1.{layer}.0/fn/fn.1"; b = f"1.{layer}.0.fn.1"
        Wq, Wk, Wv, Wo = (mm[f"{p}/{r}/MatMul"].double() for r in ("query", "key", "value", "out"))
        bq, bk, bv, bo = (W[f"{b}.{r}.bias"].double() for r in ("query", "key", "value", "out"))
        D = Wq.shape[0]; self.H, self.dh, self.D = H, dh, D
        self.softmax, self.qk, self.av = softmax, qk, av; self.diag = set()
        self.gauge_info = []
        if qk_gauge is not None:
            Wq, Wk, bq, bk, info = gauge_qk(Wq, Wk, bq, bk, H, dh, qk_gauge); self.gauge_info += [f"qk {i}" for i in info]
        if av_gauge is not None:
            Wv, bv, Wo, info = gauge_av(Wv, bv, Wo, H, dh, av_gauge); self.gauge_info += [f"av {i}" for i in info]
        f = lambda t: nn.Parameter(t.float(), requires_grad=False)
        self.Wq, self.Wk, self.Wv, self.Wo = f(Wq), f(Wk), f(Wv), f(Wo)
        self.bq, self.bk, self.bv, self.bo = f(bq), f(bk), f(bv), f(bo)
        self.scale = 1.0 / np.sqrt(dh)
        self.register_buffer("shift", torch.full((1, 1, 1, 1), float(shift_c)))   # R1 constant; may become (1,H,T,1)
        # ---- R2: per-head M_h = Wq_h Wk_h^T (D,D) and exact bias cross-terms (float64 -> float32)
        Wqh = Wq.reshape(D, H, dh).permute(1, 0, 2); Wkh = Wk.reshape(D, H, dh).permute(1, 0, 2)
        bqh = bq.reshape(H, dh); bkh = bk.reshape(H, dh)
        M = torch.einsum("hid,hjd->hij", Wqh, Wkh)
        self.Mcat = f(M.permute(1, 0, 2).reshape(D, H * D)); self.MTcat = f(M.permute(2, 0, 1).reshape(D, H * D))
        self.u = f(torch.einsum("hid,hd->hi", Wqh, bkh)); self.v = f(torch.einsum("hid,hd->hi", Wkh, bqh)); self.cc = f(torch.einsum("hd,hd->h", bqh, bkh))
        # ---- R3: per-head Wv (H,D,dh), bv (H,1,dh)
        self.Wvh = f(Wv.reshape(D, H, dh).permute(1, 0, 2)); self.bvh = f(bv.reshape(H, 1, dh))
        for nm in ("Q0", "K0", "S0", "A0", "V0"): self.register_buffer(nm, torch.zeros(1))

    def heads(self, Y):
        B, T, _ = Y.shape; return Y.reshape(B, T, self.H, self.dh).transpose(1, 2)

    def forward(self, X):
        B, T, D = X.shape; H, dh = self.H, self.dh
        if self.qk == "qk":
            Q = self.heads(X @ self.Wq + self.bq); K = self.heads(X @ self.Wk + self.bk)
            if "linQK" in self.diag:  # diagnostic: bilinear -> tangent plane at (Q0,K0)
                S = self.Q0 @ K.transpose(-1, -2) + Q @ self.K0.transpose(-1, -2) - self.Q0 @ self.K0.transpose(-1, -2)
            else:
                S = Q @ K.transpose(-1, -2)
        else:
            Xt = X.transpose(1, 2).unsqueeze(1)
            if self.qk == "xm_x":
                XM = (X @ self.Mcat).reshape(B, T, H, D).transpose(1, 2); Sb = XM @ Xt
            else:
                MXt = (X @ self.MTcat).reshape(B, T, H, D).permute(0, 2, 3, 1); Sb = X.unsqueeze(1) @ MXt
            row = (X @ self.u.t()).transpose(1, 2).unsqueeze(-1); col = (X @ self.v.t()).transpose(1, 2).unsqueeze(-2)
            S = Sb + row + col + self.cc.reshape(1, H, 1, 1)
        S = S * self.scale
        if "linSM" in self.diag:      # diagnostic: softmax -> Jacobian linearization at S0:  A0 + J0 (S-S0)
            dS = S - self.S0; t = dS * self.A0
            A = self.A0 + t - self.A0 * t.sum(-1, keepdim=True)
        elif self.softmax == "native":
            A = torch.softmax(S, dim=-1)
        else:
            E = torch.exp(S - self.shift); A = E / E.sum(dim=-1, keepdim=True)
        if self.av == "av":
            V = self.heads(X @ self.Wv + self.bv)
            O = (self.A0 @ V + A @ self.V0 - self.A0 @ self.V0) if "linAV" in self.diag else A @ V
        else:
            AX = A @ X.unsqueeze(1); O = AX @ self.Wvh + self.bvh
        O = O.transpose(1, 2).reshape(B, T, D)
        return O @ self.Wo + self.bo

    @torch.no_grad()
    def capture(self, X):
        """Record Q0,K0,S0,A0,V0 at X (for diagnostics) and return the block output."""
        Q = self.heads(X @ self.Wq + self.bq); K = self.heads(X @ self.Wk + self.bk); S = (Q @ K.transpose(-1, -2)) * self.scale
        A = torch.softmax(S, -1); V = self.heads(X @ self.Wv + self.bv)
        self.Q0, self.K0, self.S0, self.A0, self.V0 = Q, K, S, A, V
        return S

class Block(nn.Module):
    def __init__(self, W, mm, layer, H, dh, **kw):
        super().__init__()
        self.n1 = BNTok(W, f"1.{layer}.0.fn.0.norm"); self.attn = Attn(W, mm, layer, H, dh, **kw)
        self.n2 = BNTok(W, f"1.{layer}.1.fn.0.norm")
        p = f"/1/1.{layer}/1.{layer}.1/fn/fn.1"
        self.W1 = nn.Parameter(mm[f"{p}/fn.1.0/MatMul"].clone(), requires_grad=False); self.b1 = nn.Parameter(W[f"1.{layer}.1.fn.1.0.bias"].clone(), requires_grad=False)
        self.W2 = nn.Parameter(mm[f"{p}/fn.1.3/MatMul"].clone(), requires_grad=False); self.b2 = nn.Parameter(W[f"1.{layer}.1.fn.1.3.bias"].clone(), requires_grad=False)
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        h = torch.relu(self.n2(x) @ self.W1 + self.b1)
        return x + (h @ self.W2 + self.b2)

class ViT(nn.Module):
    def __init__(self, onnx_path, softmax="native", qk="qk", av="av", shift_c=0.0, qk_gauge=None, av_gauge=None):
        super().__init__()
        W, mm, H, dh = load_onnx(onnx_path)
        cw = W["0.projection.weight"]; D, _, P, _ = cw.shape
        self.proj = nn.Conv2d(3, D, P, stride=P); self.proj.weight.data = cw.clone(); self.proj.bias.data = W["0.projection.bias"].clone()
        self.cls = nn.Parameter(W["0.cls_token"].reshape(1, 1, D).clone(), requires_grad=False)
        self.pos = nn.Parameter(W["0.positions"].unsqueeze(0).clone(), requires_grad=False)
        nl = len([k for k in W if re.match(r"1\.\d+\.0\.fn\.1\.query\.bias", k)])
        _g = lambda g, l: g[l] if (torch.is_tensor(g) and g.ndim == 4) else g     # learned gauges: tensor (L,H,dh,dh)
        self.blocks = nn.ModuleList([Block(W, mm, l, H, dh, softmax=softmax, qk=qk, av=av, shift_c=shift_c, qk_gauge=_g(qk_gauge, l), av_gauge=_g(av_gauge, l)) for l in range(nl)])
        qk_gauge = f"tensor{tuple(qk_gauge.shape)}" if torch.is_tensor(qk_gauge) else qk_gauge
        av_gauge = f"tensor{tuple(av_gauge.shape)}" if torch.is_tensor(av_gauge) else av_gauge
        self.head_bn = nn.BatchNorm1d(D, eps=1e-5); self.head_bn.weight.data = W["2.1.weight"].clone(); self.head_bn.bias.data = W["2.1.bias"].clone()
        self.head_bn.running_mean.data = W["2.1.running_mean"].clone(); self.head_bn.running_var.data = W["2.1.running_var"].clone(); self.head_bn.eval()
        self.fc = nn.Linear(D, 10); self.fc.weight.data = W["2.2.weight"].clone(); self.fc.bias.data = W["2.2.bias"].clone()
        self.meta = dict(D=D, P=P, H=H, dh=dh, layers=nl, softmax=softmax, qk=qk, av=av, shift_c=shift_c, qk_gauge=str(qk_gauge), av_gauge=str(av_gauge))
        self.gauge_info = [f"L{l} {s}" for l, blk in enumerate(self.blocks) for s in blk.attn.gauge_info]
        self.eval()
    def embed(self, x):
        B = x.shape[0]; t = self.proj(x).flatten(2).transpose(1, 2)
        return torch.cat([self.cls.expand(B, -1, -1), t], dim=1) + self.pos
    def forward(self, x):
        t = self.embed(x)
        for blk in self.blocks: t = blk(t)
        return self.fc(self.head_bn(t.mean(dim=1)))
    # ---- diagnostics / R1 shift helpers
    def set_diag(self, flags):
        for blk in self.blocks: blk.attn.diag = set(flags)
    @torch.no_grad()
    def set_diag_constants(self, x0):
        t = self.embed(x0)
        for blk in self.blocks:
            blk.attn.capture(blk.n1(t)); t = blk(t)
    @torch.no_grad()
    def set_shift_from_centers(self, xs):
        """R1 per-(layer,head,row) FIXED shift = mean pre-softmax score over the given inputs (any constant is exact)."""
        t = self.embed(xs)
        for blk in self.blocks:
            S = blk.attn.capture(blk.n1(t)); blk.attn.shift = S.mean(dim=(0, 3), keepdim=True)[0:1]  # (1,H,T,1)
            t = blk(t)

VARIANTS = {
    "base":         dict(),
    "R1_c0":        dict(softmax="shift_const", shift_c=0.0),
    "R1_c10":       dict(softmax="shift_const", shift_c=10.0),
    "R1_rowmean":   dict(softmax="shift_const", shift_c=0.0),     # harness fills the per-row shift from instance centers
    "R2a_xm_x":     dict(qk="xm_x"),
    "R2b_x_mx":     dict(qk="x_mx"),
    "R3_ax_w":      dict(av="ax_w"),
    "R4_qk_svd":    dict(qk_gauge="svd"),
    "R5_av_svd":    dict(av_gauge="svd"),
    "R45_both_svd": dict(qk_gauge="svd", av_gauge="svd"),
}
for _s in range(5):   # random orthogonal gauges: probe how sensitive the bound is to G (room for optimizing G)
    VARIANTS[f"R4_rand{_s}"] = dict(qk_gauge=f"rand{_s}"); VARIANTS[f"R5_rand{_s}"] = dict(av_gauge=f"rand{_s}")

def gate_variants(onnx_path, n_pts=200, eps=0.02, seed=0, verbose=True, variants=None):
    """Two-stage exactness gate. (a) faithful: base fp32 vs onnxruntime fp32. (b) exact: every variant in
    float64 vs base in float64, on random points in eps-boxes around random normalized inputs."""
    import onnxruntime as ort
    torch.manual_seed(seed)
    base = ViT(onnx_path)
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"]); inp = sess.get_inputs()[0].name
    x = torch.randn(n_pts, 3, 32, 32) * 0.5 + (torch.rand(n_pts, 3, 32, 32) - 0.5) * 2 * eps
    with torch.no_grad():
        ref = np.concatenate([sess.run(None, {inp: x[i:i + 1].numpy()})[0] for i in range(n_pts)])
        res = {"faithful_fp32_vs_ort": float(np.abs(base(x).numpy() - ref).max())}
        b64 = ViT(onnx_path).double(); y64 = b64(x.double())
        for name, kw in (variants or VARIANTS).items():
            if name == "base": continue
            v = ViT(onnx_path, **kw).double()
            if name == "R1_rowmean": v.set_shift_from_centers(x[:50].double())
            res[name] = float((v(x.double()) - y64).abs().max().item())
            if v.gauge_info and verbose: print("#     " + " | ".join(v.gauge_info))
    if verbose:
        print(f"# gate {onnx_path.split('/')[-1]} meta={base.meta}")
        for k, v in res.items(): print(f"#   {k:24s} max|diff| = {v:.3e}")
    return res

if __name__ == "__main__":
    import sys
    gate_variants(sys.argv[1] if len(sys.argv) > 1 else "/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability/vnncomp2023_benchmarks/benchmarks/vit/onnx/pgd_2_3_16.onnx")
