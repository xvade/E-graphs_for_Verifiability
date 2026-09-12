# E-graphs for Verifiability

Measuring how TENSAT (equality-saturation-based tensor graph optimization,
built on TASO) affects neural network *verifiability* -- specifically, how
much tighter/looser the bounds alpha-beta-CROWN can compute get, before vs.
after TENSAT optimizes a network, under a fixed compute budget.

Start with `TENSAT_SUMMARY.md` and `TASO_SUMMARY.md` for the technical
deep-dive (paper summary, whole-codebase architecture, build internals) and
`PROGRESS.md` for a chronological log of what's been done. `BUGS.md` catalogs
bugs found in vanilla TASO/tensat along the way. The `taso/` and `tensat/`
submodules are **forks we own** — their entire codebases are in scope for our
documentation, specification, and testing, not just the changed files.

## Documentation map

- **`TILLICUM_SETUP.md`** — start here on the H200 cluster: what the project is, reading order,
  restore steps, scheduler discovery, the golden smoke, the final experiment list.
- **`NNs/transformer_rewrite/PROVENANCE.md`** — every headline number with the card (L40S / A100 / CPU)
  it was produced on and its sensitivity class; read before quoting or rerunning a result.
- **`NNs/README.md`** — the pipeline code, indexed by stage (model builders →
  converters → rule-gen → reconstruct → bounds). The place to start for "what
  does this script do".
- **`TASO_SUMMARY.md`** — the whole TASO fork mapped at architecture altitude
  (core / backends / generator / python bindings) with a doc-spec-test index.
- **`taso/MODIFICATIONS.md`** + **`taso/src/generator/README.md`** — this fork's
  delta from upstream TASO, and the generator's flag semantics
  (`RELAX_*`/`GEN_COMMUTE`, the presence-check gotcha, the 2 GB trap).
- **`tensat/MODIFICATIONS.md`** — this fork's added CLI modes (`verify`,
  `redundancy`, `parse_check`), verifiability-aware extraction, and weight
  provenance.
- **`docs/ADD_AN_OP.md`** — the end-to-end contract for teaching a new tensor op
  to every pipeline stage (generator → pb2egg → tensat parse/make/apply → Z3 →
  reconstruct), with the authoritative check per stage. Read this before adding
  or extending an operator.
- **`PROBLEMATIC.md`** — code/infra that resists testing or is suspected wrong;
  read before trusting or pinning it.
- **`NNs/tests/`** — the runnable regression suite (22 assertions);
  `taso/src/generator/tests/` holds the generator flag probe test.
- **`AGENTS.md`** — the documentation/spec/test conventions for this repo.

This repo lives on Hyak Klone's `gscratch` scratch space, which sysadmins
can wipe without notice -- that's why it's on GitHub at all, and why the
large, regenerable pieces below are deliberately *not* committed.

## Layout

- `egg/`, `taso/`, `tensat/` -- git submodules (own history, own commits).
  `taso` and `tensat` carry local fixes on top of their upstream forks (see
  `BUGS.md`); `egg` is unmodified. Run `git submodule update --init
  --recursive` after cloning. **Not needed for the transformer-gauge work**
  (`NNs/transformer_rewrite`, `NNs/vit_rewrite`, the final H200 experiments):
  that line uses only `alpha-beta-CROWN/` + the DeepT / VNN-COMP assets; the
  three submodules serve the e-graph rewrite pipeline (`NNs/*.sh`, `NNs/reassoc_results`).
- `NNs/` -- the concrete pipeline test case: `mnist_tiny_mlp` (PyTorch ->
  ONNX -> TASO -> tensat-optimized -> reconstructed ONNX with real weights),
  plus the scripts and verification log proving the round trip is numerically
  correct.
- `tensat.def` -- Apptainer recipe for the GPU build environment (CUDA
  12.4, cuDNN 9). Tracked; the built `.sif` is not (see below).

## Recreating excluded artifacts

### `tensat.sif` (~5.5G Apptainer container image)

Rebuild from the tracked recipe on a machine with Apptainer and enough
scratch space:

```
APPTAINER_CACHEDIR=/path/to/scratch/cache APPTAINER_TMPDIR=/path/to/scratch/tmp \
  apptainer build --fakeroot tensat.sif tensat.def
```

(`APPTAINER_CACHEDIR`/`TMPDIR` matter on clusters with small home-directory
quotas -- point them at scratch space, not `~`.)

### `alpha-beta-CROWN/` (~8.6G: repo + venv + datasets)

Not a submodule because our copy was a plain download with `.git` already
stripped and includes an 8G `uv sync`'d `.venv` plus downloaded datasets --
none of that belongs in version control. **Our copy differs from upstream in exactly one file**
(`complete_verifier/auto_LiRPA/operators/softmax.py`, verified by a full-tree diff on 2026-09-12); the diff is
tracked as `NNs/verifier_patches/auto_LiRPA_softmax_gradsafe.patch` and every gauge learner needs it. Recreate with:

```
NNs/verifier_patches/apply.sh          # clones upstream at the pinned commit (abcrown 0.7.0 / auto_LiRPA 0.7.2)
                                       # and applies our one local edit (gradient-safe softmax denominators)
cd alpha-beta-CROWN
UV_CACHE_DIR=/path/to/scratch/uv_cache uv sync   # torch 2.11+cu130 per uv.lock: the target driver must support CUDA 13
```

(Same quota issue as above -- `uv`'s cache defaults to `~/.cache/uv`.)

### Build toolchain (conda, rustup, uv cache)

Built outside this repo entirely, at `../toolchain-tensat/` (kept outside
because Miniconda's installer rejects paths containing spaces, and this
project directory's name has one). Not something to recreate from files --
just: Miniconda + `conda config --system --remove channels defaults` (avoids
Anaconda's interactive ToS prompt) + `conda install -c conda-forge
libprotobuf=3.21.12`, plus `rustup` via the normal installer. See
`TENSAT_SUMMARY.md` and `PROGRESS.md` for the full list of build-time fixes
this required.

## Submodule remotes

- `egg`: `yycdavid/egg` (upstream fork we build against; no local changes,
  so no personal fork needed -- if that repo's history ever changes
  upstream, re-point this at a personal fork of it).
- `taso`: `xvade/TASO`, branch `klone-cpu-gpu-build` (our local commits
  pushed here -- `xvade/TASO`'s `master` is an unrelated, pre-existing fork
  history, so we pushed to a new branch rather than overwrite it). The pinned
  SHA (3699a0c) is also branch `pre-h200` on that fork (2026-09-12 snapshot);
  the uncommitted `src/generator/rules.pb.{cc,h}` in a built checkout are
  protoc output, regenerated by the build, not fork changes.
- `tensat`: `xvade/tensat`, `master` (this already was the fork our local
  clone was tracking, so local commits pushed straight there).
