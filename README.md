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
  --recursive` after cloning.
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
none of that belongs in version control. Recreate with:

```
git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
cd alpha-beta-CROWN
git submodule update --init --recursive   # pulls in auto_LiRPA
UV_CACHE_DIR=/path/to/scratch/uv_cache uv sync
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
  history, so we pushed to a new branch rather than overwrite it).
- `tensat`: `xvade/tensat`, `master` (this already was the fork our local
  clone was tracking, so local commits pushed straight there).
