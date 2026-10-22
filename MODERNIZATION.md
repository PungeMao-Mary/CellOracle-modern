# CellOracle — macOS / modern-stack modernization

This branch turns CellOracle 0.22.0 into a package that installs and runs on a
current macOS + Python stack, and adds optional hardware acceleration
(Apple Accelerate BLAS and Apple-silicon **Metal GPU via MLX**).

Everything below was verified on an Apple-silicon Mac with

| component | tested versions |
|---|---|
| Python | 3.10.20 and 3.12.13 |
| NumPy | 2.2.6 and 2.5.3 |
| pandas | 2.3.3 and 3.0.6 |
| SciPy | 1.15.3 and 1.18.1 |
| scikit-learn | 1.7.2 and 1.9.1 |
| scanpy / anndata | 1.11.5 / 0.11.4 and 1.12.4 / 0.13.4 |
| python-igraph | 1.0.0 |
| matplotlib | 3.10.9 and 3.11.2 |
| MLX | 0.32.2 |

---

## 1. What changed

### 1.1 Removed the compiled `velocyto` dependency

CellOracle used a handful of correlation kernels and a `Diffusion` helper from
`velocyto`, which no longer builds against NumPy 2.x. The needed routines are
re-implemented in

```
celloracle/trajectory/_velocity_compat.py
```

with identical mathematics (BSD-2 licensed reference: `velocyto.py`,
Copyright © 2017 P. Kharchenko, S. Linnarsson, G. La Manno):

* `colDeltaCor`, `colDeltaCorpartial`
* `colDeltaCorSqrt`, `colDeltaCorSqrtpartial`
* `colDeltaCorLog10`, `colDeltaCorLog10partial`
* `Diffusion`

They are drop-in replacements (same signatures / return shapes) and are
vectorised so they run on Accelerate BLAS or the GPU. `net_util.py` now uses
CellOracle's own `hdf5_processing` instead of `velocyto.serialization`.
Set `CELLORACLE_USE_VELOCYTO=1` to use an installed `velocyto` instead.

### 1.2 Modern NumPy / pandas / igraph compatibility

* Reliable handling of dense and sparse-anndata layers in
  `trajectory/oracle_utility.py` (`todense().A` → `np.asarray(...)`), which
  matters for pandas/NumPy 2 and `scipy.sparse` *arrays*.
* `np.asarray(sparse.sum(1))` instead of the removed `np.matrix.A` path in the
  embedding-shift calculation.
* `python-igraph` 1.0 dropped the `directed=` argument semantics of
  `eigenvector_centrality`; `links.get_network_score()` now builds an
  undirected graph explicitly (forward- and backward-compatible).
* `celloracle.trajectory` is now a real package (`__init__.py`) with lazy
  exports instead of a broken `init__.py`.
* `celloracle.data.load_tutorial_*` called `load_hdf5.load_hdf5(...)` on a
  function; fixed.
* `get_markov_simulation_cell_transition_table()` returned misspelled
  attributes that were never set; fixed (both spellings are now exposed).

### 1.3 Optional dependencies are really optional

`import celloracle` no longer requires the heavy/unmaintained parts of the old
stack:

| sub-system | dependency | behaviour without it |
|---|---|---|
| `celloracle.motif_analysis` | `genomepy`, `gimmemotifs`, `pybedtools` | importable; using it raises a descriptive `ImportError` |
| notebook helpers | `IPython`, `ipywidgets` | core import works; helpers fall back / raise a clear error |
| `TFinfo` HDF5 loading | `motif_analysis` | imported lazily only when a `.tfinfo` file is loaded |

`requirements.txt` was rewritten for the modern stack
(`numpy>=1.24`, `pandas>=2.0`, …) and no longer pins `numpy==1.26.4`,
`cython`, `velocyto`, `louvain`, `jupyter`, `genomepy` or `gimmemotifs`.
The optional groups are exposed as pip extras:

```bash
pip install "celloracle[motif]"      # ATAC-seq / motif analysis
pip install "celloracle[notebook]"   # IPython / ipywidgets helpers
pip install "celloracle[gpu]"        # MLX (Apple silicon only)
pip install "celloracle[all]"        # everything above
```

### 1.4 Fixed: `UnboundLocalError` in the vector-field plotting wrapper

`Oracle.plot_simulation_flow_on_grid()` and
`plot_simulation_flow_random_on_grid()` could raise

```
UnboundLocalError: local variable 'gridpoints_coordinates' referenced before assignment
```

The plotting wrapper (`celloracle/visualizations/development_module_visualization.py`,
`_plot_simulation_flow_on_grid`) resolved the grid coordinates with

```python
if hasattr(self, "gridpoints_coordinates"):      # Gradient / dev module
    gridpoints_coordinates = self.gridpoints_coordinates
elif hasattr(self, "mass_filter"):               # <-- wrong attribute
    gridpoints_coordinates = self.flow_grid
```

A plain `Oracle` stores the grid as `flow_grid`, not `gridpoints_coordinates`,
so the second test only passed *because* the tutorials call
`calculate_mass_filter()` earlier — which incidentally made
`hasattr(self, "mass_filter")` true.  If `calculate_p_mass()` was called and the
plot was drawn **without** calling `calculate_mass_filter()`, neither branch ran,
`gridpoints_coordinates` stayed unbound, and the error surfaced at the
`ax.quiver(...)` call.

A second, related bug: the `mass_filter` local was also unguarded
(`if mass_filter_simulation … elif mass_filter …`, no `else`), so fixing only the
`elif` would have produced `UnboundLocalError: local variable 'mass_filter'`.

The fix introduces two small helpers in the same module:

* `_resolve_gridpoints_coordinates(self)` — returns `gridpoints_coordinates` or
  `flow_grid`, or raises a descriptive `AttributeError` pointing at
  `calculate_p_mass()`.
* `_resolve_filter_mask(self, n_points, *names)` — returns the first available
  mass-filter mask, or an all-`False` mask ("keep every grid point") when no
  filter has been computed.

`_plot_simulation_flow_on_grid` now also raises a clear error if `flow` (or
`flow_rndm` for the randomized plot) is missing.  The same unguarded
mask/coordinate pattern was corrected in `plot_pseudotime_on_grid`,
`plot_reference_flow_on_grid` and `plot_background_on_grid`.

Behaviour is unchanged for `Oracle_development_module` and
`Gradient_calculator` (they already own `gridpoints_coordinates` and a
simulation/whole mass filter).  This is a visualization-layer fix and does not
touch the acceleration / numerical code.

Regression tests: `celloracle/tests/test_vector_field_plot.py`.

---

## 2. Acceleration

### 2.1 Apple matrix accelerator (Accelerate / vecLib)

The official NumPy wheels for macOS already link BLAS/LAPACK against Apple's
**Accelerate** framework, so every dense `matmul`/`dot` in CellOracle runs on
the matrix accelerator. CellOracle reports what NumPy is linked against:

```python
import celloracle as co
co.accelerator_info()["blas"]
# {'blas': 'accelerate', 'lapack': 'accelerate', ...}
```

### 2.2 Apple GPU acceleration (Metal via MLX)

When [`mlx`](https://github.com/ml-explore/mlx) is installed on an Apple-silicon
Mac, CellOracle automatically uses the GPU for the two dominant kernels:

* **GRN signal propagation** (`Oracle.simulate_shift` → `_do_simulation`):
  the whole `n_propagation` loop (dense gene × gene products, masking and
  clipping) runs on-device with a single host↔device transfer.
* **Transition-probability correlations** (`Oracle.estimate_transition_prob`
  → `colDeltaCor` / `colDeltaCorpartial`): chunked gather + Gram computation on
  the GPU.

Control the backend with the module API or an environment variable:

```python
import celloracle as co

co.accelerator_info()          # backend, MLX version, Metal device, BLAS
co.set_accelerator("cpu")      # force CPU (Accelerate)
co.set_accelerator("gpu")      # force GPU (raises if unavailable)
co.set_accelerator("auto")     # default
```

```bash
CELLORACLE_ACCELERATOR=cpu  python analysis.py   # auto | cpu | gpu
CELLORACLE_DISABLE_MLX=1    python analysis.py   # ignore MLX entirely
```

Small problems stay on the CPU automatically (transfer overhead would dominate).

### 2.3 Precision

Metal has no `float64`, so the GPU path computes in `float32`. Measured
deviation from the float64 CPU path on synthetic data:

| operation | max relative deviation |
|---|---|
| `colDeltaCorpartial` / `colDeltaCor` | ~2e-7 |
| `_do_simulation` (3 propagations) | ~3e-5 |
| large dense `matmul` | ~8e-4 |

### 2.4 Benchmarks (Apple silicon, MLX 0.32.2)

| operation | CPU (Accelerate BLAS) | GPU (Metal / MLX) | speed-up |
|---|---|---|---|
| signal propagation, 20 000 cells × 2 000 genes × 3 steps | 0.833 s | 0.244 s | **3.4×** |
| `colDeltaCorpartial`, 2 000 genes × 5 000 cells × 60 neighbours | 3.334 s | 0.171 s | **19.5×** |

End-to-end, the bundled Gata1-knockout tutorial (24 cluster-specific GRNs,
simulation, vector field, gradient and perturbation scores) completes in ~40 s.

---

## 3. Packaging

```bash
# wheel + sdist (requires `pip install build`)
python -m build            # or: uv build
ls dist/
#   celloracle-0.22.2-py3-none-any.whl
#   celloracle-0.22.2.tar.gz

# install into any environment
pip install dist/celloracle-0.22.2-py3-none-any.whl
pip install "dist/celloracle-0.22.2-py3-none-any.whl[gpu]"
```

`setup.py` / `pyproject.toml` now use `find_packages()`, declare the optional
extras, allow Python 3.9 – 3.13, and build with `setuptools>=64`.

The large tutorial / motif datasets are **not** baked into the wheel (same
policy as upstream): they are downloaded on demand. To build a self-contained,
fully offline distribution bundle (wheel + sdist + data archive), run:

```bash
./scripts/make_bundle.sh          # writes to ./dist
```

The resulting `celloracle-data-0.22.2.tar.gz` (~620 MB) contains
`anndata/`, `TFinfo_data/`, `promoter_base_GRN/`, `tutorial_data/` and
`motif_data/` at the top level, so it can be extracted straight into the
default data directory:

```bash
mkdir -p ~/celloracle_data
tar -xzf dist/celloracle-data-0.22.2.tar.gz -C ~/celloracle_data
```

The wheel itself installs and runs end-to-end on a clean environment; without
the data archive the tutorial datasets are fetched from GitHub on first use.

---

## 4. How to verify

```bash
# unit tests for the accelerator + modern-stack behaviour
pytest celloracle/tests/test_accelerator.py celloracle/tests/test_modern_stack.py

# data loading + network scores
pytest celloracle/tests/data/test_load_data.py celloracle/tests/network_analysis/
```

The bundled tutorial suites (`celloracle/tests/tutorial/`) also run end-to-end;
`test_05_simulation_*` exercises simulation, vector fields, gradient and
perturbation scores.
