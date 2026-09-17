# Lab 2 — Model Training and Experiment Tracking with MLflow

Repo: https://github.com/michelabourahal/mlops-lab-1

This lab continues the project from [Lab 1](lab1.md): a git+dvc repo with the raw and
processed Food-11 datasets tracked. Here we write the training code, run a local MLflow
tracking server, log parameters and metrics for each training run, and compare runs in the
MLflow UI.

## Background

- MLflow organizes tracking into **experiments** (a named group of runs, e.g. `food11`) and
  **runs** (one training execution with its own params, metrics, and artifacts).
- A **tracking server** stores this metadata and serves the UI; without one, MLflow just
  writes to a local `./mlruns` folder.
- A **param** is a value set before training and fixed for the run (learning rate, batch size,
  model architecture, ...). A **metric** is a value produced during or after training that can
  evolve over time (loss, accuracy, ...).
- **Autologging** can capture most of this automatically for common frameworks, but logging
  explicitly gives control over exactly what gets recorded and when.

## Setup: install MLflow and the training libraries

```bash
uv add mlflow torch torchvision scikit-learn
```

This machine has no NVIDIA GPU (Intel integrated graphics only), so `pyproject.toml` routes
`torch`/`torchvision` to the CPU-only wheel index before running `uv add`:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
torchvision = { index = "pytorch-cpu" }
```

Confirmed installed versions: `torch==2.14.0+cpu`, `torchvision==0.29.0+cpu` (the `+cpu` suffix
confirms the CPU-only build was used, not the multi-GB CUDA one), `mlflow==3.16.1`,
`scikit-learn==1.9.1`.

## Lab questions and answers

### Q1 — Look at `pyproject.toml` and `uv.lock`. What changed?

**`pyproject.toml`:**
- `dependencies` gained four entries: `mlflow>=3.16.1`, `scikit-learn>=1.9.1`,
  `torch>=2.14.0`, `torchvision>=0.29.0` (alongside the pre-existing `pillow`).
- A new `[[tool.uv.index]]` block registers `pytorch-cpu`
  (`https://download.pytorch.org/whl/cpu`) as an **explicit** index — explicit means uv only
  pulls packages from it for names specifically routed there, not for every dependency.
- A new `[tool.uv.sources]` block routes `torch` and `torchvision` specifically to that
  `pytorch-cpu` index, overriding the default PyPI resolution for just those two packages.

**`uv.lock`:** grew by roughly 3,280 lines / 103 new package entries — the full CPU-only
dependency tree these four packages pull in. Notable details:

- `torch` and `torchvision` resolve to `source = { registry =
  "https://download.pytorch.org/whl/cpu" }` with version strings like `2.14.0+cpu` /
  `0.29.0+cpu` on Windows/Linux — the `+cpu` local version segment is the tell that the
  CPU-only wheel was selected. Since `uv.lock` is a cross-platform lock, it carries two
  resolutions for these packages: a plain `2.14.0`/`0.29.0` for `sys_platform == 'darwin'`
  (macOS has no CUDA variant to distinguish, so no `+cpu` suffix) and the `+cpu`-suffixed one
  for `win32`/Linux — both still pinned to the same `pytorch-cpu` registry.
- `mlflow` pulled in its own sizeable stack: `mlflow-skinny`, `mlflow-tracing`, `flask`,
  `sqlalchemy`, `alembic` (its backing DB), `pyarrow`, `pandas`, `graphene`/`graphql-core`
  (GraphQL API), `opentelemetry-*` (tracing), `gunicorn`, `docker`, `databricks-sdk`, etc.
- `scikit-learn` brought in `scipy`, `joblib`, `threadpoolctl`.
- `torch` itself brought in `sympy`, `networkx`, `filelock`, `fsspec`, `mpmath`, `jinja2`.
