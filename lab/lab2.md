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

## Running the local MLflow tracking server

```bash
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

Left running in the background; `http://127.0.0.1:5000` responds with HTTP 200 and shows a
single, empty **`Default`** experiment (id `0`), as expected. `mlflow.db`, `mlruns/`, and the
server log are runtime state, not code — added to `.gitignore` rather than committed, the same
way `data/` is DVC-managed instead of git-managed.

### Q2 — What is `--backend-store-uri` for? What is `--default-artifact-root` for? What's the difference between the metadata MLflow stores and the artifacts it stores?

MLflow splits everything it tracks into two categories, stored in two different places:

- **Metadata** — structured, queryable facts about experiments and runs: experiment/run IDs,
  names, start/end times, status, **params**, **metrics** (with their step/timestamp history),
  tags, and pointers (like a run's artifact location or a registered model version). This is
  what the UI's tables, charts, and comparison views are built from. `--backend-store-uri`
  tells the server **where to store this metadata** — here, `sqlite:///mlflow.db`, a local
  SQLite database file at the repo root. (It also accepts a plain local directory for a
  simple file-based store, or a real database URI like `postgresql://...`/`mysql://...` for
  multi-user/production setups.)

  We can see this directly: `mlflow.db` is a real SQLite database (confirmed via
  `sqlite3`) with tables like `experiments`, `runs`, `params`, `metrics`, `tags`,
  `registered_models`, etc. Right now `experiments` has exactly one row — `(0, 'Default',
  'file:///.../mlruns/0')` — created the moment the server started, before any run existed.

- **Artifacts** — arbitrary files a run produces or consumes: model weights/checkpoints,
  plots, sample predictions, a `requirements.txt` snapshot, the dataset used, etc. These are
  not queryable rows in a database; they're just files. `--default-artifact-root` tells the
  server **where to physically store these files** by default — here, `./mlruns`, a local
  folder. (Just like the backend store, this can point elsewhere — S3, Azure Blob Storage,
  GCS, DagsHub, etc. — for shared/remote access.)

  The metadata row for the `Default` experiment already records its artifact location as
  `file:///C:/Users/miche/OneDrive/Desktop/mlops-lab-1/mlruns/0` — but the `mlruns/` folder
  itself doesn't exist on disk yet, since no run has actually logged an artifact there. This
  is the clearest illustration of the split: the *metadata* ("this experiment's artifacts will
  live here") is created eagerly in the database, while the *artifact storage* itself is only
  materialized lazily, on first actual write.

In short: `--backend-store-uri` is "where do the facts and numbers go" (a database, queried by
the UI/API), and `--default-artifact-root` is "where do the actual files go" (a filesystem or
object store, referenced by path/URI from the metadata but not itself indexed or queryable).

### Q3 — Why shouldn't `mlflow.db` and `mlruns/` be tracked by git, and why not by DVC either?

**Not git:** `mlflow.db` is a SQLite file that gets rewritten on essentially every training
run — every logged param, every metric step, every tag is a write to the same binary file.
Git diffs and merges text line-by-line; a binary database has no meaningful diff, so every
commit would just replace the whole blob, bloating the repo with unreadable history and
guaranteeing merge conflicts the moment two people (or two runs) touch it. `mlruns/` is the
same problem for artifacts: it can hold arbitrarily large files (model checkpoints, images)
that are *generated*, not authored — exactly the category `__pycache__/` and `.venv/` are
already excluded for elsewhere in this `.gitignore`. Generated, machine-specific, constantly
mutating local state doesn't belong in source control.

**Not DVC either**, but for a different reason than "it's data": DVC is built to version a
*fixed, shareable snapshot* — `dvc add data` freezes `data/` at a point in time so anyone can
`dvc pull` the exact same bytes back. `mlflow.db`/`mlruns/` aren't a snapshot of anything; they
*are* the live tracking log itself, growing with every run anyone does locally. There's no
single meaningful "version" of them to pin — you'd need a new `dvc add` after literally every
run, which defeats the point (and would still hit the same binary-diff-storage problem inside
DVC's own cache). More fundamentally, MLflow already **is** the versioning/tracking system for
this kind of data — layering DVC on top of it to snapshot MLflow's own bookkeeping is using the
wrong tool for a job that's already solved.

There's also concrete proof in this repo that copying these files between machines wouldn't
even work correctly: the `experiments` row for `Default` in `mlflow.db` stores its artifact
location as the *absolute local path* `file:///C:/Users/miche/OneDrive/Desktop/mlops-lab-1/mlruns/0`.
If `mlflow.db` were checked out on a teammate's machine (via git or DVC), every artifact
lookup would try to resolve that path on *their* filesystem and fail unless their username and
folder layout happened to match exactly.

The right way to share run results across a team isn't versioning these files at all — it's
pointing everyone's MLflow client at one shared tracking server (a real database backend
reachable by everyone, plus a shared artifact store like S3/DagsHub), the same role a remote
plays for DVC.
