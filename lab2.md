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

## Pointing code at the tracking server

```python
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("food11")
```

### Q4 — What happens the first time you call `set_experiment` with a name that doesn't exist yet?

Tested directly against the running server:

```
>>> mlflow.set_experiment("food11")
INFO mlflow.tracking.fluent: Experiment with name 'food11' does not exist. Creating a new experiment.
experiment_id: 1
name: food11
artifact_location: file:///C:/Users/miche/OneDrive/Desktop/mlops-lab-1/mlruns/1
lifecycle_stage: active
```

MLflow doesn't error — it silently **creates the experiment** (auto-increments to the next
free experiment ID, `1`, since `0` is already `Default`), assigns it a default artifact
location under the tracking server's artifact root (`mlruns/1`), and returns it, all in one
call. Checking the UI/API afterward confirms `food11` now shows up alongside `Default`.

Two follow-on details worth noting:

- **It's idempotent.** Calling `set_experiment("food11")` again afterward does *not* log the
  "does not exist" message or create a second experiment — it just looks it up by name and
  returns the same `experiment_id: 1`. This is what makes it safe to put at the top of a
  training script that gets run many times.
- **Still no `mlruns/` folder on disk** at this point, same as Q2's observation for `Default`
  — creating the experiment only writes a metadata row (in `mlflow.db`) recording where its
  artifacts *will* go; the artifact directory itself is only materialized once an actual run
  under this experiment logs something.

## Training script

`src/food11/train.py` loads `food11_processed`/`food11_processed_mini` via `ImageFolder` +
`DataLoader`, fine-tunes a pretrained `resnet18` (final layer swapped to 11 classes), and logs
everything to the tracking server inside `with mlflow.start_run():`. Ran:

```bash
uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.001 --batch-size 32
```

This surfaced two real bugs in the model-logging step, both fixed in the script: `mlflow.pytorch.log_model` needed an `input_example` (this MLflow version defaults to the `pt2`
export format, which traces the model graph and therefore requires a sample input), and
MLflow's end-of-run emoji banner crashed on Windows' default console encoding (fixed by
forcing stdout/stderr to UTF-8). Verified run `641a7744dffb4d89a861375310863e7e`
(`mercurial-ray-845`) under the `food11` experiment: 5 epochs of `train_loss`/`val_loss`/
`val_accuracy`, final `test_accuracy=0.6022`, and a logged model artifact.

### Q5 — What's the difference between `mlflow.log_param` and `mlflow.log_metric`? Why does `log_metric` take a `step` argument and `log_param` doesn't?

`mlflow.log_param` records a value that's **fixed for the entire run** — a hyperparameter
chosen before training starts (`lr`, `batch_size`, `dataset`, ...) that never changes once
training begins. `mlflow.log_metric` records a value that's **expected to evolve during** the
run — something re-measured at multiple points (`train_loss`, `val_loss`, `val_accuracy` once
per epoch here).

That difference is exactly why `log_metric` takes `step` and `log_param` doesn't: a metric is a
*time series* — the same key gets logged multiple times over the course of a run (our script
calls `mlflow.log_metric("train_loss", train_loss, step=epoch)` once per epoch, five times per
run), and `step` is what tells MLflow which point in training each value belongs to, so the UI
can plot it as a chart instead of just overwriting the previous value. A param, by contrast, is
logged exactly once and stays constant — there's no "which point in time" for a single fixed
value to attach to, so `log_param` has no `step` and calling it twice for the same key in one
run is actually an error (params are meant to be immutable per run), whereas calling
`log_metric` many times for the same key is the normal, expected usage.

### Q6 — Open the run in the mlflow UI. Find the params, the metric charts, and the logged model artifact. Where does the model artifact actually live on disk?

*(Using the MLflow API/filesystem directly here — screenshots of the actual UI to follow.)*

**Params** (`GET /runs/get`, run `641a7744dffb4d89a861375310863e7e`): `dataset=mini`,
`epochs=5`, `lr=0.001`, `batch_size=32`, `model=resnet18`, `num_classes=11`,
`train_samples=1100`, `val_samples=1096`, `test_samples=1096` — this is exactly what the UI's
run page **Parameters** table renders.

**Metric charts**: `GET /metrics/get-history?run_id=...&metric_key=train_loss` returns the full
time series, one point per epoch, e.g.:

```
step=0  train_loss=1.8652
step=1  train_loss=1.0173
step=2  train_loss=0.6704
step=3  train_loss=0.4088
step=4  train_loss=0.3049
```

This `(step, value)` history for each of `train_loss`, `val_loss`, `val_accuracy` (plus the
single-point `test_accuracy`) is precisely the data the UI's **Metrics** tab turns into line
charts — `step` on the x-axis, value on the y-axis, one line per metric key.

**Where the model artifact lives on disk**: the run's own `artifact_uri` is
`mlruns/1/641a7744.../artifacts`, but that folder stayed empty — in this MLflow version,
`mlflow.pytorch.log_model` creates a separate first-class **logged model** entity (its own
`model_id`, linked to the run via metadata rather than nested inside the run's plain artifact
folder). Its actual files are at:

```
mlruns/1/models/m-bc15fb3cd7984d39b7c3a4cc25e3f529/artifacts/
├── MLmodel              # model metadata: flavor, signature, run_id, model_id, size
├── conda.yaml / python_env.yaml / requirements.txt   # environment for reloading the model
├── data/model.pt2       # the actual serialized weights (pt2 = torch.export trace format)
├── input_example.json
└── serving_input_example.json
```

The `MLmodel` file itself confirms the link back to this run (`run_id:
641a7744dffb4d89a861375310863e7e`) and records an auto-inferred signature from the
`input_example` we passed: input `[-1, 3, 128, 128]` (a batch of RGB 128×128 images), output
`[-1, 11]` (per-class logits) — this is what the UI's **Artifacts** tab uses to render the
model's schema without needing to load the weights.
