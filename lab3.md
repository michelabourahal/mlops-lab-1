# Lab 3 — Containerizing the model with Docker

Repo: https://github.com/michelabourahal/mlops-lab-1

This lab continues the project from [Lab 1](lab1.md) (git+dvc data) and [Lab 2](lab2.md)
(training + MLflow tracking). The best run from Lab 2's sweep — `marvelous-stag-139`
(`lr=0.0001`, `batch_size=32`, `val_accuracy=0.7126`, run id
`935472bb46f8451f88f473130c6c44ef`) — is registered here, wrapped in a FastAPI serving app,
and packaged into a Docker image.

## Register the best model

Used the code path (Option B), pointing at the Lab 2 sweep winner:

```bash
uv run python -c "
import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.register_model('runs:/935472bb46f8451f88f473130c6c44ef/model', 'food11')
"
```

```
Successfully registered model 'food11'.
Created version '1' of model 'food11'.
```

### Q1 — What version number was your model given? What's the difference between a run's logged model artifact and a registered model?

**Version 1** — the first version of the newly created `food11` registered model.

A run's logged model artifact (what `mlflow.pytorch.log_model` wrote in Lab 2, at
`mlruns/1/models/m-.../artifacts/`) is just files tied permanently to that one run — useful
for reproducing exactly what that run produced, but with no concept of "this is the one
serving traffic" or "this supersedes that older one." A **registered model** is a separate,
named entity (`food11`) in the Model Registry that points *at* a specific run's artifact but
adds a layer on top: version numbers that increment independently of run IDs, aliases (below),
tags, and a stable URI (`models:/food11@champion`) that serving code can depend on without
caring which run produced the current version. Registering is the step that turns "a run that
happened to produce a decent model" into "the model my API loads."

### Assign it the `champion` alias

```bash
uv run python -c "
import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')
client = mlflow.MlflowClient()
client.set_registered_model_alias('food11', 'champion', 1)
"
```

Verified: `client.get_model_version_by_alias('food11', 'champion')` returns version `1`,
`run_id=935472bb46f8451f88f473130c6c44ef`.

### Q2 — What aliases replaced the old built-in stages? Why version a model separately from the run that produced it, and why is an alias more flexible than a fixed stage name?

Current mlflow has no built-in alias names at all — `Staging`/`Production` weren't swapped
for a different fixed pair, they were replaced by **arbitrary, user-defined aliases**
(`champion`, `challenger`, `shadow`, or anything else you want to call a pointer). This repo
uses `champion` for "the version `serve.py` loads."

Versioning the model separately from the run matters because a run is a training *execution*
record — full of things irrelevant to serving (hyperparameters tried, intermediate metrics,
which dataset variant was used) — while a registered version is a *deployable unit* with its
own lifecycle: something can be promoted, rolled back, or compared without touching the
training history that produced it. Several runs might even resolve to model versions that get
compared before one is ever aliased `champion`.

An alias beats a fixed stage name because it's just a mutable pointer with no hardcoded
meaning to mlflow itself — moving `champion` from version 1 to version 2 is a single
`set_registered_model_alias` call with no "transition" workflow, and nothing stops defining
`champion`, `challenger`, and `canary` simultaneously on different versions for A/B testing,
something the old fixed `Staging`/`Production` enum couldn't express.

## Serving script — `src/food11/serve.py`

FastAPI app exposing `GET /health` and `POST /predict`. Loads the model once at process
startup via a `lifespan` context manager (not per-request), reads `MLFLOW_TRACKING_URI` from
the environment (defaulting to `http://127.0.0.1:5000`), and reuses `CLASS_NAMES` from
`src/food11/data.py` — the same source of truth `train.py`'s `ImageFolder` sorts classes from
— instead of re-typing the category list.

```bash
uv add fastapi uvicorn python-multipart
uv run uvicorn src.food11.serve:app --host 0.0.0.0 --port 8000
```

Tested locally against three classes before touching Docker at all:

```
Dairy product -> {"category":"Dairy product","confidence":0.7009}
Seafood       -> {"category":"Seafood","confidence":0.9131}
Soup          -> {"category":"Soup","confidence":0.9863}
```

### Q3 — Why load the model through an mlflow model URI instead of pointing at the `.pth` file directly? What would you have to change to serve a newer model version?

Loading `models:/food11@champion` instead of a `.pth` path means `serve.py` never needs to
know *which* run produced the current model, what its hyperparameters were, or where its
weights physically live — mlflow resolves all of that from the registry at startup. A raw
`.pth` file is also just weights: `mlflow.pyfunc.load_model` instead reconstructs a full
`pyfunc` model (architecture + weights + the preprocessing/signature metadata logged
alongside it), so `serve.py` calls `.predict()` on something self-describing rather than
having to separately reconstruct a bare `resnet18` and hope its shape matches whatever `.pth`
happens to be on disk.

To serve a newer version, **nothing in `serve.py` or the Docker image changes** — only
`client.set_registered_model_alias('food11', 'champion', <new_version>)` moves the pointer.
The next container restart (or the next time a long-lived process reloads it) picks up the
new version automatically, which is the entire point of aliasing instead of hardcoding a
version number or path.

## Dockerfile

Multi-stage build. One wrinkle not anticipated by the lab: this repo's `pyproject.toml` keeps
`mlops-lab-1` as an installable project (`[project.scripts] mlops-lab-1 = "mlops_lab_1:main"`,
no `[tool.uv] package = false`) rather than the `package = false` layout described in
[Lab 1](lab1.md) Q1 — this reverted with the "restore `mlops_lab_1` as its entrypoint" commit.
That means `uv sync` tries to **build the local project itself**, which needs `src/` and
`README.md` present — running it before `COPY src/` (as the lab suggests) fails outright.
Fixed with the standard two-step uv pattern: sync dependencies only first
(`--no-install-project`), then copy source and sync again (fast, since deps are already
cached):

```dockerfile
FROM python:3.11-slim AS builder
RUN pip install --no-cache-dir uv
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/src ./src

ENV PATH="/app/.venv/bin:$PATH" \
    MLFLOW_TRACKING_URI=http://127.0.0.1:5000

EXPOSE 8000
CMD ["uvicorn", "src.food11.serve:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Q4 — Why copy `pyproject.toml`/`uv.lock` and run `uv sync` before copying the rest of the source, instead of copying everything at once? What happens to the cache when only `serve.py` changes?

Docker caches each layer and only reinvalidates a layer (and everything after it) when its
own inputs change. If `COPY . .` happened before `uv sync`, editing *any* file — even a
comment in `serve.py` — would change the build context hash for that `COPY` layer and force
`uv sync` to rerun from scratch, redownloading and reinstalling the entire ~1.4GB
torch/mlflow/scikit-learn dependency tree on every rebuild. Splitting it as done here means
the expensive `COPY pyproject.toml uv.lock ./` + `RUN uv sync ... --no-install-project` layer
only reruns when a dependency actually changes.

Confirmed empirically: after editing `serve.py` and rebuilding, `docker build` output showed
`CACHED` for both the `pyproject.toml`/`uv.lock` copy and the dependency-only `uv sync` layer
— only `COPY README.md`, `COPY src/`, the second `uv sync` (fast — deps already installed),
and everything in the runtime stage reran.

## .dockerignore

```
.venv/
data/
mlruns/
mlflow.db
mlflow_server.log
serve_local.log
.git/
.dvc/cache/
.dvc/tmp/
__pycache__/
*.pyc
screenshots/
lab1.md
lab2.md
lab3.md
```

### Q5 — What's the size difference between a naive single-stage image and the multi-stage one? Which layers are biggest?

Built a throwaway single-stage comparison (`pip install uv`, `COPY . .`, `uv sync
--frozen --no-dev` all in one `python:3.11-slim` stage, no separate runtime stage):

| Image | Size |
|---|---|
| Naive single-stage | **2.15GB** |
| Multi-stage (`food11-api:latest`) | **2.01GB** |

A ~140MB (~6.5%) saving — real, but modest, because the dominant cost is unavoidable
regardless of build strategy: in both images the dependency install layer is by far the
largest single layer (`RUN uv sync` in the naive image: **1.48GB**; `COPY /app/.venv
./.venv` in the multi-stage image: **1.44GB**) — the CPU-only `torch`/`torchvision` build
plus mlflow's own dependency stack (pandas, pyarrow, sqlalchemy, scikit-learn, etc.) are
identical in both since both resolve the same `uv.lock`. What multi-stage actually buys is
**not shipping the build tools**: the naive image's `RUN pip install --no-cache-dir uv`
layer (**64MB**) plus the apt `build-essential`-adjacent layers from the base image sit
permanently in the final image; the multi-stage runtime stage never installs `uv` or `pip`
build machinery at all — only the already-built `.venv` and `src/` are copied across.

### Q6 — What happens to build speed/size without `.dockerignore`? Which excluded folders would break the build?

Without it, the entire `data/` folder (~1.27GB, 36k+ files per [Lab 1](lab1.md)) and
`.dvc/cache/` (DVC's own content-addressable store, comparably large) would be sent to the
Docker daemon as build context on *every* build — even though neither `COPY` instruction in
the Dockerfile ever references them, since only `pyproject.toml`, `uv.lock`, `README.md`, and
`src/` are copied. That's pure wasted I/O and time on every invocation, not a correctness
issue by itself.

`.git/` is the one that could actually **break** things if left in: `docker build`'s context
transfer has no reason to fail on it, but including multi-hundred-MB of git history/objects in
the tarball sent to the daemon needlessly risks hitting context-size limits on constrained
CI runners, and there's no scenario where the build needs it (only working-tree files are
copied, never `.git` internals). `mlflow.db`/`mlruns/` aren't referenced either, but leaving
them in would risk `COPY . .`-style Dockerfiles accidentally baking a snapshot of local
tracking state into the image — not our multi-stage Dockerfile (which copies explicit paths),
but a real risk in a less careful one.

## Build and run

```bash
docker build -t food11-api:latest .
```

The host's mlflow server needed two fixes beyond what the lab anticipated before the
container could actually serve a prediction — both are genuine, verified findings, not
guesses:

**1. MLflow's host-header security middleware.** This mlflow version (`3.16.1`) ships a
security middleware, on by default, that rejects requests whose `Host` header isn't
localhost/a private IP (`Invalid Host header - possible DNS rebinding attack detected`,
`mlflow/server/security_utils.py`). Since the container reaches the server as
`host.docker.internal:5000`, that hostname has to be added explicitly:

```bash
uv run mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns \
  --allowed-hosts "localhost,localhost:*,127.0.0.1,127.0.0.1:*,host.docker.internal:*"
```

**2. The artifact root is a local filesystem path, not a proxied one.** Lab 2 started the
server with `--default-artifact-root ./mlruns` — a *local* path. `mlflow.pyfunc.load_model`
resolves the champion version's artifacts to an absolute URI,
`file:///C:/Users/.../mlops-lab-1/mlruns/1/models/.../artifacts` (verified via
`get_logged_model(...).artifact_location`), and because that scheme is `file://` (not
`mlflow-artifacts://`), the mlflow **client itself** opens it directly off disk rather than
proxying the bytes through the tracking server's HTTP API. A container has no access to the
host's `C:\` filesystem by default, so `load_model` failed with `No such artifact: ''` even
though the server's metadata API was perfectly reachable. Fixed by bind-mounting `mlruns` into
the container at the *same absolute path* the metadata already points to, so the `file://` URI
resolves identically inside the container:

```bash
docker run -d --name food11-api -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  --mount type=bind,source="C:\Users\miche\OneDrive\Desktop\mlops-lab-1\mlruns",target=/C:/Users/miche/OneDrive/Desktop/mlops-lab-1/mlruns \
  food11-api:latest
```

This is really the same lesson as [Lab 2](lab2.md) Q2/Q3 (metadata vs. artifacts, why local
state doesn't survive leaving the machine it was created on) taken one step further: it's not
just that `mlruns/` shouldn't be *versioned* elsewhere, it's that a **local-disk artifact
root can't be read by any client that isn't the same machine/filesystem as the one that wrote
it** — a container included. A real deployment would point `--default-artifact-root` at a
proxied (`mlflow-artifacts:/`) or remote (S3/Azure Blob/GCS) store precisely to avoid this.

```bash
curl -X POST -F "file=@data/food11_processed_mini/validation/Soup/<file>.jpg" http://127.0.0.1:8000/predict
# {"category":"Soup","confidence":0.9863488674163818}
```

### Q7 — Why can't the container use `127.0.0.1:5000` to reach the host's mlflow server? What does `host.docker.internal` resolve to?

A container gets its own network namespace, isolated from the host's by default — inside the
container, `127.0.0.1` means *the container itself*, not the machine it's running on, so
nothing is listening there. `host.docker.internal` is a special DNS name Docker Desktop
injects into the container's network that resolves to the **host machine's** IP address as
seen from inside the container's virtual network, giving containerized code a stable way to
reach services bound to the host's loopback interface without knowing the host's real IP.
(On native Linux Docker, without Docker Desktop, this name isn't available by default —
`--network host` or `--add-host=host.docker.internal:host-gateway` is used instead.)

### Q8 — Stop the container and start a new one from the same image. Does the model still load without rebuilding? What does that tell you about what's baked in vs. fetched at runtime?

Yes — verified directly: stopped `food11-api`, ran a second container (`food11-api-2`) from
the exact same `food11-api:latest` image with no rebuild, and it served an identical
prediction for the same `Seafood` test image (`confidence=0.9130866527557373`, matching the
first container exactly). This confirms the image itself contains **no model weights at
all** — only the Python environment and `serve.py`. The actual model (architecture + weights)
is fetched fresh from the mlflow tracking server + artifact store on every container start,
via `models:/food11@champion`. That's the entire point of loading through a registry URI
(Q3): the image is reusable across any model version without rebuilding, and conversely, a
new container from an old image will always serve whatever version `champion` currently
points to — not whatever was current when the image was built.

## Commit

```bash
git add Dockerfile .dockerignore src/food11/serve.py pyproject.toml uv.lock lab3.md
git commit -m "Containerize model serving with Docker"
git push
```

### Q9 — What's still missing before another machine could reliably pull and run the exact image just built?

The Dockerfile is versioned in git; the *image* built from it is not — `food11-api:latest`
only exists in this machine's local Docker image cache right now. For a CI runner or a
Kubernetes cluster to run the exact bytes tested here (not just "an image built from the same
Dockerfile, possibly with different base-image or dependency versions resolved at a later
date), it would need to be **pushed to a registry** (Docker Hub, GHCR, ECR, etc.) and referenced
by an immutable **digest** (`food11-api@sha256:...`), not just the mutable `latest` tag, plus
CI credentials configured to pull it. Beyond the image itself, the two Windows-specific fixes
above (`--allowed-hosts`, and the local-artifact-root bind mount) are host-specific workarounds
that wouldn't transfer to a different machine unmodified — a production setup needs the mlflow
server's `--allowed-hosts` to include whatever hostname the *real* deployment environment uses
(not `host.docker.internal`), and a real remote artifact store so no path-mounting hack is
needed at all.
