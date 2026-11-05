# Lab 4 — Orchestrating the app with Docker Compose

Repo: https://github.com/michelabourahal/mlops-lab-1

This lab continues the project from [Lab 1](lab1.md) (git+dvc data), [Lab 2](lab2.md)
(training + MLflow tracking) and [Lab 3](lab3.md) (a single Dockerfile serving the model).
mlflow itself is now containerized, a Streamlit frontend is added, and all three services are
brought up together by Docker Compose:

| Service | Built from | Published port | Talks to |
|---|---|---|---|
| `mlflow` | `./mlflow/Dockerfile` | `5000` (UI, and `train.py` on the host) | — |
| `inference` | `./Dockerfile` (Lab 3, unchanged) | none | `http://mlflow:5000` |
| `frontend` | `./frontend/Dockerfile` | `8501` (browser) | `http://inference:8000` |

## Deviations from the lab text

Same Windows + Docker Desktop + mlflow `3.16.1` setup as Lab 3, so the same two mlflow
server issues from Lab 3 come back, plus one naming change:

- **`--artifacts-destination` instead of `--default-artifact-root`.** With a plain local
  artifact root, the server records artifact locations as `file:///mlflow-data/mlruns/...` —
  a path that only exists inside the *mlflow* container — and the mlflow client in the
  inference container tries to open it off its own disk, exactly the `No such artifact`
  failure from Lab 3. `--artifacts-destination` makes the server store artifacts on the volume
  but hand out `mlflow-artifacts:/` URIs, so every client uploads/downloads model files over
  HTTP through the server. This is the "proxied artifact store" Lab 3 Q9 said a real deployment
  needs — and it removes Lab 3's absolute-path bind-mount hack entirely.
- **`--allowed-hosts "mlflow:5000,localhost:5000,127.0.0.1:5000"`.** mlflow 3's host-header
  middleware (Lab 3, fix 1) would otherwise reject the inference container's requests, whose
  `Host` header is now `mlflow:5000`.
- **`mlflow==3.16.1` pinned** in `mlflow/Dockerfile`, the same version `uv.lock` resolves for
  `train.py` and the inference image, so server and clients run the same API.
- **`RUN mkdir -p /mlflow-data`**, so the image also runs standalone without a volume (Q1).
- **Staging → `champion` alias.** As in Lab 3 Q2, stages are deprecated and `serve.py` loads
  `models:/food11@champion`; wherever this lab says "move to Staging", the `champion` alias was
  moved instead.
- **Fresh registry.** The containerized mlflow starts with an empty database on its own volume
  (the host's `mlflow.db`/`mlruns/` from Labs 2–3 aren't visible to it), so the best Lab 2
  configuration was retrained against it — `placid-grub-210` (`lr=0.0001`, `batch_size=32`,
  `epochs=5`, `test_accuracy=0.7473`, close to Lab 2's winner) — registered as `food11`
  version 1 and given the `champion` alias. `train.py` needed no change: its hardcoded
  `http://127.0.0.1:5000` now hits the mlflow container through the published port.

`serve.py`, the Lab 3 `Dockerfile` and `.dockerignore` are otherwise unchanged (`lab4.md` was
added to `.dockerignore` alongside the other lab docs).

## The mlflow service — `mlflow/Dockerfile`

```dockerfile
FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==3.16.1
RUN mkdir -p /mlflow-data

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow-data/mlflow.db", \
     "--artifacts-destination", "/mlflow-data/mlruns", \
     "--allowed-hosts", "mlflow:5000,localhost:5000,127.0.0.1:5000"]
```

### Q1 — What happens to everything written to `/mlflow-data` if you never mount a volume there and just `docker run` this image standalone?

It's lost with the container. Without a volume, `/mlflow-data` is just a directory in that one
container's writable layer. Tested with the Compose mlflow stopped to free port 5000:

```bash
docker run -d --name mlflow-q1 -p 5000:5000 mlops-lab-1-mlflow
# created experiment "q1-test" against it
docker stop mlflow-q1 && docker rm mlflow-q1
docker run -d --name mlflow-q1 -p 5000:5000 mlops-lab-1-mlflow
```

```
first container:  experiments: ['q1-test', 'Default']
new container:    experiments: ['Default']
                  registered models: []
```

The second container, started from the identical image, has only mlflow's auto-created
`Default` experiment. The image holds the mlflow *installation*, never the data written at
runtime, so every new container from it starts from an empty SQLite database. This is the
same baked-in vs. runtime split as Lab 3 Q8, seen from the other side: there it was good news
(the model isn't in the image, so it can change without a rebuild), here it means state has to
live outside the container.

### Q2 — Why a named volume instead of a bind mount to a folder in the repo? Would a bind mount work just as well?

A bind mount would *work* for local dev, and is easier to inspect (the files just appear in
the repo folder), but a named volume fits better here:

- **No host path in the compose file.** `mlflow-data:/mlflow-data` means the same thing on any
  machine; a bind mount hardcodes a host directory — the exact kind of host-specific path that
  forced Lab 3's `C:\Users\miche\...` → `/C:/Users/miche/...` mount workaround.
- **Runtime state stays out of the repo.** Lab 2 had to `.gitignore` `mlflow.db`/`mlruns/` so
  tracking state wouldn't get committed; a named volume never lands in the working tree at all.
  Same separation as `data/` vs. git in Lab 1: the repo holds code and pointers, not state.
- **Windows-specific: SQLite on a bind mount.** On Docker Desktop, a named volume lives inside
  the Linux VM's filesystem, while a bind mount of a Windows folder (here, one inside OneDrive)
  goes through a file-sharing layer that is slower and has weaker file-locking semantics —
  a known source of `database is locked` errors for SQLite.

## Update the inference service

No code change. `serve.py` already reads `MLFLOW_TRACKING_URI` (default
`http://127.0.0.1:5000`) and Compose overrides it with `http://mlflow:5000`.

### Q3 — In Lab 3 the container had to use `host.docker.internal`. Why does `http://mlflow:5000` resolve now?

`docker compose up` creates a project network (`mlops-lab-1_default`), attaches every service
to it, and runs an embedded DNS server that resolves each **service name** to that service's
container IP on the network. In Lab 3, mlflow wasn't a container — it was a process on the
host, and the inference container sat on Docker's default bridge network, where no container
named `mlflow` existed (and where containers don't get name-based DNS anyway), so the only way
out was `host.docker.internal`.

The mlflow server log shows the inference container (`172.18.0.3` on the Compose network)
resolving `mlflow`, then downloading the model over HTTP via the proxied artifact store:

```
mlflow-1  | INFO: 172.18.0.3:38266 - "GET /api/2.0/mlflow-artifacts/artifacts/1/models/m-dd516f1908024a448429c195cf0fe9b9/artifacts/MLmodel HTTP/1.1" 200 OK
mlflow-1  | INFO: 172.18.0.3:38244 - "GET /api/2.0/mlflow-artifacts/artifacts/1/models/m-dd516f1908024a448429c195cf0fe9b9/artifacts/data/model.pt2 HTTP/1.1" 200 OK
inference-1  | INFO:     Application startup complete.
inference-1  | INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

(Requests from `train.py` on the host show up as `172.18.0.1` instead — the network's
gateway, i.e. traffic arriving through the published `5000:5000` port.)

Even before any model existed, the very first `docker compose up` already proved name
resolution worked: inference exited with `RESOURCE_DOES_NOT_EXIST: Registered Model with
name=food11 not found` — an answer *from* the mlflow server, not a connection error. The name
only resolves while the service is running, though: with mlflow stopped (Q6), the same lookup
fails with `Failed to resolve 'mlflow'`.

## The frontend service — `frontend/app.py`, `frontend/Dockerfile`

Taken as given in the lab: a Streamlit upload page that POSTs the image to
`{INFERENCE_URL}/predict`, on `python:3.11-slim` with `streamlit` and `requests`.

### Q4 — Why read `INFERENCE_URL` from an environment variable instead of hardcoding `http://inference:8000`?

Because `inference` is only a hostname *inside this Compose project's network*. `docker run`
the frontend image on its own and it lands on the default bridge network, where `inference`
doesn't resolve — every upload would fail. Confirmed directly in Q6: with the inference
container gone, the frontend's request fails with `NameResolutionError: Failed to resolve
'inference'`.

With an environment variable, the same image works everywhere and only configuration changes:
Compose sets `INFERENCE_URL=http://inference:8000`; a standalone container can be given
`-e INFERENCE_URL=http://host.docker.internal:8000`; running `streamlit run app.py` directly on
the host uses the `http://127.0.0.1:8000` default. The same reasoning is why `serve.py` reads
`MLFLOW_TRACKING_URI` instead of hardcoding it — which is exactly what let this lab reuse the
Lab 3 inference image untouched.

## `docker-compose.yml`

```yaml
services:
  mlflow:
    build: ./mlflow
    ports:
      - "5000:5000"
    volumes:
      - mlflow-data:/mlflow-data

  inference:
    build: .
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
    depends_on:
      - mlflow

  frontend:
    build: ./frontend
    ports:
      - "8501:8501"
    environment:
      INFERENCE_URL: http://inference:8000
    depends_on:
      - inference

volumes:
  mlflow-data:
```

### Q5 — Only `mlflow` and `frontend` publish a port. Why not `inference`, and how does the frontend still reach it?

A port only needs publishing if something *on the host* has to reach it: the browser opens the
frontend on 8501, and the mlflow UI on 5000 is used to inspect runs and move aliases (and
`train.py`, which runs on the host, logs through it). The inference API is only ever called by
the frontend, and container-to-container traffic on the Compose network can reach any port a
container listens on — publishing is purely about the host → container direction. The frontend
calls `http://inference:8000` by service name, and uvicorn listens on `0.0.0.0:8000` inside the
inference container.

Verified both sides:

```
$ curl -s http://127.0.0.1:8000/health          # from the host
(connection refused — inference not reachable from host, as intended)

$ docker compose exec frontend python -c "import requests; print(requests.post(
    'http://inference:8000/predict', files={'file': open('/tmp/Seafood.jpg','rb')}).text)"
{"category":"Seafood","confidence":0.9816603660583496}
```

Not publishing it is also a small security win: the model API isn't exposed on the host's
network interfaces at all; the only way in is through the frontend.

### Q6 — `depends_on` only waits for the mlflow container to *start*. If mlflow isn't ready when `serve.py` loads the model, what happens to `inference`?

In normal `up` runs the race was never lost: inference spends several seconds importing
torch/mlflow before it even makes its first request, and by then the mlflow server is
listening. To force the failure, mlflow was stopped and inference restarted:

```bash
docker compose stop mlflow
docker compose restart inference
docker compose logs inference
```

```
inference-1  | INFO:     Waiting for application startup.
inference-1  | urllib3.exceptions.NameResolutionError: HTTPConnection(host='mlflow', port=5000): Failed to resolve 'mlflow' ([Errno -2] Name or service not known)
inference-1  | urllib3.exceptions.MaxRetryError: HTTPConnectionPool(host='mlflow', port=5000): Max retries exceeded with url: /api/2.0/mlflow/registered-models/alias?name=food11&alias=champion (...)
inference-1  | mlflow.exceptions.MlflowException: API request to http://mlflow:5000/api/2.0/mlflow/registered-models/alias failed with exception (...)
inference-1  | ERROR:    Application startup failed. Exiting.
```

```
frontend   Up 20 minutes
inference  Exited (3) 5 seconds ago
mlflow     Exited (0) 5 minutes ago
```

Two findings:

1. **It doesn't fail immediately.** The mlflow client retries with exponential backoff; the
   container sat at `Waiting for application startup.` (status `Up`) for **about four and a
   half minutes** before giving up. That retry window is what quietly covers the normal
   startup race — but it also means a genuinely dead mlflow looks "Up" for minutes.
2. **Then it exits for good.** The model load is in `serve.py`'s `lifespan`, so a failed load
   aborts uvicorn's startup (exit code 3). With no `restart:` policy the container stays
   `Exited` — while the frontend keeps running happily and every upload fails with
   `Failed to resolve 'inference'` (the stopped container drops off the network's DNS too).
   Starting mlflow again does *not* bring inference back on its own; it needs another
   `docker compose up -d`.

Ways to make this robust: a `healthcheck` on `mlflow` (e.g. polling `/health`) plus
`depends_on: mlflow: condition: service_healthy` on `inference`; `restart: on-failure` on
`inference` so it retries after mlflow comes back; and/or a `/health` endpoint on inference
that reports whether the model actually loaded, rather than the current unconditional
`{"status": "ok"}`.

## Bring the stack up

```bash
docker compose up --build
```

- mlflow UI: http://127.0.0.1:5000 — HTTP 200
- frontend: http://127.0.0.1:8501 — HTTP 200

End-to-end predictions along the frontend → inference → mlflow path, with version 1 as
`champion`, on the first validation image of three classes:

```
Bread/0_0.jpg   -> {"category":"Fried food","confidence":0.5478}
Seafood/8_0.jpg -> {"category":"Seafood","confidence":0.9817}
Soup/9_0.jpg    -> {"category":"Soup","confidence":0.9977}
```

### Q7 — Run `docker compose ps`. Which services have a published port listed?

```
NAME                      IMAGE                   SERVICE     STATUS          PORTS
mlops-lab-1-frontend-1    mlops-lab-1-frontend    frontend    Up 8 minutes    0.0.0.0:8501->8501/tcp, [::]:8501->8501/tcp
mlops-lab-1-inference-1   mlops-lab-1-inference   inference   Up 49 seconds   8000/tcp
mlops-lab-1-mlflow-1      mlops-lab-1-mlflow      mlflow      Up 8 minutes    0.0.0.0:5000->5000/tcp, [::]:5000->5000/tcp
```

| Service | PORTS | Published to host? |
|---|---|---|
| `frontend` | `0.0.0.0:8501->8501/tcp` | Yes |
| `mlflow` | `0.0.0.0:5000->5000/tcp` | Yes |
| `inference` | `8000/tcp` | No |

Matches the compose file exactly: only `mlflow` and `frontend` have a `ports:` section. The
`8000/tcp` shown for `inference` has no `host->` part — it comes from `EXPOSE 8000` in the Lab 3
Dockerfile, which only *documents* the port the container listens on; it doesn't map it to the
host.

## Promote a new model version and reload

Version 2 was trained with a 10× larger learning rate so its predictions would differ visibly
from version 1's — `honorable-ox-538` (`lr=0.001`, `batch_size=32`, `epochs=5`,
`test_accuracy=0.5648`) — registered as `food11` version 2, and `champion` moved to it:

```
Created version '2' of model 'food11'.
champion -> 2
```

Version 2 is clearly the worse model (0.5648 vs. 0.7473 test accuracy, in line with Lab 2's
sweep, where the same `lr=0.001, batch_size=32` config scored 0.5265) and wouldn't be promoted in practice; it's
used purely to make the switch observable.

### Q8 — Does the prediction come from the new version or the old one? What single command picks up the new version without rebuilding?

Same three images, before and after `docker compose restart inference`:

| Image | After alias move, **before** restart | **After** `docker compose restart inference` |
|---|---|---|
| `Bread/0_0.jpg` | Fried food (0.5478) | **Bread** (0.4897) |
| `Seafood/8_0.jpg` | Seafood (0.9817) | **Meat** (0.8745) |
| `Soup/9_0.jpg` | Soup (0.9977) | Soup (0.9774) |

Before the restart the predictions are byte-for-byte identical to version 1's — the **old**
model. `serve.py` resolves `models:/food11@champion` exactly once, in `lifespan`, and keeps the
loaded model in memory; moving the alias in the registry changes nothing for an
already-running process. The single command is:

```bash
docker compose restart inference
```

after which all three predictions change to version 2's (getting Bread right but Seafood wrong
— a net-worse model, as its test accuracy said).

### Q9 — Why does `restart` alone work — no rebuild? What does that say about what's baked into the image versus fetched at startup?

The inference image contains the Python environment (`.venv`) and `serve.py` — **no model
weights**. `restart` stops and starts the *same* container from the *same* image, which re-runs
`lifespan`: ask mlflow which version `champion` points to *now*, download that version's
artifacts over HTTP, load them. Nothing in the image needed to change, so there was nothing to
rebuild — the same conclusion as Lab 3 Q8, now demonstrated with an actual version change
rather than just a second container. Promoting a model is a *registry* operation (moving a
pointer), not a code deployment.

## Persistence check

### Q10 — Is the registered model and its alias still there after `down`/`up`? What's different after `down -v`?

**`docker compose down` + `docker compose up -d`:** everything survives.

```
 Container mlops-lab-1-frontend-1  Removed
 Container mlops-lab-1-inference-1  Removed
 Container mlops-lab-1-mlflow-1  Removed
 Network mlops-lab-1_default  Removed
$ docker volume ls        ->  mlops-lab-1_mlflow-data      (still there)

versions: ['1', '2']  champion -> 2
Bread -> {"category":"Bread","confidence":0.48971545696258545}    (version 2's answer)
```

`down` removes containers and the project network but leaves named volumes alone, so the new
mlflow container mounted the same `mlflow-data` volume — same SQLite database, same artifacts —
and the new inference container loaded version 2 straight away.

**`docker compose down -v` + `docker compose up -d`:** the registry is gone.

```
 Volume mlops-lab-1_mlflow-data  Removing
 Volume mlops-lab-1_mlflow-data  Removed

registered models: []  | experiments: ['Default']

inference-1  | mlflow.exceptions.RestException: RESOURCE_DOES_NOT_EXIST: Registered Model with name=food11 not found
inference-1  | ERROR:    Application startup failed. Exiting.
```

`-v` also deletes the named volumes declared in the compose file, so mlflow came up on a brand
new, empty volume — exactly the Q1 situation, just triggered deliberately. Inference then failed
the same way as on the very first `up` of this lab. Recovering means retraining (or restoring
a backup of the volume) and re-registering + re-aliasing a model. The volume, not the
container, is the only copy of the registry here — which is also why Q11's last point matters.

## Commit

```bash
git add mlflow/Dockerfile frontend/Dockerfile frontend/app.py docker-compose.yml .dockerignore lab4.md
git commit -m "Orchestrate mlflow, inference and frontend with Docker Compose"
git push
```

### Q11 — What would have to change for `inference` to run as three replicas behind a load balancer, or for mlflow to survive a machine failure? What can't Compose give you?

- **More than one machine.** Compose drives a single Docker host; if that host dies, every
  service dies with it. Spreading services across machines and rescheduling them onto a healthy
  one after a failure needs a cluster orchestrator (Kubernetes, or Docker Swarm).
- **Real load balancing with health-aware routing.** `docker compose up --scale inference=3`
  can start three copies on one host, and Compose's DNS will return all three IPs for
  `inference`, but that's plain DNS round-robin with no health checks — a replica that failed
  its model load (Q6) would still receive traffic. Kubernetes gives this as a Deployment with
  `replicas: 3` behind a Service, with readiness probes deciding who gets traffic.
- **Rolling updates and self-healing.** Replacing replicas one at a time with zero downtime on a
  new image or model version, and automatically replacing crashed ones, goes beyond Compose's
  basic `restart:` policies.
- **mlflow state that doesn't live on one machine's disk.** Right now the entire registry is a
  SQLite file plus artifacts on a local Docker volume (Q10 showed how easily that's lost).
  Surviving a machine failure needs the backend store in a replicated database (e.g. managed
  PostgreSQL) and artifacts in object storage (S3/GCS/Azure Blob, or MinIO) — at which point the
  mlflow server is stateless and can itself run as several replicas behind a load balancer.
