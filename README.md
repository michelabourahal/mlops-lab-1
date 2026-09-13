# MLOps Lab 1 — Git + DVC + Data Preparation

This is the first lab in a series that takes the Food-11 image classification project from
raw data through training, tracking, containerization, CI/CD, Kubernetes deployment, and
monitoring. This lab covers project setup, versioning code with Git, versioning data with
DVC (remote: [DagsHub](https://dagshub.com/michelabourahal/mlops-lab-1)), and preparing the
Food-11 dataset for model training.

## Project structure

```
.
├── data/                       # DVC-tracked, not in git (see data.dvc)
│   ├── food11_raw/              training/evaluation/validation, as downloaded from Kaggle
│   ├── food11_processed/        resized to 128x128, sorted into per-category folders
│   └── food11_processed_mini/   same as above, capped at 100 images/category, for dev
├── src/food11/
│   ├── __init__.py
│   └── data.py                 # builds food11_processed and food11_processed_mini from food11_raw
├── data.dvc                     # DVC pointer to the data/ folder (committed to git)
├── .dvc/config                  # non-secret DVC project config (default remote: origin/DagsHub)
├── .dvcignore
├── pyproject.toml
└── uv.lock
```

## Setup

```bash
uv sync                          # install dependencies into .venv
dvc pull                         # fetch data/ from the DagsHub remote
uv run python ./src/food11/data.py   # (re)build food11_processed and food11_processed_mini
```

## The dataset

Food-11 (Kaggle) contains 512x512 images across 11 categories, pre-split into
`training`, `evaluation`, and `validation` folders. The raw filenames encode the category as
a **numeric prefix** (`0_123.jpg`, `1_45.jpg`, …), which `src/food11/data.py` maps to the
actual category name before writing the processed datasets:

| Index | Category |
|---|---|
| 0 | Bread |
| 1 | Dairy product |
| 2 | Dessert |
| 3 | Egg |
| 4 | Fried food |
| 5 | Meat |
| 6 | Noodles-Pasta |
| 7 | Rice |
| 8 | Seafood |
| 9 | Soup |
| 10 | Vegetable-Fruit |

`food11_processed` mirrors `food11_raw`'s split structure but resizes every image to 128x128
and sorts it into `<split>/<category>/` folders. `food11_processed_mini` applies the same
transform but keeps at most 100 images per category per split, so the pipeline can be
developed and debugged quickly before running against the full dataset.

## Lab questions and answers

### Q1 — `uv init` output: what do the generated files contain?

`uv init` scaffolds a minimal, installable Python project:

- **`pyproject.toml`** — project metadata (name, version, Python version constraint) and
  the dependency list `uv add`/`uv sync` manage.
- **`.python-version`** — pins the interpreter version (`3.11` here) so `uv` always
  provisions the same Python regardless of what's on the host machine.
- **`README.md`** — placeholder project readme (this file).
- **`src/<package_name>/__init__.py`** — a starter package matching the project name, with a
  `main()` entry point wired into `[project.scripts]`. We removed this boilerplate package
  since the project is a set of data-processing scripts under `src/food11/`, not an installable
  library — `pyproject.toml` now sets `[tool.uv] package = false` accordingly.
- **`uv.lock`** (created on first `uv sync`/`uv run`) — the fully resolved, pinned dependency
  graph. It's what makes installs reproducible across machines and belongs in git, alongside
  `.python-version` and `pyproject.toml`.

### Q2 — What files does `dvc init` create, what are they for, and which go to git?

`dvc init` creates:

- **`.dvc/config`** — the shared, non-secret project configuration (which remote is default,
  remote URLs, etc). **Committed to git** — this is what lets a teammate `git clone` and
  immediately know where to `dvc pull`/`dvc push`.
- **`.dvc/.gitignore`** — tells git to ignore the three paths below. **Committed to git.**
- **`.dvc/cache/`** — the local content-addressable object store (the actual file contents,
  keyed by hash). **Never committed** — this is exactly what DVC pushes to/pulls from the
  remote instead of git; it's also potentially huge and machine-specific.
- **`.dvc/tmp/`** — runtime state (locks, update checks). **Never committed.**
- **`.dvc/config.local`** (created later, when local/secret overrides exist) — machine-specific
  config, e.g. credentials. **Never committed.**
- **`.dvcignore`** (repo root) — a `.gitignore`-style file listing paths DVC itself should
  skip when scanning directories. **Committed to git.**

### Q3 — Where are DVC remote credentials stored? What are the alternatives to `--global`, and should credentials go to GitHub?

Because we ran `dvc remote modify origin --global ...`, the DagsHub username/password were
written to DVC's **global** config file, outside any git repository, at
`%LOCALAPPDATA%\iterative\dvc\config` on Windows (`~/.config/dvc/config` on Linux/macOS). This
file is never part of a git repo, so there's no risk of it being pushed.

`dvc remote modify` supports four scopes:

- **`--global`** — used here; writes to the per-user config described above.
- **`--system`** — a single config shared by every user on the machine.
- **`--project`** (the default when no flag is given) — writes to the shared, git-committed
  `.dvc/config`. Fine for the remote's name/URL, never for secrets.
- **`--local`** — writes to the project's own git-ignored `.dvc/config.local`; scoped to one
  clone, still never committed.

Credentials must **never** be pushed to GitHub. That's exactly why DVC splits config into a
committed, secret-free `.dvc/config` (which only holds the remote's name/URL) and an
uncommitted local/global file for the username and password.

### Q4 — What happened to `.gitignore` after `dvc add data`?

`dvc add data` moved `data/`'s contents into DVC's cache (hashed by content) and replaced it
in git's view with a single pointer file, `data.dvc`. To stop git from trying to track the
(now DVC-managed) `data/` folder itself, DVC appended `/data` to `.gitignore`. Git now only
ever sees `data.dvc`; the actual files are entirely DVC's responsibility.

### Q5 — What's in the `.dvc` file DVC creates?

`data.dvc` is a small YAML pointer, e.g.:

```yaml
outs:
- md5: 94afad23dc86ad238cbb965ec28aecd4.dir
  size: 1274882585
  nfiles: 36578
  hash: md5
  path: data
```

`md5` is the hash of the directory's manifest (which in turn lists the hash of every file
inside), `size` and `nfiles` are metadata, and `path` is what this pointer resolves to on
checkout. This is the file git actually versions — every commit that changes the dataset
changes this one small pointer, and `dvc checkout`/`dvc pull` use its hash to fetch the
matching content from cache or the remote.

### Q6 — On GitHub: is the code there? The data? A pointer to it? And on DagsHub?

- **GitHub** has the code (`src/`), the DVC pointer (`data.dvc`), and DVC's own non-secret
  config (`.dvc/config`, `.dvcignore`) — but **not** the actual image files; `data/` is
  git-ignored.
- **DagsHub** (the DVC remote) holds the actual dataset content, addressed by hash, pushed
  there via `dvc push`. Its file browser also renders `data/` directly (DagsHub understands
  `.dvc` pointers and resolves them to the underlying DVC remote for preview), even though
  that data was never committed to git.

  We found the default DVC remote had drifted to a local Windows path
  (`.dvc/config` had `core.remote = localstore` → `C:\dvc-storage`) after the preprocessing
  commit, meaning the processed datasets were pushed to disk instead of DagsHub. This has been
  corrected: `.dvc/config` now sets `core.remote = origin` (DagsHub) again, and the corrected
  data has been re-pushed.

### Q7 — Cloning fresh: do you see the data folder? What command fetches it?

A fresh `git clone` gives you the code and `data.dvc`, but **no `data/` folder** — it's
git-ignored and was never committed. Running

```bash
dvc pull
```

reads `data.dvc`, matches its hash against the configured remote (DagsHub), downloads the
content, and materializes `data/` in the working directory.

### Q8 — Checking out an older commit: are `food11_processed`/`food11_processed_mini` still there?

No. `git log --oneline -- data.dvc` shows which commits changed the dataset pointer; checking
out a commit from before the preprocessing step (and running `dvc checkout`) restores
`data.dvc` — and therefore `data/` — to the state it pointed to *at that commit*, which only
contains `food11_raw`. The processed folders only exist in commits made after
`src/food11/data.py` was run and `dvc add data` picked up its output. `git checkout main &&
dvc checkout` brings both folders back.

## Known issues fixed during review

- DVC's default remote had silently drifted from DagsHub (`origin`) to a local path
  (`C:\dvc-storage`) in the preprocessing commit — reverted so `dvc push`/`dvc pull` go to
  DagsHub again.
- `food11_processed`/`food11_processed_mini` were organized into folders named `0`–`10`
  (the raw numeric class index) instead of the actual category names — fixed and the datasets
  regenerated.
- Compiled `__pycache__/*.pyc` files were committed to git — removed and ignored going forward.
- `uv.lock` and `.python-version` were never committed — added, since they're what make the
  environment reproducible.
- An unused `uv init` boilerplate package (`src/mlops_lab_1/`) was removed; `pyproject.toml`
  now sets `[tool.uv] package = false` since this project is scripts, not an installable
  library.

---
*Prepared as part of an MLOps course lab series.*
