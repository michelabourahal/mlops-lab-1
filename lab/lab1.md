# Lab 1 — Git + DVC + Data Preparation

Repo: https://github.com/michelabourahal/mlops-lab-1

## Data remote strategy

We initially set up DagsHub (`origin`, `https://dagshub.com/michelabourahal/mlops-lab-1.dvc`)
as the DVC remote, per the base instructions, and successfully pushed the raw dataset
(`food11_raw`, ~9.6k files) to it early on.

After reprocessing the dataset (adding `food11_processed` and `food11_processed_mini`, ~36.5k
files / ~1.27GB total), `dvc push` to DagsHub reliably stalled with no progress for 13+
minutes and no error — the same issue flagged in the course announcement.

**Adopted solution: Solution 1 — local remote outside the repo.** The DVC default remote
(`.dvc/config` → `core.remote`) now points to `local_backup`, a folder at
`C:\Users\miche\dvc-storage\mlops-lab-1`, entirely outside the git working tree. All 32,034
files pushed there in under 2 minutes, confirming the DagsHub upload — not the data or the
pipeline — was the bottleneck.

- `origin` (DagsHub) is still defined in the global DVC config and still holds the raw dataset
  from the earlier successful push; it's just no longer the default remote for `dvc push`/`dvc
  pull` since it can't reliably take the full processed dataset.
- `dvc push`/`dvc pull` in this repo now operate against `local_backup`. Anyone reproducing
  this exactly would need read/write access to that path (or to re-point the remote at their
  own local/shared storage) — this is the tradeoff of Solution 1, made explicit here rather
  than left implicit.

## Lab questions and answers

### Q1 — `uv init` output: what do the generated files contain?

`uv init` scaffolds a minimal, installable Python project:

- **`pyproject.toml`** — project metadata (name, version, Python version constraint) and the
  dependency list `uv add`/`uv sync` manage.
- **`.python-version`** — pins the interpreter version (`3.11` here) so `uv` always provisions
  the same Python regardless of what's on the host machine.
- **`README.md`** — placeholder project readme.
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
- **`.dvcignore`** (repo root) — a `.gitignore`-style file listing paths DVC itself should skip
  when scanning directories. **Committed to git.**

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
uncommitted local/global file for the username and password. Our local-remote workaround
(`local_backup`) needs no credentials at all — it's just a filesystem path — which is one
reason it's simple to fall back to.

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
- **DagsHub** holds only the earlier raw-data push (`food11_raw`); it was not able to take the
  full processed dataset (see "Data remote strategy" above), so it does not have
  `food11_processed`/`food11_processed_mini`.
- **The local remote** (`local_backup`, outside the repo) holds the complete, current dataset —
  raw and both processed variants — and is what `data.dvc`'s hash currently resolves against.

### Q7 — Cloning fresh: do you see the data folder? What command fetches it? *(verified empirically)*

A fresh `git clone` gives you the code and `data.dvc`, but no `data/` folder — it's git-ignored
and was never committed. `dvc pull` reads `data.dvc`, matches its hash against the configured
remote, downloads the content, and materializes `data/` in the working directory.

We verified this directly: cloned the repo into a scratch temp folder, confirmed `data/` was
absent, ran `dvc pull` (pointed at `local_backup`), and confirmed all three dataset folders
appeared with the expected file counts. See "Empirical verification" below for the exact
output.

### Q8 — Checking out an older commit: are `food11_processed`/`food11_processed_mini` still there? *(verified empirically)*

No. `git log --oneline -- data.dvc` shows which commits changed the dataset pointer; checking
out a commit from before the preprocessing step (and running `dvc checkout`) restores
`data.dvc` — and therefore `data/` — to the state it pointed to *at that commit*, which only
contains `food11_raw`. The processed folders only exist in commits made after
`src/food11/data.py` was run and `dvc add data` picked up its output. `git checkout main && dvc
checkout` brings both folders back.

We verified this directly by checking out the pre-preprocessing commit, confirming
`food11_processed`/`food11_processed_mini` disappeared while `food11_raw` remained, then
switching back to `main` and confirming both folders returned. See "Empirical verification"
below.

## Empirical verification

Both checks below were run for real, not just reasoned about.

### Q7 check: fresh clone + `dvc pull`

```
$ git clone https://github.com/michelabourahal/mlops-lab-1.git
$ cd mlops-lab-1
$ ls data
ls: cannot access 'data': No such file or directory

$ dvc pull
A       data\
32034 files fetched and 36578 files added
```

After the pull, `data/` contained all three expected folders with the right structure and
counts:

| Folder | Files | Notes |
|---|---|---|
| `food11_raw` (training+evaluation+validation) | 16,643 | matches the Kaggle source |
| `food11_processed` | 16,643 | folders named `Bread`, `Dairy product`, …, `Vegetable-Fruit` |
| `food11_processed_mini` | 3,292 | same category folders, capped at 100/class |

(Note: this first attempt failed with `No space left on device` because the machine's `C:`
drive was, independently of this lab, completely full at the time. It was re-run successfully
once disk space was freed — confirming the failure was a host disk-space issue, not a DVC or
pipeline problem.)

### Q8 check: checkout an older commit, then back to `main`

```
$ git log --oneline -- data.dvc
0c4f798 Fix category mapping, restore DagsHub remote, and clean up repo
883d96d Add Food-11 preprocessing and DVC data
c5e4a73 Track food11 dataset with DVC
40f242f Track food11 dataset with DVC

$ git checkout c5e4a73
$ dvc checkout
M       data\
$ ls data
food11_raw
$ ls data/food11_processed
ls: cannot access 'data/food11_processed': No such file or directory
```

At `c5e4a73` — before the preprocessing script had been run — only `food11_raw` (16,643 files)
exists; `food11_processed` and `food11_processed_mini` are both absent, exactly as expected.

```
$ git checkout main
$ dvc checkout
M       data\
$ ls data
food11_processed  food11_processed_mini  food11_raw
```

Back on `main`, all three folders returned with their full counts (16,643 / 16,643 / 3,292),
and `git status` reported a clean working tree — confirming `data.dvc` + `dvc checkout` fully
round-trips the dataset state across commits.
