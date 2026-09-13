# MLOps Lab 1 — Git + DVC + Data Preparation

This is the first lab in a series that takes the Food-11 image classification project from
raw data through training, tracking, containerization, CI/CD, Kubernetes deployment, and
monitoring. This lab covers project setup, versioning code with Git, versioning data with DVC,
and preparing the Food-11 dataset for model training.

The full lab write-up — setup steps, the data remote strategy (including a DagsHub upload
issue and its workaround), and answers to all 8 lab questions — is in
[`lab/lab1.md`](lab/lab1.md).

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
dvc pull                         # fetch data/ from the configured DVC remote
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

See [`lab/lab1.md`](lab/lab1.md) for the data remote strategy and the answers to all 8 lab
questions.

---
*Prepared as part of an MLOps course lab series.*
