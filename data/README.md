# Dataset download guide

Raw benchmark files are **not** included in this repository. Download each dataset from the original providers and place the files under `data/` as described below.

Training and evaluation expect `--data_root data` from the repository root.

To **verify bundled metrics without downloading data**, run `python3 scripts/verify_results.py` from the repository root (see the main README).

---

## Quick start

```bash
mkdir -p data && cd data

# Example: MovieLens-1M (required for smoke test)
curl -L -o ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip && rm ml-1m.zip

cd ..
export PYTHONPATH="${PWD}/src"
python3 -m driftla.train --data_root data --dataset ml-1m --device cpu --smoke
```

Download only the datasets you need. Dense headline benchmarks require MovieLens-1M, Ciao, MovieLens-10M, and MovieLens-20M. Sparse / negative-result benchmarks require Gowalla, Yelp, and Amazon Reviews 2023.

---

## Expected directory layout

```
data/
├── ml-1m/          ratings.dat, users.dat, movies.dat
├── ml-10m/         ratings.dat  (or ratings.csv)
├── ml-20m/         ratings.csv
├── ciao/           rating.txt
├── gowala/         Gowalla_totalCheckins.txt
├── yelp/           yelp_academic_dataset_*.json
└── amazon23/
    └── raw/review_categories/<Category>.jsonl.gz
```

Generated caches (created automatically on first load; safe to delete):

- `yelp/_implicit_reviews_stars_ge_4p0.tsv`
- `amazon23/_implicit_<Category>_rating_ge_4p0.tsv`
- `_materialized_dense/` optional k-core exports from `scripts/data/materialize_dense_kcore.py`

---

## Download links

### Dense headline benchmarks

| Dataset | Folder | Used in paper as | Download |
|---------|--------|------------------|----------|
| MovieLens-1M | `data/ml-1m/` | ML-1M | [GroupLens page](https://grouplens.org/datasets/movielens/1m/) · [Direct ZIP](https://files.grouplens.org/datasets/movielens/ml-1m.zip) |
| Ciao | `data/ciao/` | Ciao | [UIC dataset page](https://www.cs.uic.edu/~liub/CA/research.html) · [Direct RAR](https://www.cs.uic.edu/~liub/CA/rating.rar) |
| MovieLens-10M | `data/ml-10m/` | ML-10M-cap | [GroupLens page](https://grouplens.org/datasets/movielens/10m/) · [Direct ZIP](https://files.grouplens.org/datasets/movielens/ml-10m.zip) |
| MovieLens-20M | `data/ml-20m/` | ML-20M-cap | [GroupLens page](https://grouplens.org/datasets/movielens/20m/) · [Direct ZIP](https://files.grouplens.org/datasets/movielens/ml-20m.zip) |

**Placement notes**

- **MovieLens-1M:** unzip so `data/ml-1m/ratings.dat` exists.
- **MovieLens-10M:** unzip so `data/ml-10m/ratings.dat` exists (or `ratings.csv`). Loaders apply a 300k interaction cap (`ml10m_cap300k`).
- **MovieLens-20M:** unzip so `data/ml-20m/ratings.csv` exists. Loaders apply a 300k interaction cap (`ml20m_cap300k`).
- **Ciao:** extract `rating.txt` into `data/ciao/rating.txt` (also accepts `ratings.txt` or `ratings.dat`).

### Sparse and negative-result benchmarks

| Dataset | Folder | Used in paper as | Download |
|---------|--------|------------------|----------|
| Gowalla | `data/gowala/` | Gowalla 5-core | [SNAP dataset page](https://snap.stanford.edu/data/loc-gowalla_totalCheckins.html) · [Direct TXT.GZ](https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz) |
| Yelp Academic | `data/yelp/` | Yelp 5-core | [Yelp Open Dataset](https://www.yelp.com/dataset) (free registration required) |
| Amazon Reviews 2023 | `data/amazon23/` | Amazon23-Magazine | [Project page](https://amazon-reviews-2023.github.io/) · [Data repo](https://github.com/McAuley-Lab/Amazon-Reviews-2023) |

**Placement notes**

- **Gowalla:** `gunzip -c loc-gowalla_totalCheckins.txt.gz > data/gowala/Gowalla_totalCheckins.txt`
- **Yelp:** after accepting the license, place the JSON bundle under `data/yelp/`. The loader reads `yelp_academic_dataset_review.json` and builds an implicit-feedback cache (stars >= 4.0).
- **Amazon23:** place category files under `data/amazon23/raw/review_categories/`. For the paper's Magazine negative result, download `Magazine_Subscriptions.jsonl.gz` and run with `--dataset amazon23 --amz23_category Magazine_Subscriptions`.

---

## Example commands

### MovieLens-10M and MovieLens-20M

```bash
cd data
curl -L -o ml-10m.zip https://files.grouplens.org/datasets/movielens/ml-10m.zip
unzip ml-10m.zip && rm ml-10m.zip

curl -L -o ml-20m.zip https://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip ml-20m.zip && rm ml-20m.zip
cd ..
```

### Ciao

```bash
mkdir -p data/ciao && cd data/ciao
curl -L -o rating.rar https://www.cs.uic.edu/~liub/CA/rating.rar
unrar x rating.rar    # or: bsdtar -xf rating.rar
mv rating.txt . 2>/dev/null || true
cd ../..
```

### Gowalla

```bash
mkdir -p data/gowala && cd data/gowala
curl -L -o checkins.txt.gz https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz
gunzip checkins.txt.gz
mv loc-gowalla_totalCheckins.txt Gowalla_totalCheckins.txt 2>/dev/null || mv checkins.txt Gowalla_totalCheckins.txt
cd ../..
```

### Yelp

1. Register at [https://www.yelp.com/dataset](https://www.yelp.com/dataset)
2. Download the academic dataset JSON bundle
3. Extract into `data/yelp/`

### Amazon Reviews 2023

Follow the instructions at [https://amazon-reviews-2023.github.io/data_processing/0download.py.html](https://amazon-reviews-2023.github.io/data_processing/0download.py.html) or clone [https://github.com/McAuley-Lab/Amazon-Reviews-2023](https://github.com/McAuley-Lab/Amazon-Reviews-2023) and place review JSONL files under `data/amazon23/raw/review_categories/`.

---

## Approximate download sizes (extracted)

| Dataset | Typical size |
|---------|--------------|
| ml-1m     | ~25 MB       |
| ml-10m    | ~250 MB      |
| ml-20m    | ~800 MB      |
| ciao      | ~1 GB        |
| gowala    | ~400 MB      |
| yelp      | ~9 GB        |
| amazon23  | varies by category |

---

## Verify installation

```bash
export PYTHONPATH="${PWD}/src"
python3 -m driftla.train --data_root data --dataset ml-1m --device cpu --smoke
```

A successful smoke test confirms MovieLens-1M is installed correctly. Repeat with other `--dataset` values after downloading the corresponding files.

---

## Licensing

Each dataset remains subject to its provider's terms of use.
