
from __future__ import annotations

import json
import os
import gzip
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


def _load_movielens_ratings_any(
    ratings_path: str,
    *,
    min_rating: float = 4.0,
) -> pd.DataFrame:
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"Missing {ratings_path}")

    base = os.path.basename(ratings_path)
    if base.endswith(".dat"):
        df = pd.read_csv(
            ratings_path,
            sep="::",
            header=None,
            names=["user_id", "item_id", "rating", "timestamp"],
            engine="python",
        )
    else:
        df = pd.read_csv(ratings_path)
        rename = {}

        if "userId" in df.columns:
            rename["userId"] = "user_id"
        if "movieId" in df.columns:
            rename["movieId"] = "item_id"
        if "rating" in df.columns:
            rename["rating"] = "rating"
        if "timestamp" in df.columns:
            rename["timestamp"] = "timestamp"
        df = df.rename(columns=rename)
        missing = {"user_id", "item_id", "rating", "timestamp"} - set(df.columns)
        if missing:
            raise ValueError(
                f"Unsupported ratings csv schema in {ratings_path}; missing {sorted(missing)}"
            )
        df = df[["user_id", "item_id", "rating", "timestamp"]]

    df = df[df["rating"] >= float(min_rating)]
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _movielens_to_stream(
    df: pd.DataFrame,
    init_ratio: float,
    n_batches: int,
) -> Tuple[
    List[Tuple[int, int]],
    List[List[Tuple[int, int]]],
    Dict[Tuple[int, int], float],
    int,
    int,
]:
    u_map = {uid: idx for idx, uid in enumerate(sorted(df["user_id"].unique()))}
    i_map = {iid: idx for idx, iid in enumerate(sorted(df["item_id"].unique()))}
    df = df.copy()
    df["user_id"] = df["user_id"].map(u_map)
    df["item_id"] = df["item_id"].map(i_map)
    n_users, n_items = len(u_map), len(i_map)

    interactions = list(zip(df["user_id"].values, df["item_id"].values))
    timestamps = {
        (u, i): float(t)
        for u, i, t in zip(df["user_id"], df["item_id"], df["timestamp"])
    }

    split = int(len(interactions) * float(init_ratio))
    init_data = interactions[:split]
    stream_data = interactions[split:]
    batch_size = len(stream_data) // int(n_batches)
    batches: List[List[Tuple[int, int]]] = []
    for b in range(int(n_batches)):
        lo = b * batch_size
        hi = lo + batch_size if b < int(n_batches) - 1 else len(stream_data)
        batches.append(stream_data[lo:hi])
    return init_data, batches, timestamps, n_users, n_items


def _parse_ciao_date(date_str: str) -> float:
    try:
        dt = datetime.strptime(date_str.strip(), "%d.%m.%Y")
        return float(dt.timestamp())
    except Exception:
        return 0.0


def load_ml1m_chronological(
    data_root: str,
    init_ratio: float = 0.5,
    n_batches: int = 10,
) -> Tuple[
    List[Tuple[int, int]],
    List[List[Tuple[int, int]]],
    Dict[Tuple[int, int], float],
    int,
    int,
]:
    ratings_file = os.path.join(data_root, "ml-1m", "ratings.dat")
    if not os.path.exists(ratings_file):
        raise FileNotFoundError(
            f"Missing {ratings_file}. Place MovieLens-1M files under {data_root}/ml-1m/."
        )

    df = _load_movielens_ratings_any(ratings_file, min_rating=4.0)
    return _movielens_to_stream(df, init_ratio=init_ratio, n_batches=n_batches)


def load_ml10m_chronological(
    data_root: str,
    init_ratio: float = 0.5,
    n_batches: int = 10,
    max_interactions: Optional[int] = None,
) -> Tuple[
    List[Tuple[int, int]],
    List[List[Tuple[int, int]]],
    Dict[Tuple[int, int], float],
    int,
    int,
]:
    p_dat = os.path.join(data_root, "ml-10m", "ratings.dat")
    p_csv = os.path.join(data_root, "ml-10m", "ratings.csv")
    path = p_dat if os.path.exists(p_dat) else p_csv
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {p_dat} (or {p_csv}). Place MovieLens-10M under {data_root}/ml-10m/."
        )
    df = _load_movielens_ratings_any(path, min_rating=4.0)
    if max_interactions is not None and max_interactions > 0 and len(df) > max_interactions:
        df = df.iloc[: int(max_interactions)].copy()
    return _movielens_to_stream(df, init_ratio=init_ratio, n_batches=n_batches)


def load_ml20m_chronological(
    data_root: str,
    init_ratio: float = 0.5,
    n_batches: int = 10,
    max_interactions: Optional[int] = None,
) -> Tuple[
    List[Tuple[int, int]],
    List[List[Tuple[int, int]]],
    Dict[Tuple[int, int], float],
    int,
    int,
]:
    ratings_csv = os.path.join(data_root, "ml-20m", "ratings.csv")
    if not os.path.exists(ratings_csv):
        raise FileNotFoundError(
            f"Missing {ratings_csv}. Place MovieLens-20M under {data_root}/ml-20m/."
        )
    df = _load_movielens_ratings_any(ratings_csv, min_rating=4.0)
    if max_interactions is not None and max_interactions > 0 and len(df) > max_interactions:
        df = df.iloc[: int(max_interactions)].copy()
    return _movielens_to_stream(df, init_ratio=init_ratio, n_batches=n_batches)


def load_ciao_chronological(
    data_root: str,
    init_ratio: float = 0.5,
    n_batches: int = 10,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
) -> Tuple[
    List[Tuple[int, int]],
    List[List[Tuple[int, int]]],
    Dict[Tuple[int, int], float],
    int,
    int,
]:
    ratings_file = os.path.join(data_root, "ciao", "rating.txt")
    if not os.path.exists(ratings_file):
        raise FileNotFoundError(
            f"Missing {ratings_file}. Place Ciao files under {data_root}/ciao/."
        )

    raw = []
    with open(ratings_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split("::::")
            if len(parts) < 6:
                continue
            user_id = parts[0].strip()
            product = parts[1].strip()
            date_str = parts[5].strip()
            raw.append((user_id, product, _parse_ciao_date(date_str)))

    df = pd.DataFrame(raw, columns=["user", "item", "ts"])
    prev_len = 0
    while len(df) != prev_len:
        prev_len = len(df)
        u_counts = df["user"].value_counts()
        valid_u = u_counts[u_counts >= min_user_interactions].index
        df = df[df["user"].isin(valid_u)]
        i_counts = df["item"].value_counts()
        valid_i = i_counts[i_counts >= min_item_interactions].index
        df = df[df["item"].isin(valid_i)]

    df = df.sort_values("ts").reset_index(drop=True)

    u_map = {uid: idx for idx, uid in enumerate(sorted(df["user"].unique()))}
    i_map = {iid: idx for idx, iid in enumerate(sorted(df["item"].unique()))}
    df["user_id"] = df["user"].map(u_map)
    df["item_id"] = df["item"].map(i_map)
    n_users, n_items = len(u_map), len(i_map)

    interactions = list(zip(df["user_id"].values, df["item_id"].values))
    timestamps = {
        (u, i): float(t)
        for u, i, t in zip(df["user_id"], df["item_id"], df["ts"])
    }

    split = int(len(interactions) * init_ratio)
    init_data = interactions[:split]
    stream_data = interactions[split:]
    chunk = len(stream_data) // n_batches
    batches: List[List[Tuple[int, int]]] = []
    for b in range(n_batches):
        lo = b * chunk
        hi = lo + chunk if b < n_batches - 1 else len(stream_data)
        batches.append(stream_data[lo:hi])

    return init_data, batches, timestamps, n_users, n_items


def load_gowala_chronological(
    data_root: str,
    init_ratio: float = 0.5,
    n_batches: int = 10,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    max_interactions: int = 300_000,
) -> Tuple[
    List[Tuple[int, int]],
    List[List[Tuple[int, int]]],
    Dict[Tuple[int, int], float],
    int,
    int,
]:
    checkins = os.path.join(data_root, "gowala", "Gowalla_totalCheckins.txt")
    if not os.path.exists(checkins):
        raise FileNotFoundError(
            f"Missing {checkins}. Place Gowalla files under {data_root}/gowala/."
        )

    df = pd.read_csv(
        checkins,
        sep="\t",
        header=None,
        names=["user", "ts", "lat", "lon", "item"],
        usecols=[0, 1, 4],
    )
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])

    if min_user_interactions > 1 or min_item_interactions > 1:
        prev_len = 0
        while len(df) != prev_len:
            prev_len = len(df)
            u_counts = df["user"].value_counts()
            valid_u = u_counts[u_counts >= min_user_interactions].index
            df = df[df["user"].isin(valid_u)]
            i_counts = df["item"].value_counts()
            valid_i = i_counts[i_counts >= min_item_interactions].index
            df = df[df["item"].isin(valid_i)]

    df = df.sort_values("ts").reset_index(drop=True)
    if max_interactions is not None and max_interactions > 0 and len(df) > max_interactions:

        df = df.iloc[:max_interactions].copy()

    u_map = {uid: idx for idx, uid in enumerate(sorted(df["user"].unique()))}
    i_map = {iid: idx for idx, iid in enumerate(sorted(df["item"].unique()))}
    df["user_id"] = df["user"].map(u_map)
    df["item_id"] = df["item"].map(i_map)
    n_users, n_items = len(u_map), len(i_map)

    interactions = list(zip(df["user_id"].values, df["item_id"].values))
    ts_int = (df["ts"].astype("int64") // 10**9).astype(np.int64)
    timestamps = {
        (u, i): float(t)
        for u, i, t in zip(df["user_id"], df["item_id"], ts_int)
    }

    split = int(len(interactions) * init_ratio)
    init_data = interactions[:split]
    stream_data = interactions[split:]
    chunk = len(stream_data) // n_batches
    batches: List[List[Tuple[int, int]]] = []
    for b in range(n_batches):
        lo = b * chunk
        hi = lo + chunk if b < n_batches - 1 else len(stream_data)
        batches.append(stream_data[lo:hi])

    return init_data, batches, timestamps, n_users, n_items


def _parse_yelp_date(date_str: str) -> float:
    try:
        dt = datetime.strptime(date_str.strip()[:19], "%Y-%m-%d %H:%M:%S")
        return float(dt.timestamp())
    except Exception:
        return 0.0


def _yelp_cached_implicit_path(data_root: str, min_stars: float) -> str:

    ms = str(min_stars).replace(".", "p")
    return os.path.join(data_root, "yelp", f"_implicit_reviews_stars_ge_{ms}.tsv")


def _ensure_yelp_cached_implicit(
    data_root: str,
    min_stars: float,
) -> str:
    cache_path = _yelp_cached_implicit_path(data_root, min_stars)
    if os.path.exists(cache_path):
        return cache_path

    reviews = os.path.join(data_root, "yelp", "yelp_academic_dataset_review.json")
    if not os.path.exists(reviews):
        raise FileNotFoundError(
            f"Missing {reviews}. Place Yelp Academic JSON under {data_root}/yelp/."
        )

    tmp_path = cache_path + ".tmp"
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    n_written = 0
    with open(reviews, "r", encoding="utf-8", errors="ignore") as fin, open(
        tmp_path, "w", encoding="utf-8"
    ) as fout:
        fout.write("user\titem\tts\n")
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            stars = o.get("stars")
            if stars is None or float(stars) < min_stars:
                continue
            uid = o.get("user_id")
            bid = o.get("business_id")
            ds = o.get("date")
            if not uid or not bid or not ds:
                continue
            ts = _parse_yelp_date(str(ds))
            fout.write(f"{uid}\t{bid}\t{ts}\n")
            n_written += 1

    os.replace(tmp_path, cache_path)
    print(f"[yelp] wrote implicit cache: {cache_path} ({n_written} interactions)")
    return cache_path


def _amz23_cached_implicit_path(data_root: str, category: str, min_rating: float) -> str:
    mr = str(min_rating).replace(".", "p")
    safe_cat = category.replace("/", "_")
    return os.path.join(data_root, "amazon23", f"_implicit_{safe_cat}_rating_ge_{mr}.tsv")


def _ensure_amz23_cached_implicit(
    data_root: str,
    category: str,
    min_rating: float,
) -> str:
    cache_path = _amz23_cached_implicit_path(data_root, category, min_rating)
    if os.path.exists(cache_path):
        return cache_path

    src = os.path.join(
        data_root,
        "amazon23",
        "raw",
        "review_categories",
        f"{category}.jsonl.gz",
    )
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"Missing {src}. Download from https://amazon-reviews-2023.github.io/ "
            f"and place it under data/amazon23/raw/review_categories/."
        )

    tmp_path = cache_path + ".tmp"
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    n_written = 0
    with gzip.open(src, "rt", encoding="utf-8", errors="ignore") as fin, open(
        tmp_path, "w", encoding="utf-8"
    ) as fout:
        fout.write("user\titem\tts\n")
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            r = o.get("rating")
            if r is None or float(r) < float(min_rating):
                continue
            uid = o.get("user_id")

            iid = o.get("parent_asin") or o.get("asin")
            ts = o.get("sort_timestamp") or o.get("timestamp")
            if not uid or not iid or ts is None:
                continue
            try:
                ts_f = float(ts)
            except Exception:
                continue

            if ts_f > 1e12:
                ts_f = ts_f / 1000.0
            fout.write(f"{uid}\t{iid}\t{ts_f}\n")
            n_written += 1

    os.replace(tmp_path, cache_path)
    print(f"[amazon23] wrote implicit cache: {cache_path} ({n_written} interactions)")
    return cache_path


def load_amazon23_category_chronological(
    data_root: str,
    category: str,
    init_ratio: float = 0.5,
    n_batches: int = 10,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    max_interactions: Optional[int] = 300_000,
    min_rating: float = 4.0,
) -> Tuple[
    List[Tuple[int, int]],
    List[List[Tuple[int, int]]],
    Dict[Tuple[int, int], float],
    int,
    int,
]:
    cache_path = _ensure_amz23_cached_implicit(data_root, category, min_rating=min_rating)
    df = pd.read_csv(cache_path, sep="\t")
    if df.empty:
        raise ValueError(f"No Amazon'23 interactions passed filter for category={category}.")

    if min_user_interactions > 1 or min_item_interactions > 1:
        prev_len = 0
        while len(df) != prev_len:
            prev_len = len(df)
            u_counts = df["user"].value_counts()
            valid_u = u_counts[u_counts >= min_user_interactions].index
            df = df[df["user"].isin(valid_u)]
            i_counts = df["item"].value_counts()
            valid_i = i_counts[i_counts >= min_item_interactions].index
            df = df[df["item"].isin(valid_i)]

    df = df.sort_values("ts").reset_index(drop=True)
    if max_interactions is not None and max_interactions > 0 and len(df) > max_interactions:
        df = df.iloc[: int(max_interactions)].copy()

    u_map = {uid: idx for idx, uid in enumerate(sorted(df["user"].unique()))}
    i_map = {iid: idx for idx, iid in enumerate(sorted(df["item"].unique()))}
    df["user_id"] = df["user"].map(u_map)
    df["item_id"] = df["item"].map(i_map)
    n_users, n_items = len(u_map), len(i_map)

    interactions = list(zip(df["user_id"].values, df["item_id"].values))
    timestamps = {
        (u, i): float(t)
        for u, i, t in zip(df["user_id"], df["item_id"], df["ts"])
    }

    split = int(len(interactions) * float(init_ratio))
    init_data = interactions[:split]
    stream_data = interactions[split:]
    chunk = len(stream_data) // int(n_batches)
    batches: List[List[Tuple[int, int]]] = []
    for b in range(int(n_batches)):
        lo = b * chunk
        hi = lo + chunk if b < int(n_batches) - 1 else len(stream_data)
        batches.append(stream_data[lo:hi])
    return init_data, batches, timestamps, n_users, n_items


def load_yelp_chronological(
    data_root: str,
    init_ratio: float = 0.5,
    n_batches: int = 10,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    max_interactions: int = 300_000,
    min_stars: float = 4.0,
) -> Tuple[
    List[Tuple[int, int]],
    List[List[Tuple[int, int]]],
    Dict[Tuple[int, int], float],
    int,
    int,
]:
    cache_path = _ensure_yelp_cached_implicit(data_root, min_stars=min_stars)
    df = pd.read_csv(cache_path, sep="\t")
    if df.empty:
        raise ValueError("No Yelp reviews passed the star filter; check the file path and format.")

    if min_user_interactions > 1 or min_item_interactions > 1:
        prev_len = 0
        while len(df) != prev_len:
            prev_len = len(df)
            u_counts = df["user"].value_counts()
            valid_u = u_counts[u_counts >= min_user_interactions].index
            df = df[df["user"].isin(valid_u)]
            i_counts = df["item"].value_counts()
            valid_i = i_counts[i_counts >= min_item_interactions].index
            df = df[df["item"].isin(valid_i)]

    df = df.sort_values("ts").reset_index(drop=True)
    if max_interactions is not None and max_interactions > 0 and len(df) > max_interactions:
        df = df.iloc[:max_interactions].copy()

    u_map = {uid: idx for idx, uid in enumerate(sorted(df["user"].unique()))}
    i_map = {iid: idx for idx, iid in enumerate(sorted(df["item"].unique()))}
    df["user_id"] = df["user"].map(u_map)
    df["item_id"] = df["item"].map(i_map)
    n_users, n_items = len(u_map), len(i_map)

    interactions = list(zip(df["user_id"].values, df["item_id"].values))
    timestamps = {
        (u, i): float(t)
        for u, i, t in zip(df["user_id"], df["item_id"], df["ts"])
    }

    split = int(len(interactions) * init_ratio)
    init_data = interactions[:split]
    stream_data = interactions[split:]
    chunk = len(stream_data) // n_batches
    batches: List[List[Tuple[int, int]]] = []
    for b in range(n_batches):
        lo = b * chunk
        hi = lo + chunk if b < n_batches - 1 else len(stream_data)
        batches.append(stream_data[lo:hi])

    return init_data, batches, timestamps, n_users, n_items


def create_negative_samples(
    users: np.ndarray,
    user_pos_items: Dict[int, set],
    n_items: int,
    num_negatives: int = 1,
) -> np.ndarray:
    negatives = []
    for u in users:
        pos = user_pos_items.get(int(u), set())
        negs = []
        while len(negs) < num_negatives:
            j = np.random.randint(0, n_items)
            if j not in pos:
                negs.append(j)
        negatives.extend(negs)
    return np.array(negatives, dtype=np.int64)
