
import json
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict


def load_ml1m(data_dir):
    ratings_file = os.path.join(data_dir, "ratings.dat")
    if not os.path.exists(ratings_file):
        raise FileNotFoundError(
            f"{ratings_file} not found.  Download ML-1M from "
            "https://grouplens.org/datasets/movielens/1m/ and place ratings.dat here."
        )

    df = pd.read_csv(
        ratings_file, sep="::", header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    df = df[df["rating"] >= 4]

    u_map = {uid: idx for idx, uid in enumerate(sorted(df["user_id"].unique()))}
    i_map = {iid: idx for idx, iid in enumerate(sorted(df["item_id"].unique()))}
    df["user_id"] = df["user_id"].map(u_map)
    df["item_id"] = df["item_id"].map(i_map)

    interactions = list(zip(df["user_id"], df["item_id"]))
    timestamps = {
        (u, i): float(t)
        for u, i, t in zip(df["user_id"], df["item_id"], df["timestamp"])
    }
    return interactions, timestamps, len(u_map), len(i_map)


def _parse_ciao_date(date_str):
    try:
        dt = datetime.strptime(date_str.strip(), "%d.%m.%Y")
        return dt.timestamp()
    except Exception:
        return 0.0


def load_ciao(data_dir, min_user_interactions=5, min_item_interactions=5):
    ratings_file = os.path.join(data_dir, "rating.txt")
    if not os.path.exists(ratings_file):
        raise FileNotFoundError(f"{ratings_file} not found.")

    print(f"  Parsing {ratings_file} ...")
    raw = []
    with open(ratings_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split("::::")
            if len(parts) < 6:
                continue
            user_id = parts[0].strip()
            product = parts[1].strip()
            date_str = parts[5].strip()
            ts = _parse_ciao_date(date_str)
            raw.append((user_id, product, ts))

    print(f"  Raw reviews: {len(raw)}")


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

    print(f"  After {min_user_interactions}/{min_item_interactions}-core filtering: "
          f"{len(df)} interactions, {df['user'].nunique()} users, {df['item'].nunique()} items")

    u_map = {uid: idx for idx, uid in enumerate(sorted(df["user"].unique()))}
    i_map = {iid: idx for idx, iid in enumerate(sorted(df["item"].unique()))}

    interactions = [(u_map[u], i_map[i]) for u, i in zip(df["user"], df["item"])]
    timestamps = {
        (u_map[u], i_map[i]): float(t)
        for u, i, t in zip(df["user"], df["item"], df["ts"])
    }
    return interactions, timestamps, len(u_map), len(i_map)


def load_ml1m_chronological_streaming(
    data_root: str,
    init_ratio: float = 0.5,
    n_batches: int = 10,
):
    data_dir = os.path.join(data_root, "ml-1m")
    ratings_file = os.path.join(data_dir, "ratings.dat")
    if not os.path.exists(ratings_file):
        raise FileNotFoundError(
            f"Missing {ratings_file}. Place MovieLens-1M under {data_root}/ml-1m/."
        )

    df = pd.read_csv(
        ratings_file,
        sep="::",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    df = df[df["rating"] >= 4]
    df = df.sort_values("timestamp").reset_index(drop=True)

    u_map = {uid: idx for idx, uid in enumerate(sorted(df["user_id"].unique()))}
    i_map = {iid: idx for idx, iid in enumerate(sorted(df["item_id"].unique()))}
    df["user_id"] = df["user_id"].map(u_map)
    df["item_id"] = df["item_id"].map(i_map)
    n_users, n_items = len(u_map), len(i_map)

    interactions = list(zip(df["user_id"].values, df["item_id"].values))
    timestamps = {
        (u, i): float(t)
        for u, i, t in zip(df["user_id"], df["item_id"], df["timestamp"])
    }

    split = int(len(interactions) * init_ratio)
    init_data = interactions[:split]
    stream_data = interactions[split:]
    chunk = len(stream_data) // n_batches
    batches = []
    for b in range(n_batches):
        lo = b * chunk
        hi = lo + chunk if b < n_batches - 1 else len(stream_data)
        batches.append(stream_data[lo:hi])

    return init_data, batches, timestamps, n_users, n_items


def load_ciao_chronological_streaming(
    data_root: str,
    init_ratio: float = 0.5,
    n_batches: int = 10,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
):
    data_dir = os.path.join(data_root, "ciao")
    ratings_file = os.path.join(data_dir, "rating.txt")
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
            ts = _parse_ciao_date(date_str)
            raw.append((user_id, product, ts))

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
    batches = []
    for b in range(n_batches):
        lo = b * chunk
        hi = lo + chunk if b < n_batches - 1 else len(stream_data)
        batches.append(stream_data[lo:hi])

    return init_data, batches, timestamps, n_users, n_items


def load_gowala_chronological_streaming(
    data_root: str,
    init_ratio: float = 0.5,
    n_batches: int = 10,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    max_interactions: int = 300_000,
):
    data_dir = os.path.join(data_root, "gowala")
    checkins_file = os.path.join(data_dir, "Gowalla_totalCheckins.txt")
    if not os.path.exists(checkins_file):
        raise FileNotFoundError(
            f"Missing {checkins_file}. Place Gowalla files under {data_root}/gowala/."
        )

    df = pd.read_csv(
        checkins_file,
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
    batches = []
    for b in range(n_batches):
        lo = b * chunk
        hi = lo + chunk if b < n_batches - 1 else len(stream_data)
        batches.append(stream_data[lo:hi])

    return init_data, batches, timestamps, n_users, n_items


def _parse_yelp_date_streaming(date_str):
    try:
        dt = datetime.strptime(date_str.strip()[:19], "%Y-%m-%d %H:%M:%S")
        return float(dt.timestamp())
    except Exception:
        return 0.0


def load_yelp_chronological_streaming(
    data_root: str,
    init_ratio: float = 0.5,
    n_batches: int = 10,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    max_interactions: int = 300_000,
    min_stars: float = 4.0,
):
    reviews_file = os.path.join(data_root, "yelp", "yelp_academic_dataset_review.json")
    if not os.path.exists(reviews_file):
        raise FileNotFoundError(
            f"Missing {reviews_file}. Place Yelp Academic JSON under {data_root}/yelp/."
        )

    raw = []
    with open(reviews_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
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
            raw.append((str(uid), str(bid), _parse_yelp_date_streaming(str(ds))))

    df = pd.DataFrame(raw, columns=["user", "item", "ts"])
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
    batches = []
    for b in range(n_batches):
        lo = b * chunk
        hi = lo + chunk if b < n_batches - 1 else len(stream_data)
        batches.append(stream_data[lo:hi])

    return init_data, batches, timestamps, n_users, n_items


def split_data(interactions, test_ratio=0.2, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(interactions))
    split = int(len(interactions) * (1 - test_ratio))
    train = [interactions[i] for i in idx[:split]]
    test = [interactions[i] for i in idx[split:]]
    return train, test


def create_negative_samples(users, user_pos_items, n_items, num_negatives=1):
    negatives = []
    for u in users:
        pos = user_pos_items.get(int(u), set())
        negs = []
        while len(negs) < num_negatives:
            j = np.random.randint(0, n_items)
            if j not in pos:
                negs.append(j)
        negatives.extend(negs)
    return np.array(negatives)


def load_dataset(dataset_name, data_root):
    data_dir = os.path.join(data_root, dataset_name)
    if dataset_name == "ml-1m":
        interactions, timestamps, n_users, n_items = load_ml1m(data_dir)
    elif dataset_name == "ciao":
        interactions, timestamps, n_users, n_items = load_ciao(data_dir)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train, test = split_data(interactions)
    return train, test, timestamps, n_users, n_items
