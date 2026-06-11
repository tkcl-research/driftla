
from __future__ import annotations

from typing import Any, List, Tuple

from driftla.src.streaming.data import (
    load_amazon23_category_chronological,
    load_ciao_chronological,
    load_gowala_chronological,
    load_ml1m_chronological,
    load_ml10m_chronological,
    load_ml20m_chronological,
    load_yelp_chronological,
)


def load_chronological(
    dataset: str,
    data_root: str,
    max_interactions: int,
    kcore: int = 0,
    amz23_category: str = "",
) -> Tuple[
    List[Tuple[int, int]],
    List[List[Tuple[int, int]]],
    Any,
    int,
    int,
]:
    if dataset == "ml-1m":
        return load_ml1m_chronological(data_root, init_ratio=0.5, n_batches=10)
    if dataset == "ml-10m":
        mi = None if max_interactions <= 0 else max_interactions
        return load_ml10m_chronological(data_root, init_ratio=0.5, n_batches=10, max_interactions=mi)
    if dataset == "ml-20m":
        mi = None if max_interactions <= 0 else max_interactions
        return load_ml20m_chronological(data_root, init_ratio=0.5, n_batches=10, max_interactions=mi)
    if dataset == "amz23_digital_music":
        mi = None if max_interactions <= 0 else max_interactions
        k = int(kcore) if kcore > 0 else 5
        return load_amazon23_category_chronological(
            data_root,
            category="Digital_Music",
            init_ratio=0.5,
            n_batches=10,
            min_user_interactions=k,
            min_item_interactions=k,
            max_interactions=mi,
            min_rating=4.0,
        )
    if dataset == "amz23_all_beauty":
        mi = None if max_interactions <= 0 else max_interactions
        k = int(kcore) if kcore > 0 else 5
        return load_amazon23_category_chronological(
            data_root,
            category="All_Beauty",
            init_ratio=0.5,
            n_batches=10,
            min_user_interactions=k,
            min_item_interactions=k,
            max_interactions=mi,
            min_rating=4.0,
        )
    if dataset == "amazon23":
        mi = None if max_interactions <= 0 else max_interactions
        k = int(kcore) if kcore > 0 else 5
        cat = (amz23_category or "").strip()
        if not cat:
            raise ValueError("--amz23_category is required when --dataset amazon23")
        return load_amazon23_category_chronological(
            data_root,
            category=cat,
            init_ratio=0.5,
            n_batches=10,
            min_user_interactions=k,
            min_item_interactions=k,
            max_interactions=mi,
            min_rating=4.0,
        )
    if dataset == "ciao":
        return load_ciao_chronological(data_root, init_ratio=0.5, n_batches=10)
    if dataset == "gowala":
        mi = None if max_interactions <= 0 else max_interactions
        k = int(kcore) if kcore > 0 else 5
        return load_gowala_chronological(
            data_root,
            init_ratio=0.5,
            n_batches=10,
            min_user_interactions=k,
            min_item_interactions=k,
            max_interactions=mi,
        )
    if dataset == "gowala_dense":

        mi = None if max_interactions <= 0 else max_interactions
        k = int(kcore) if kcore > 0 else 20
        return load_gowala_chronological(
            data_root,
            init_ratio=0.5,
            n_batches=10,
            min_user_interactions=k,
            min_item_interactions=k,
            max_interactions=mi,
        )
    if dataset == "yelp":
        mi = None if max_interactions <= 0 else max_interactions
        k = int(kcore) if kcore > 0 else 5
        return load_yelp_chronological(
            data_root,
            init_ratio=0.5,
            n_batches=10,
            min_user_interactions=k,
            min_item_interactions=k,
            max_interactions=mi,
        )
    if dataset == "yelp_dense":
        mi = None if max_interactions <= 0 else max_interactions
        k = int(kcore) if kcore > 0 else 10
        return load_yelp_chronological(
            data_root,
            init_ratio=0.5,
            n_batches=10,
            min_user_interactions=k,
            min_item_interactions=k,
            max_interactions=mi,
        )
    raise ValueError(f"Unknown dataset: {dataset}")
