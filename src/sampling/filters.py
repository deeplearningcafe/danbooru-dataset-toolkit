import re
import numpy as np
import pandas as pd
from typing import Dict, Set, Optional


def is_artist_sample(df: pd.DataFrame, artist_tags: Optional[Set[str]]) -> pd.Series:
    """Returns a boolean mask where True indicates an artist sample."""
    if not artist_tags or "tag_string_artist" not in df.columns:
        return pd.Series(False, index=df.index)

    escaped = [re.escape(a) for a in artist_tags if a]
    if not escaped:
        return pd.Series(False, index=df.index)

    pattern = r"(?:^|\s)(" + "|".join(escaped) + r")(?:$|\s)"
    return df["tag_string_artist"].str.contains(pattern, regex=True, na=False)


def apply_skip_tags(
    df: pd.DataFrame,
    skip_tags: Dict[str, float],
    rng: np.random.RandomState,
    probability_multiplier: float = 1.0,
    artist_tags: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Applies probabilistic tag filtering.

    Artist samples are explicitly protected from being skipped.
    """
    if not skip_tags or df.empty:
        return df

    work_df = df.copy()
    rows_to_keep = pd.Series(True, index=work_df.index)

    artist_mask = is_artist_sample(work_df, artist_tags)

    for tag, base_prob in skip_tags.items():
        effective_prob = base_prob * probability_multiplier
        if effective_prob > 0:
            mask = work_df["tag_string"].str.contains(
                f"\\b{tag}\\b", case=False, na=False, regex=True
            )
            candidate_indices = work_df.index[mask & (~artist_mask)]

            if not candidate_indices.empty:
                skip_mask = rng.rand(len(candidate_indices)) < effective_prob
                rows_to_keep.loc[candidate_indices[skip_mask]] = False

    return work_df[rows_to_keep]


def sample_face_dataset(
    df: pd.DataFrame, output_csv: str, verbose: bool = True
) -> pd.DataFrame:
    """Filters dataset for anime faces enforcing 1girl/1boy, solo."""
    if verbose:
        print(f"Original DataFrame size: {len(df)}")

    if "tag_count_character" not in df.columns:
        df["tag_count_character"] = (
            df["tag_string_character"]
            .fillna("")
            .str.split(" ")
            .apply(lambda tags: len(tags) if tags != [""] else 0)
        )
    df = df[df["tag_count_character"] <= 1]

    mask_solo = df["tag_string_general"].str.contains(r"\bsolo\b", regex=True, na=False)
    mask_gender = df["tag_string_general"].str.contains(
        r"\b(1girl|1boy)\b", regex=True, na=False
    )
    df = df[mask_solo & mask_gender]

    exclude_meta = [
        "animated",
        "duplicate",
        "pixel-perfect_duplicate",
        "lowres",
        "watermark",
    ]
    if "tag_string_meta" in df.columns:
        mask_meta = ~df["tag_string_meta"].str.contains(
            "|".join(exclude_meta), regex=True, na=False
        )
        df = df[mask_meta]

    if "image_width" in df.columns and "image_height" in df.columns:
        df = df[(df["image_width"] >= 256) & (df["image_height"] >= 256)]

    if verbose:
        print(f"Filtered face dataset size: {len(df)}")

    df.to_csv(output_csv, index=False)
    return df
