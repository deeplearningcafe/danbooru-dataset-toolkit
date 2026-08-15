import pandas as pd
from typing import Optional, Set


def impute_missing_metrics(
    df: pd.DataFrame,
    artist_tags: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Imputes missing metrics (aes_score, fav_count, score) using

    hierarchical fallback: Artist Mean -> Character Mean -> Global Mean.
    """
    if df.empty:
        return df

    metrics = [c for c in ["aes_score", "fav_count", "score"] if c in df.columns]
    if not metrics:
        return df

    work_df = df.copy()

    for col in metrics:
        if work_df[col].isna().sum() == 0:
            continue

        global_mean = work_df[col].mean()
        if pd.isna(global_mean):
            global_mean = 0.0

        # Step 1: Artist-level mean fallback
        if "tag_string_artist" in work_df.columns:
            valid_art = work_df.dropna(subset=[col, "tag_string_artist"])
            if not valid_art.empty:
                art_exploded = valid_art.assign(
                    art=lambda x: x["tag_string_artist"].str.split(" ")
                ).explode("art")
                artist_means = art_exploded.groupby("art")[col].mean().to_dict()

                def _get_artist_mean(tags_str):
                    if not isinstance(tags_str, str) or not tags_str:
                        return None
                    means = [
                        artist_means[t]
                        for t in tags_str.split(" ")
                        if t in artist_means
                    ]
                    return sum(means) / len(means) if means else None

                art_mask = work_df[col].isna()
                imp_art = work_df.loc[art_mask, "tag_string_artist"].map(
                    _get_artist_mean
                )
                work_df.loc[art_mask, col] = imp_art

        # Step 2: Character-level mean fallback
        if "tag_string_character" in work_df.columns:
            char_mask = work_df[col].isna()
            if char_mask.any():
                valid_char = work_df.dropna(subset=[col, "tag_string_character"])
                if not valid_char.empty:
                    chr_exploded = valid_char.assign(
                        chr=lambda x: x["tag_string_character"].str.split(" ")
                    ).explode("chr")
                    char_means = chr_exploded.groupby("chr")[col].mean().to_dict()

                    def _get_char_mean(tags_str):
                        if not isinstance(tags_str, str) or not tags_str:
                            return None
                        means = [
                            char_means[t]
                            for t in tags_str.split(" ")
                            if t in char_means
                        ]
                        return sum(means) / len(means) if means else None

                    imp_char = work_df.loc[char_mask, "tag_string_character"].map(
                        _get_char_mean
                    )
                    work_df.loc[char_mask, col] = imp_char

        # Step 3: Global mean fallback
        work_df[col] = work_df[col].fillna(global_mean)

    return work_df
