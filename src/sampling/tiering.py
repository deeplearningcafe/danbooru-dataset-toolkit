import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Set
from ..prompts.prompt_utils import count_tags
from .filters import is_artist_sample


def create_quality_tiers(
    df: pd.DataFrame,
    negative_tags_list: Optional[List[str]] = None,
    quality_tags_list: Optional[List[str]] = None,
    single_characters: Optional[bool] = False,
    reports_dir: str = "reports",
    artist_tags: Optional[Set[str]] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filters and categorizes dataframe into four quality tiers:

    masterpiece, good_score, bad_score, worse_score.
    Ensures artist samples are protected and assigned at least bad_score.
    """
    initial_row_count = len(df)
    if verbose:
        print(f"Original DataFrame size: {initial_row_count}")

    required_cols = ["id", "score", "fav_count", "aes_score", "tag_string"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Core columns {missing_cols} missing in DataFrame.")

    df_processed = df.copy()
    initial_row_count = len(df)

    if "aes_score" in df_processed.columns:
        df_processed.dropna(subset=["aes_score"], inplace=True)
        if verbose:
            removed_na_count = initial_row_count - len(df_processed)
            if removed_na_count > 0:
                print(
                    f"Removed {removed_na_count} rows with NaN 'aes_score' "
                    f"values. Size after NaN drop: {len(df_processed)}"
                )
            else:
                print("No rows with NaN 'aes_score' values found to remove.")

    if "is_deleted" in df_processed.columns:
        df_processed["is_deleted"] = df_processed["is_deleted"].astype("boolean")
        count_before = len(df_processed)
        df_processed = df_processed[df_processed["is_deleted"].fillna(False) == False]
        if verbose:
            print(
                f"Removed {count_before - len(df_processed)} 'is_deleted' "
                f"items. Size: {len(df_processed)}"
            )

    if "is_banned" in df_processed.columns:
        df_processed["is_banned"] = df_processed["is_banned"].astype("boolean")
        count_before = len(df_processed)
        df_processed = df_processed[df_processed["is_banned"].fillna(False) == False]
        if verbose:
            print(
                f"Removed {count_before - len(df_processed)} 'is_banned' "
                f"items. Size: {len(df_processed)}"
            )

    if "md5" in df_processed.columns:
        df_processed.sort_values("score", ascending=False, inplace=True)
        count_before = len(df_processed)
        df_processed.drop_duplicates(subset=["md5"], keep="first", inplace=True)
        df_processed.sort_index(inplace=True)
        if verbose:
            print(
                f"Removed {count_before - len(df_processed)} MD5 duplicates "
                f"(kept highest score). Size: {len(df_processed)}"
            )

    if "image_width" in df_processed.columns and "image_height" in df_processed.columns:
        pixels = df_processed["image_width"] * df_processed["image_height"]
        count_before = len(df_processed)
        df_processed = df_processed[pixels >= 512 * 512]
        if verbose:
            removed_count = count_before - len(df_processed)
            print(
                f"Removed {removed_count} low-resolution images "
                f"(below 384x384 pixels). Size: {len(df_processed)}"
            )

    artist_mask = is_artist_sample(df_processed, artist_tags)
    protected_ids = set(df_processed.loc[artist_mask, "id"])

    if "parent_id" in df_processed.columns and "id" in df_processed.columns:
        count_before = len(df_processed)
        df_processed["tag_count"] = df_processed["tag_string"].str.split(" ").str.len()
        children_df = df_processed[df_processed["parent_id"].notna()].copy()
        children_df["parent_id"] = children_df["parent_id"].astype(int)
        unique_parents = children_df["parent_id"].unique()
        if verbose:
            print(
                f"Found {len(unique_parents)} unique parents"
                f"With {len(children_df)} childs samples."
            )

        parents_in_df = df_processed[df_processed["id"].isin(unique_parents)]
        parent_lookup = parents_in_df.set_index("id")

        ids_to_drop = []
        for parent_id in unique_parents:
            if parent_id not in parent_lookup.index:
                continue

            p_row = parent_lookup.loc[parent_id]
            curr_children = children_df[children_df["parent_id"] == parent_id]
            n_child = len(curr_children)

            if n_child == 1:
                c_row = curr_children.iloc[0]
                if c_row["tag_count"] <= p_row["tag_count"]:
                    ids_to_drop.append(c_row["id"])
            elif n_child == 2:
                sorted_c = curr_children.sort_values(
                    by=["tag_count", "score"], ascending=[False, False]
                )
                ids_to_drop.append(sorted_c.iloc[1]["id"])
            elif n_child >= 3:
                c_max_tags = curr_children.loc[curr_children["tag_count"].idxmax()]
                c_max_score = curr_children.loc[curr_children["score"].idxmax()]
                keep_ids = {c_max_tags["id"], c_max_score["id"]}
                for cid in curr_children["id"]:
                    if cid not in keep_ids:
                        ids_to_drop.append(cid)

        ids_to_drop = [i for i in ids_to_drop if i not in protected_ids]
        if ids_to_drop:
            df_processed = df_processed[~df_processed["id"].isin(ids_to_drop)]

        df_processed.drop(columns=["tag_count"], inplace=True)

        if verbose:
            removed_count = count_before - len(df_processed)
            print(
                f"Processed parent/child relationships, removing "
                f"{removed_count} child images based on new logic. "
                f"{len(protected_ids)} ids that were from artists and protected"
                f"Size: {len(df_processed)}"
            )

    if negative_tags_list is None:
        negative_tags_list = [
            "bad_anatomy",
            "bad_hands",
            "bad_feet",
            "bad_perspective",
            "english_text",
            "bad_proportions",
            "extra_digits",
            "lowres",
            "jpeg_artifacts",
            "blurry",
            "low_quality",
            "worst_quality",
            "gif_artifacts",
            "corrupted_file",
            "watermark",
            "banned_artist",
            "error",
            "cropped",
            "bad_aspect_ratio",
            "scan_artifacts",
            "scan_dust",
            "adversarial_noise",
        ]

    if quality_tags_list is None:
        quality_tags_list = [
            "highres",
            "absurdres",
            "ultra-detailed",
            "official_art",
            "novel_illustration",
            "official_wallpaper",
            "incredibly_absurdres",
            "key_visual",
            "promotional_art",
        ]

    df_processed["tag_string_lower"] = df_processed["tag_string"].str.lower().fillna("")

    if "tag_count_character" not in df_processed.columns:
        df_processed["tag_count_character"] = (
            df_processed["tag_string_character"]
            .fillna("")
            .str.split(" ")
            .apply(lambda tags: len(tags) if tags != [""] else 0)
        )

    if "tag_count_copyright" not in df_processed.columns:
        df_processed["tag_count_copyright"] = (
            df_processed["tag_string_copyright"]
            .fillna("")
            .str.split(" ")
            .apply(lambda tags: len(tags) if tags != [""] else 0)
        )

    initial_count = len(df_processed)
    max_chars = 4 if not single_characters else 1
    df_processed = df_processed[
        (df_processed["tag_count_character"] <= max_chars)
        & (df_processed["tag_count_copyright"] <= max_chars)
    ]
    if verbose:
        removed_count = initial_count - len(df_processed)
        print(
            f"Removed {removed_count} samples with more than {max_chars} "
            f"characters. Size: {len(df_processed)}"
        )

    if verbose:
        total_removed = initial_row_count - len(df_processed)
        print(f"Total pre-processing removed {total_removed} images.")
        print(f"Proceeding with quality tiering on {len(df_processed)} images.")

    if df_processed.empty:
        empty_df = pd.DataFrame(columns=df.columns.tolist() + ["quality_label"])
        return empty_df, empty_df, empty_df, empty_df

    count_tags(
        df_processed,
        normalized_upsampled_tags=None,
        output_path=os.path.join(reports_dir, "tag_counts_sampling_filtered.csv"),
    )

    df_processed["negative_tag_count"] = 0
    for tag in negative_tags_list:
        df_processed["negative_tag_count"] += (
            df_processed["tag_string_lower"]
            .str.contains(tag.lower(), regex=False)
            .astype(int)
        )

    df_processed["positive_tag_count"] = 0
    for tag in quality_tags_list:
        df_processed["positive_tag_count"] += (
            df_processed["tag_string_lower"]
            .str.contains(tag.lower(), regex=False)
            .astype(int)
        )

    s_p20, s_p60, s_p92 = (
        np.percentile(df_processed["score"].values, [20, 60, 92])
        if len(df_processed["score"].unique()) > 3
        else (0, 0, 0)
    )
    f_p20, f_p60, f_p92 = (
        np.percentile(df_processed["fav_count"].values, [20, 60, 92])
        if len(df_processed["fav_count"].unique()) > 3
        else (0, 0, 0)
    )
    aes_p25, aes_p38, aes_p50 = (
        np.percentile(df_processed["aes_score"].values, [25, 38, 50])
        if len(df_processed["aes_score"].unique()) > 4
        else (0, 0, 0)
    )
    if verbose:
        print(f"Score percentiles: {s_p20, s_p60, s_p92}")
        print(f"Fav count percentiles: {f_p20, f_p60, f_p92}")
        print(f"Aesthetic percentiles: {aes_p25, aes_p38, aes_p50}")
        print(f"Unique num of aes_scores: {len(df_processed['aes_score'].unique())}")

    df_processed["quality_label"] = "unassigned"
    min_edge_high = df_processed[["image_width", "image_height"]].min(axis=1) >= 512

    masterpiece_mask = (
        (df_processed["score"] >= s_p92)
        & (df_processed["fav_count"] >= f_p92)
        & (df_processed["aes_score"] >= aes_p50)
        & (df_processed["negative_tag_count"] <= 1)
        & (df_processed["positive_tag_count"] > 0)
        & min_edge_high
    )
    df_processed.loc[masterpiece_mask, "quality_label"] = "masterpiece"

    current_artist_mask = df_processed["id"].isin(protected_ids)

    worse_mask_tags = (
        (df_processed["quality_label"] == "unassigned")
        & (df_processed["negative_tag_count"] >= 2)
        & (~current_artist_mask)
    )
    if "banned_artist" in negative_tags_list:
        worse_mask_tags |= (
            (df_processed["quality_label"] == "unassigned")
            & df_processed["tag_string_lower"].str.contains(
                "banned_artist", regex=False
            )
            & (~current_artist_mask)
        )

    worse_mask_metrics = (
        (df_processed["quality_label"] == "unassigned")
        & (df_processed["score"] < s_p20)
        & (df_processed["fav_count"] < f_p20)
        & (df_processed["aes_score"] < aes_p25)
        & (~current_artist_mask)
    )
    df_processed.loc[worse_mask_tags | worse_mask_metrics, "quality_label"] = (
        "worse_score"
    )

    good_score_mask = (
        (df_processed["quality_label"] == "unassigned")
        & (df_processed["score"] >= s_p60)
        & (df_processed["fav_count"] >= f_p60)
        & (df_processed["aes_score"] >= aes_p38)
        & (df_processed["negative_tag_count"] <= 2)
    )
    df_processed.loc[good_score_mask, "quality_label"] = "good_score"

    df_processed.loc[df_processed["quality_label"] == "unassigned", "quality_label"] = (
        "bad_score"
    )

    if "new_aesthetic_label" in df_processed.columns:
        demotion_mask = df_processed["quality_label"].isin(
            ["masterpiece", "good_score"]
        ) & (df_processed["new_aesthetic_label"] == "worst")
        demote_non_artist = demotion_mask & (~current_artist_mask)
        demote_artist = demotion_mask & current_artist_mask

        df_processed.loc[demote_non_artist, "quality_label"] = "worse_score"
        df_processed.loc[demote_artist, "quality_label"] = "bad_score"

        promotion_mask = df_processed["quality_label"].isin(
            ["bad_score", "worse_score"]
        ) & (df_processed["new_aesthetic_label"] == "best")
        df_processed.loc[promotion_mask, "quality_label"] = "good_score"

    worse_artist_fix = (
        df_processed["quality_label"] == "worse_score"
    ) & current_artist_mask
    df_processed.loc[worse_artist_fix, "quality_label"] = "bad_score"

    masterpiece_df = df_processed[df_processed["quality_label"] == "masterpiece"].copy()
    good_score_df = df_processed[df_processed["quality_label"] == "good_score"].copy()
    bad_score_df = df_processed[df_processed["quality_label"] == "bad_score"].copy()
    worse_score_df = df_processed[df_processed["quality_label"] == "worse_score"].copy()

    if verbose:
        print("\n--- Tier Statistics ---")
        print(f"Total images processed for tiering: {len(df_processed)}")
        tier_info = {
            "Masterpiece": masterpiece_df,
            "Good Score": good_score_df,
            "Bad Score": bad_score_df,
            "Worse Score": worse_score_df,
        }
        for name, T_df in tier_info.items():
            print(f"\n{name} Tier: {len(T_df)} rows")
            if not T_df.empty:
                print(
                    f"  Score range: {T_df['score'].min():.1f} - "
                    f"{T_df['score'].max():.1f}, "
                    f"Mean: {T_df['score'].mean():.1f}"
                )
                print(
                    f"  Fav range: {T_df['fav_count'].min():.1f} - "
                    f"{T_df['fav_count'].max():.1f}, "
                    f"Mean: {T_df['fav_count'].mean():.1f}"
                )
                print(
                    f"  Aes_score range: {T_df['aes_score'].min():.2f} - "
                    f"{T_df['aes_score'].max():.2f}, "
                    f"Mean: {T_df['aes_score'].mean():.2f}"
                )
                print(f"  Avg Negative Tags: {T_df['negative_tag_count'].mean():.2f}")
                if "rating" in T_df.columns:  # If rating column exists
                    dist_str = ", ".join(
                        [
                            f"{idx}: {val:.1%}"
                            for idx, val in T_df["rating"]
                            .value_counts(normalize=True)
                            .items()
                        ]
                    )
                    print(f"  Rating Distribution: {dist_str}")
            else:
                print("  Tier is empty.")

    return masterpiece_df, good_score_df, bad_score_df, worse_score_df
