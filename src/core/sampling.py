import numpy as np
import pandas as pd
import os
import re
from typing import Tuple, Dict, List, Optional
from ..prompts.prompt_utils import (
    get_tags_from_file,
    count_tags,
    analyze_tag_distribution,
)


def load_and_merge_classifier_labels(
    df: pd.DataFrame, classifier_csv_path: str, verbose: bool = True
) -> pd.DataFrame:
    """
    Loads aesthetic labels from a CSV and merges them into the main DataFrame.

    The function assumes the CSV has 'relative_path' and 'aesthetic_label'
    columns. It extracts the image ID from the path to perform the merge.

    Args:
        df (pd.DataFrame): The main dataframe with an 'id' column.
        classifier_csv_path (str): Path to the CSV from the classifier.
        verbose (bool): If True, prints status messages.

    Returns:
        pd.DataFrame: The dataframe with a new 'new_aesthetic_label' column.
    """
    if not os.path.exists(classifier_csv_path):
        if verbose:
            print(
                f"Warning: Classifier labels file not found at "
                f"'{classifier_csv_path}'. Skipping refinement step."
            )
        return df

    if verbose:
        print(f"Loading new aesthetic labels from '{classifier_csv_path}'...")

    labels_df = pd.read_csv(classifier_csv_path)

    # --- Extract ID from relative_path ---
    # Example path: '3/123456.jpg' -> '123456'
    def extract_id(path):
        try:
            # Get filename, remove extension, convert to int
            return int(os.path.splitext(os.path.basename(path))[0])
        except (ValueError, TypeError):
            return None  # Return None if conversion fails

    labels_df["id"] = labels_df["relative_path"].apply(extract_id)
    labels_df.dropna(subset=["id"], inplace=True)
    labels_df["id"] = labels_df["id"].astype(int)

    # Rename for clarity and select relevant columns
    labels_df = labels_df[["id", "aesthetic_label"]].rename(
        columns={"aesthetic_label": "new_aesthetic_label"}
    )

    # Merge into the main dataframe
    initial_rows = len(df)
    df = pd.merge(df, labels_df, on="id", how="left")

    if verbose:
        merged_count = df["new_aesthetic_label"].notna().sum()
        print(f"Successfully merged {merged_count} new aesthetic labels.")
        # Verify no rows were added or lost unexpectedly
        assert len(df) == initial_rows, "Merge should not change row count."

    return df


def _apply_skip_tags(
    df: pd.DataFrame,
    skip_tags: Dict[str, float],
    rng: np.random.RandomState,
    probability_multiplier: float = 1.0,
) -> pd.DataFrame:
    """
    Applies probabilistic tag-based filtering to a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        skip_tags (Dict[str, float]): A dictionary where keys are tags to
                                      check for and values are the base
                                      probabilities of skipping a row
                                      if the tag is present.
        rng (np.random.RandomState): The random number generator for
                                     probabilistic skipping.
        probability_multiplier (float): A factor to scale the skip
                                        probabilities. Defaults to 1.0.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    if not skip_tags or df.empty:
        return df

    work_df = df.copy()
    rows_to_keep = pd.Series(True, index=work_df.index)

    for tag, base_prob in skip_tags.items():
        # Apply the multiplier to get the effective probability
        effective_prob = base_prob * probability_multiplier
        if effective_prob > 0:
            # Find rows containing the tag as a whole word
            mask = work_df["tag_string"].str.contains(
                f"\\b{tag}\\b", case=False, na=False, regex=True
            )
            indices = work_df.index[mask]

            # For the identified rows, decide which ones to skip
            if not indices.empty:
                skip_mask = rng.rand(len(indices)) < effective_prob
                rows_to_keep.loc[indices[skip_mask]] = False

    return work_df[rows_to_keep]


# --- Updated create_quality_tiers function ---
def create_quality_tiers(
    df: pd.DataFrame,
    negative_tags_list: Optional[List[str]] = None,
    quality_tags_list: Optional[List[str]] = None,
    single_characters: Optional[bool] = False,
    reports_dir: str = "reports",
    artist_tags: Optional[set] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Filters and categorizes the dataframe into four quality tiers:
    "masterpiece", "good_score", "bad_score", and "worse_score".
    This version prioritizes quality definitions using EDA-informed thresholds
    for score, fav_count, aes_score, and tag analysis. It incorporates
    pre-processing logic similar to 'filter_dataset'.
    This version includes a refinement step using a new aesthetic classifier
    to implement a "Veto and Rescue" system for outlier correction.

    Args:
        df (pd.DataFrame): The input dataframe. Must contain 'id', 'score',
                           'fav_count', 'aes_score', 'tag_string'.
                           Optional: 'md5', 'is_deleted', 'is_banned',
                           'parent_id'. Can optionally contain a
                           'new_aesthetic_label' column for refinement.
        negative_tags_list (List[str], optional): List of tags indicating
                                                  lower quality.
        quality_tags_list (List[str], optional): List of tags indicating
                                                 higher quality (e.g., "highres").
        single_characters (bool, optional): if true then samples with 1 single
                                                character will be sampled.
        artist_tags (set, optional): Set of artist tags to protect from
                                     parent/child deduplication.
        verbose (bool): Whether to print info about the tiers and processing.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        Dataframes for "masterpiece", "good_score", "bad_score",
        and "worse_score" tiers.
    """
    initial_row_count = len(df)
    if verbose:
        print(f"Original DataFrame size: {initial_row_count}")

    # --- Start of Pre-processing ---
    # Validate core columns
    required_core_cols = ["id", "score", "fav_count", "aes_score", "tag_string"]
    missing_core_cols = [col for col in required_core_cols if col not in df.columns]
    if missing_core_cols:
        raise ValueError(f"Core columns {missing_core_cols} not found in DataFrame.")

    df_processed = df.copy()
    # --- Start of NaN Filtering for 'aes_score' ---
    # This step is crucial as NaN values in 'aes_score' would cause
    # np.percentile to return NaN, breaking subsequent filtering logic.
    # We drop rows where 'aes_score' is NaN.
    if "aes_score" in df_processed.columns:
        count_before_na_drop = len(df_processed)
        df_processed.dropna(subset=["aes_score"], inplace=True)
        if verbose:
            removed_na_count = count_before_na_drop - len(df_processed)
            if removed_na_count > 0:
                print(
                    f"Removed {removed_na_count} rows with NaN 'aes_score' "
                    f"values. Size after NaN drop: {len(df_processed)}"
                )
            else:
                print("No rows with NaN 'aes_score' values found to remove.")
    else:
        # This case should ideally be caught by the core column check later,
        # but good to be defensive.
        if verbose:
            print("Warning: 'aes_score' column not found for NaN checking.")

    # STEP 1: Filter out deleted or banned images
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

    # STEP 2: Handle MD5 duplicates (keep highest score)
    if "md5" in df_processed.columns:
        count_before = len(df_processed)
        df_processed.sort_values("score", ascending=False, inplace=True)
        df_processed.drop_duplicates(subset=["md5"], keep="first", inplace=True)
        df_processed.sort_index(inplace=True)  # Restore original index order
        if verbose:
            print(
                f"Removed {count_before - len(df_processed)} MD5 duplicates "
                f"(kept highest score). Size: {len(df_processed)}"
            )

    # Filter out low-resolution images
    if "image_width" in df_processed.columns and "image_height" in df_processed.columns:
        count_before = len(df_processed)
        # Calculate the total number of pixels for each image.
        # The calculation is done element-wise for efficiency.
        image_pixels = df_processed["image_width"] * df_processed["image_height"]
        # Define the minimum resolution threshold.
        # we must use a lower than 500x500 to include low_res samples
        min_resolution = 512 * 512
        # Filter out images with a pixel count below the threshold.
        df_processed = df_processed[image_pixels >= min_resolution]
        if verbose:
            removed_count = count_before - len(df_processed)
            print(
                f"Removed {removed_count} low-resolution images "
                f"(below 384x384 pixels). Size: {len(df_processed)}"
            )
    else:
        if verbose:
            print(
                "Warning: 'image_width' or 'image_height' columns not found. "
                "Skipping low-resolution image filtering."
            )

    # STEP 3: Handle parent/child relationships with nuanced logic
    if "parent_id" in df_processed.columns and "id" in df_processed.columns:
        count_before = len(df_processed)

        # Pre-calculate tag counts for efficiency. As per the project's
        # requirements, the number of tags is determined by splitting the
        # space-delimited tag_string.
        df_processed["tag_count"] = df_processed["tag_string"].str.split(" ").str.len()

        # Isolate all child images and their unique parent IDs
        children_df = df_processed[df_processed["parent_id"].notna()].copy()
        children_df["parent_id"] = children_df["parent_id"].astype(int)
        unique_parent_ids = children_df["parent_id"].unique()
        if verbose:
            print(
                f"Found {len(unique_parent_ids)} unique parents ids."
                f"With {len(children_df)} childs samples."
            )
        ids_to_drop = []

        # Identify artist samples to protect them from deduplication
        protected_ids = set()
        if artist_tags and "tag_string_artist" in df_processed.columns:
            escaped_artists = [re.escape(t) for t in artist_tags]
            artist_pattern = r"(?:^|\s)(" + "|".join(escaped_artists) + r")(?:$|\s)"
            artist_mask = df_processed["tag_string_artist"].str.contains(
                artist_pattern, regex=True, na=False
            )
            protected_ids = set(df_processed.loc[artist_mask, "id"])

        # For performance, create a lookup table for parent rows instead of
        # searching the full dataframe in each iteration of the loop.
        parents_in_df = df_processed[df_processed["id"].isin(unique_parent_ids)]
        parent_lookup = parents_in_df.set_index("id")

        for parent_id in unique_parent_ids:
            # If the parent was filtered out in a prior step, skip its children
            if parent_id not in parent_lookup.index:
                continue

            parent_row = parent_lookup.loc[parent_id]
            current_children = children_df[children_df["parent_id"] == parent_id]
            num_children = len(current_children)

            if num_children == 1:
                # Case 1: One child. Keep the parent. Keep the child only if
                # its tag string is longer than the parent's.
                child_row = current_children.iloc[0]
                if child_row["tag_count"] <= parent_row["tag_count"]:
                    ids_to_drop.append(child_row["id"])

            elif num_children == 2:
                # Case 2: Two children. Keep the parent and the single best
                # child. Best is defined as having the longest tag string,
                # with score used as a tie-breaker.
                sorted_children = current_children.sort_values(
                    by=["tag_count", "score"], ascending=[False, False]
                )
                # The second child in the sorted list is the one to drop.
                ids_to_drop.append(sorted_children.iloc[1]["id"])

            elif num_children >= 3:
                # Case 3: Three or more children. Keep the parent, the child
                # with the longest tag string, and the child with the
                # highest score.
                child_max_tags = current_children.loc[
                    current_children["tag_count"].idxmax()
                ]
                child_max_score = current_children.loc[
                    current_children["score"].idxmax()
                ]

                # Use a set to automatically handle the case where the child
                # with the max tags is also the one with the max score.
                ids_to_keep_from_children = {
                    child_max_tags["id"],
                    child_max_score["id"],
                }

                # Drop all children that were not selected.
                for child_id in current_children["id"]:
                    if child_id not in ids_to_keep_from_children:
                        ids_to_drop.append(child_id)

        if ids_to_drop:
            # Filter out protected IDs (artists) from being dropped
            if protected_ids:
                ids_to_drop = [i for i in ids_to_drop if i not in protected_ids]

            if ids_to_drop:
                df_processed = df_processed[~df_processed["id"].isin(ids_to_drop)]

        # Clean up the temporary column used for the logic.
        df_processed = df_processed.drop(columns=["tag_count"])

        if verbose:
            removed_count = count_before - len(df_processed)
            print(
                f"Processed parent/child relationships, removing "
                f"{removed_count} child images based on new logic. "
                f"{len(protected_ids)} ids that were from artists and protected"
                f"Size: {len(df_processed)}"
            )

    # Define default tag lists if not provided
    if negative_tags_list is None:
        negative_tags_list = [
            "bad_anatomy",
            "bad_hands",
            "bad_feet",
            "bad_perspective",
            "text",
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

    # --- Feature Engineering ---
    df_processed["tag_string_lower"] = df_processed["tag_string"].str.lower().fillna("")

    # To ensure the filtering logic is robust, we first check if the
    # tag count columns exist. If not, we create them on the fly from the
    # raw tag string columns. This makes the script more resilient and
    # removes dependencies on prior preprocessing steps.

    # Create 'tag_count_character' if it's missing.
    if "tag_count_character" not in df_processed.columns:
        if verbose:
            print(
                "Column 'tag_count_character' not found. "
                "Creating it from 'tag_string_character'..."
            )
        # We calculate the count by splitting the tag string by spaces.
        # .fillna('') handles empty or NaN entries to prevent errors.
        # The subsequent lambda function correctly assigns a count of 0
        # to empty strings, as split() on '' results in [''].
        df_processed["tag_count_character"] = (
            df_processed["tag_string_character"]
            .fillna("")
            .str.split(" ")
            .apply(lambda tags: len(tags) if tags != [""] else 0)
        )

    # Create 'tag_count_copyright' if it's missing, following the same logic.
    if "tag_count_copyright" not in df_processed.columns:
        if verbose:
            print(
                "Column 'tag_count_copyright' not found. "
                "Creating it from 'tag_string_copyright'..."
            )
        df_processed["tag_count_copyright"] = (
            df_processed["tag_string_copyright"]
            .fillna("")
            .str.split(" ")
            .apply(lambda tags: len(tags) if tags != [""] else 0)
        )

    # Remove samples with more than 4 characters. This is a more robust
    # method than checking for specific tags like "4+girls", as it uses the
    # dedicated 'tag_count_character' column.
    if "tag_count_character" in df_processed.columns:
        initial_count = len(df_processed)
        # Keep rows where the character count is 4 or less. For lora training we need single characters
        max_num_chars = 4 if not single_characters else 1
        df_processed = df_processed[
            df_processed["tag_count_character"] <= max_num_chars
        ]
        df_processed = df_processed[
            df_processed["tag_count_copyright"] <= max_num_chars
        ]

        if verbose:
            removed_count = initial_count - len(df_processed)
            print(
                f"Removed {removed_count} samples with more than {max_num_chars} "
                f"characters. Size: {len(df_processed)}"
            )
    else:
        if verbose:
            print(
                "Warning: 'tag_count_character' column not found. "
                "Skipping character count filtering."
            )
    if verbose:
        total_removed = initial_row_count - len(df_processed)
        print(f"Total pre-processing removed {total_removed} images.")
        print(f"Proceeding with quality tiering on {len(df_processed)} images.")

    if df_processed.empty:
        if verbose:
            print("DataFrame is empty after pre-processing. Returning empty tiers.")
        empty_df = pd.DataFrame(columns=df.columns.tolist() + ["quality_label"])
        return empty_df, empty_df, empty_df, empty_df

    print("Counting tags before applying quality tiers")
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

    # --- Dynamic Threshold Calculation ---
    # Ensure there are enough unique values for percentile calculation
    # Don't use unique values because the high quantiles don't get populated masterpiece is 0 samples
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
        else (0, 0, 0, 0)
    )
    if verbose:
        print(f"Score percentiles: {s_p20, s_p60, s_p92}")
        print(f"Fav count percentiles: {f_p20, f_p60, f_p92}")
        print(f"Aesthetic percentiles: {aes_p25, aes_p38, aes_p50}")
        print(f"Unique num of aes_scores: {len(df_processed['aes_score'].unique())}")

    # --- Tier Assignment ---
    df_processed["quality_label"] = "unassigned"  # Default label
    # Added a resolution gate. Images with a shorter side < 512px
    # cannot be masterpieces, preserving them for lower tiers.
    min_edge_is_high_res = (
        df_processed[["image_width", "image_height"]].min(axis=1) >= 512
    )

    # 1. "Masterpiece" Tier
    masterpiece_mask = (
        (df_processed["score"] >= s_p92)
        & (df_processed["fav_count"] >= f_p92)
        & (df_processed["aes_score"] >= aes_p50)  # Strictest AES
        & (df_processed["negative_tag_count"] <= 1)
        & (df_processed["positive_tag_count"] > 0)  # Encourage positive tags
        & (min_edge_is_high_res)  # The new quality gate
    )
    df_processed.loc[masterpiece_mask, "quality_label"] = "masterpiece"
    if verbose:
        print(f"Masterpiece: Length using score 92%")
        print(len(df_processed.loc[(df_processed["score"] >= s_p92)]))
        print(f"Length using score 92% and fav_count 92%:")
        print(
            len(
                df_processed.loc[
                    (
                        (df_processed["score"] >= s_p92)
                        & (df_processed["fav_count"] >= f_p92)
                    )
                ]
            )
        )
        print(f"Length using score, fav and aes 50%")
        print(
            len(
                df_processed.loc[
                    (
                        (df_processed["score"] >= s_p92)
                        & (df_processed["fav_count"] >= f_p92)
                        & (df_processed["aes_score"] >= aes_p50)
                    )
                ]
            )
        )

    # 2. "Worse Score" Tier (from remaining)
    # Prioritize images with many negative tags or explicitly banned artists
    # Also, very low score/fav/aes
    worse_score_mask_tags = (
        (df_processed["quality_label"] == "unassigned")
        & (df_processed["negative_tag_count"] >= 2)  # More than 1 neg tag
    )
    # Check for 'banned_artist' specifically if it's a critical separator
    if "banned_artist" in negative_tags_list:
        worse_score_mask_tags |= (df_processed["quality_label"] == "unassigned") & (
            df_processed["tag_string_lower"].str.contains("banned_artist", regex=False)
        )

    worse_score_mask_metrics = (
        (df_processed["quality_label"] == "unassigned")
        & (df_processed["score"] < s_p20)
        & (df_processed["fav_count"] < f_p20)
        & (df_processed["aes_score"] < aes_p25)  # Lowest AES
    )
    df_processed.loc[
        worse_score_mask_tags | worse_score_mask_metrics, "quality_label"
    ] = "worse_score"

    # 3. "Good Score" Tier (from remaining)
    good_score_mask = (
        (df_processed["quality_label"] == "unassigned")
        & (df_processed["score"] >= s_p60)
        & (df_processed["fav_count"] >= f_p60)
        & (df_processed["aes_score"] >= aes_p38)  # Mid-to-high AES
        & (df_processed["negative_tag_count"] <= 2)
    )
    if verbose:
        print(f"Good Score: Length using score 60%")
        print(
            len(
                df_processed.loc[
                    (
                        (df_processed["quality_label"] == "unassigned")
                        & (df_processed["score"] >= s_p60)
                    )
                ]
            )
        )
        print(f"Length using score 60% and fav_count 60%:")
        print(
            len(
                df_processed.loc[
                    (
                        (df_processed["quality_label"] == "unassigned")
                        & (df_processed["score"] >= s_p60)
                        & (df_processed["fav_count"] >= f_p60)
                    )
                ]
            )
        )
        print(f"Length using score, fav and aes 38%")
        print(
            len(
                df_processed.loc[
                    (
                        (df_processed["quality_label"] == "unassigned")
                        & (df_processed["score"] >= s_p60)
                        & (df_processed["fav_count"] >= f_p60)
                        & (df_processed["aes_score"] >= aes_p38)
                    )
                ]
            )
        )

    df_processed.loc[good_score_mask, "quality_label"] = "good_score"

    # 4. "Bad Score" Tier (all remaining)
    # These are typically mid-low quality or don't fit stricter criteria
    df_processed.loc[df_processed["quality_label"] == "unassigned", "quality_label"] = (
        "bad_score"
    )

    if "new_aesthetic_label" in df_processed.columns:
        if verbose:
            print("\n--- Refining Tiers with New Aesthetic Classifier ---")

        # Rule 1: Veto (Demotion)
        # Demote images from top tiers if the new model calls them 'worst'.
        demotion_mask = df_processed["quality_label"].isin(
            ["masterpiece", "good_score"]
        ) & (df_processed["new_aesthetic_label"] == "worst")
        num_demoted = demotion_mask.sum()
        if num_demoted > 0:
            df_processed.loc[demotion_mask, "quality_label"] = "worse_score"
            if verbose:
                print(
                    f"Demoted {num_demoted} images from high-quality "
                    "tiers to 'worse_score' based on new classifier."
                )

        # Rule 2: Rescue (Promotion)
        # Promote images from bottom tiers if the new model calls them 'best'.
        promotion_mask = df_processed["quality_label"].isin(
            ["bad_score", "worse_score"]
        ) & (df_processed["new_aesthetic_label"] == "best")
        num_promoted = promotion_mask.sum()
        if num_promoted > 0:
            df_processed.loc[promotion_mask, "quality_label"] = "good_score"
            if verbose:
                print(
                    f"Promoted {num_promoted} images from low-quality "
                    "tiers to 'good_score' based on new classifier."
                )

    # Create the four tier DataFrames
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


def _sample_from_tier(
    tier_df: pd.DataFrame,
    target_count: int,
    ratings_percentage: Dict[str, float],
    include_tags: Optional[Dict[str, float]],
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """
    Internal helper to perform efficient, weighted sampling on a single tier DF.

    This function is the core of the new, efficient sampling logic. It uses
    vectorized operations to handle parent de-duplication and weighted sampling.

    Args:
        tier_df (pd.DataFrame): The dataframe for a single quality tier.
        target_count (int): The total number of images to sample from this tier.
        ratings_percentage (Dict[str, float]): Distribution of ratings.
        include_tags (Optional[Dict[str, float]]): Tags for weighted sampling.
        rng (np.random.RandomState): The random number generator.

    Returns:
        pd.DataFrame: A dataframe of sampled images from the tier.
    """
    if tier_df.empty or target_count == 0:
        return pd.DataFrame(columns=tier_df.columns)

    # --- 1. Parent Group De-duplication (Vectorized) ---
    # We select only the best representative (highest score) from each
    # parent group to ensure we don't sample siblings.
    if "parent_group" in tier_df.columns:
        # Get the index of the row with the max score for each parent group
        best_reps_indices = tier_df.groupby("parent_group")["score"].idxmax()
        work_df = tier_df.loc[best_reps_indices].copy()
    else:
        # If no parent info, the entire tier is eligible
        work_df = tier_df.copy()

    if work_df.empty:
        return pd.DataFrame(columns=work_df.columns)

    # --- 2. Efficient Weight Calculation (Optimized Single-Pass Method) ---
    if include_tags:
        # Define the columns to search for tags in priority order.
        tag_columns = [
            "tag_string_character",
            "tag_string_copyright",
            "tag_string_artist",
            "tag_string",
        ]
        # Filter to only include columns that actually exist in the DataFrame
        # to prevent errors.
        available_tag_columns = [col for col in tag_columns if col in work_df.columns]

        def _calculate_weight_for_row(row: pd.Series) -> float:
            """
            Calculates the sampling weight for a single DataFrame row.

            This inner function is designed to be used with `DataFrame.apply()`.
            It works by extracting all tags from a single row and then checking
            which of those tags are present in the `include_tags` dictionary.
            This avoids repeatedly scanning the entire DataFrame for each tag.
            """
            # Use a set to efficiently store unique tags from the current row.
            row_tags = set()
            for col in available_tag_columns:
                # Ensure the column value is a string before splitting.
                if isinstance(row[col], str):
                    row_tags.update(row[col].split())

            if not row_tags:
                return 1.0

            weight = 1.0
            # Iterate only through the tags present in the current row.
            for tag in row_tags:
                # Check if this tag is in our dictionary of weighted tags.
                # Dictionary lookups are very fast (average O(1) complexity).
                if tag in include_tags:
                    weight *= include_tags[tag]

            return weight

        # Apply the optimized function to each row of the DataFrame.
        # The `axis=1` argument ensures the function is applied row-by-row.
        # This single pass is much more efficient than the previous method.
        work_df["sampling_weight"] = work_df.apply(_calculate_weight_for_row, axis=1)
    else:
        # No weights needed if include_tags is not provided
        work_df["sampling_weight"] = 1.0

    # --- 3. Calculate Per-Rating Targets ---
    # Determine how many samples to draw for each rating category ('g', 's', etc.)
    rating_targets = {
        rating: int(target_count * percentage)
        for rating, percentage in ratings_percentage.items()
    }
    # Distribute remainder to ensure total target is met
    remainder = target_count - sum(rating_targets.values())
    for i in range(remainder):
        rating_to_increment = list(rating_targets.keys())[i % len(rating_targets)]
        rating_targets[rating_to_increment] += 1

    # --- 4. Perform Grouped, Weighted Sampling (First Pass) ---
    # This pass attempts to sample according to the desired rating distribution.
    sampled_dfs = []
    for rating, n_samples in rating_targets.items():
        if n_samples == 0:
            continue

        rating_group = work_df[work_df["rating"] == rating]
        if len(rating_group) == 0:
            continue

        # .sample() is highly optimized C code under the hood.
        # It handles undersampling (n > len(group)) gracefully.
        sampled_dfs.append(
            rating_group.sample(
                n=min(n_samples, len(rating_group)),  # Don't sample more than available
                weights="sampling_weight",
                random_state=rng,
                replace=False,  # Ensure we don't pick the same image twice
            )
        )

    # --- 5. Permissive Filling Logic (Second Pass) ---
    # If the first pass didn't meet target_count, this pass fills the
    # remainder from any available images, ignoring rating distribution.
    if sampled_dfs:
        first_pass_samples = pd.concat(sampled_dfs)
    else:
        first_pass_samples = pd.DataFrame(columns=work_df.columns)

    remaining_needed = target_count - len(first_pass_samples)
    print(
        f"The ratings distribution didn't fill the tier target count, sampling {remaining_needed} samples from all ratings."
    )
    if remaining_needed > 0:
        # Exclude already-sampled images to create a pool of leftovers.
        if not first_pass_samples.empty:
            leftover_df = work_df.drop(first_pass_samples.index)
        else:
            leftover_df = work_df

        if not leftover_df.empty:
            # Sample from the leftovers to fill the gap.
            fill_samples = leftover_df.sample(
                n=min(remaining_needed, len(leftover_df)),
                weights="sampling_weight",
                random_state=rng,
                replace=False,
            )
            final_samples = pd.concat([first_pass_samples, fill_samples])
        else:
            final_samples = first_pass_samples
    else:
        final_samples = first_pass_samples

    if final_samples.empty:
        return pd.DataFrame(columns=work_df.columns)

    # Drop the temporary weight column before returning
    return final_samples.drop(columns=["sampling_weight"])


def filter_and_sample_by_quality(
    df: pd.DataFrame,
    total_samples: int,
    quality_percentages: Dict[str, float],
    ratings_percentage: Dict[str, float],
    prior_knowledge_path: Optional[str] = None,
    knowledge_bases_paths: Optional[List[str]] = None,
    aes_scores_csv_path: Optional[str] = None,
    skip_tags: Optional[Dict[str, float]] = None,
    include_tags: Optional[Dict[str, float]] = None,
    exclude_path: Optional[pd.DataFrame] = None,
    is_lora: bool = False,
    random_seed: int = 42,
    output_csv: str = "sampled_ids.csv",
    reports_dir: str = "reports",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Filters and samples a dataset based on quality tiers, ratings, and tags.

    This function orchestrates the new, efficient pipeline:
    1. Creates quality tiers.
    2. Iteratively calls a helper to sample from each tier.
    3. Manages global state (like sampled parents) cleanly.
    4. Returns a final DataFrame and statistics.
    *This function no longer handles image downloading.*

    Args:
        df (pd.DataFrame): Input dataframe.
        total_samples (int): The total number of images to sample.
        quality_percentages (Dict[str, float]): Percentage for each quality tier.
        ratings_percentage (Dict[str, float]): Percentage for each rating.
        prior_knowledge_path (Optional[str]): Path to the Parquet file.
        artists_txt (Optional[str]): Path to artists list.
        aes_scores_csv_path (Optional[str]): Path to aesthetic scores.
        skip_tags (Optional[Dict[str, float]]): Tags to skip.
        include_tags (Optional[Dict[str, float]]): Tags to weight higher.
        exclude_path (Optional[str]): Path of IDs to exclude.
        character_list (Optional[List[str]]): List of specific character tags
                                              to filter for. If provided, only
                                              images containing these characters
                                              are sampled.
        random_seed (int): Seed for reproducibility.
        output_csv (str): Path to save the CSV.
        verbose (bool): If True, prints detailed logs.

    Returns:
        Tuple[pd.DataFrame, Dict]: The final sampled dataframe and stats.
    """
    print(f"Sampling {total_samples} samples from the dataset.")
    np.random.seed(random_seed)
    rng = np.random.RandomState(random_seed)

    # Create a mutable copy to avoid modifying the original dict passed in.
    local_include_tags = include_tags.copy() if include_tags else {}
    stats = {}
    all_sampled_dfs = []

    artist_list = []
    character_list = []
    if knowledge_bases_paths:
        if len(knowledge_bases_paths) > 0:
            artist_list = get_tags_from_file(knowledge_bases_paths[0])
        if len(knowledge_bases_paths) > 1:
            character_list = get_tags_from_file(knowledge_bases_paths[1])
    artist_tags = set(artist_list)

    # --- Step 0: Load and Prepare Prior Knowledge Dataset ---
    prior_knowledge_samples = pd.DataFrame()
    if prior_knowledge_path and os.path.exists(prior_knowledge_path):
        if verbose:
            print(f"\nStep 0: Loading prior knowledge from '{prior_knowledge_path}'...")
        try:
            prior_knowledge_samples = pd.read_csv(
                prior_knowledge_path, header=0, low_memory=False
            )
            prior_knowledge_samples = (
                prior_knowledge_samples.drop_duplicates().reset_index(drop=True)
            )

            # Merge aesthetic scores for data consistency
            if aes_scores_csv_path and os.path.exists(aes_scores_csv_path):
                aes_df = pd.read_csv(aes_scores_csv_path)
                aes_df = aes_df.rename(columns={"score": "aes_score"})
                prior_knowledge_samples = pd.merge(
                    prior_knowledge_samples,
                    aes_df[["id", "aes_score"]],
                    on="id",
                    how="left",
                )

            # Fill missing aes_score values with the mean to include new images.
            if "aes_score" in prior_knowledge_samples.columns:
                nan_count = prior_knowledge_samples["aes_score"].isna().sum()
                if nan_count > 0:
                    # the highest masterpiece number comes from applying the fillnan here and not after filtering
                    mean_aes = prior_knowledge_samples["aes_score"].mean()
                    prior_knowledge_samples["aes_score"] = prior_knowledge_samples[
                        "aes_score"
                    ].fillna(mean_aes)

                    if verbose:
                        print(
                            f"  - Filled {nan_count} missing 'aes_score' "
                            f"values with the mean ({mean_aes:.4f})."
                        )
            if "fav_count" in prior_knowledge_samples.columns:
                nan_count = prior_knowledge_samples["fav_count"].isna().sum()
                if nan_count > 0:
                    mean_aes = prior_knowledge_samples["fav_count"].mean()
                    prior_knowledge_samples["fav_count"] = prior_knowledge_samples[
                        "fav_count"
                    ].fillna(mean_aes)

                    if verbose:
                        print(
                            f"  - Filled {nan_count} missing 'fav_count' "
                            f"values with the mean ({mean_aes:.4f})."
                        )
            else:
                print(
                    "  - Warning: aes_scores_csv_path not provided. 'aes_score' may be missing."
                )

        except Exception as e:
            print(f"  - Warning: Could not load or process prior knowledge file: {e}")
            prior_knowledge_samples = pd.DataFrame()

    # Set single_characters explicitly based on the is_lora flag,
    # decoupling it from whether a character list was provided.
    single_characters = is_lora

    has_char_filter = character_list and len(character_list) > 0
    has_artist_filter = artist_list and len(artist_list) > 0
    # If a character or artist list is provided, we filter BOTH the prior
    # knowledge and the main dataframe immediately.
    if has_char_filter or has_artist_filter:
        if verbose:
            print(f"\n--- Filtering for specific targets ---")
            if has_char_filter:
                print(f"  Characters: {character_list}")
            if has_artist_filter:
                print(f"  Artists: {artist_list}")

        def apply_filters(temp_df: pd.DataFrame) -> pd.DataFrame:
            if temp_df.empty:
                return temp_df

            # Start with a mask of False. We will use OR logic so if an image
            # has EITHER the requested character OR the requested artist, we keep it.
            final_mask = pd.Series(False, index=temp_df.index)

            if has_char_filter:
                if "tag_string_character" in temp_df.columns:
                    escaped_tags = [re.escape(t) for t in character_list]
                    char_pattern = r"(?:^|\s)(" + "|".join(escaped_tags) + r")(?:$|\s)"
                    char_mask = temp_df["tag_string_character"].str.contains(
                        char_pattern, regex=True, na=False
                    )
                    final_mask = final_mask | char_mask
                elif verbose:
                    print(
                        "  - Warning: 'tag_string_character' missing. "
                        "Skipping character filter."
                    )

            if has_artist_filter:
                if "tag_string_artist" in temp_df.columns:
                    escaped_artists = [re.escape(t) for t in artist_list]
                    artist_pattern = (
                        r"(?:^|\s)(" + "|".join(escaped_artists) + r")(?:$|\s)"
                    )
                    artist_mask = temp_df["tag_string_artist"].str.contains(
                        artist_pattern, regex=True, na=False
                    )
                    final_mask = final_mask | artist_mask
                elif verbose:
                    print(
                        "  - Warning: 'tag_string_artist' missing. "
                        "Skipping artist filter."
                    )

            return temp_df[final_mask]

        # Filter Prior Knowledge
        if not prior_knowledge_samples.empty:
            before_len = len(prior_knowledge_samples)
            prior_knowledge_samples = apply_filters(prior_knowledge_samples)
            if verbose:
                print(
                    f"  - Prior Knowledge filtered from {before_len} to "
                    f"{len(prior_knowledge_samples)} samples."
                )

        # Filter Main DataFrame
        before_len = len(df)
        df = apply_filters(df)
        if verbose:
            print(f"  - Main Dataset filtered from {before_len} to {len(df)} samples.")

        if df.empty and prior_knowledge_samples.empty:
            print("  - Error: No samples found for the requested characters/artists.")
            return pd.DataFrame(), {}

    # Exclude prior knowledge and already-excluded IDs from the main df
    prior_ids = (
        set(prior_knowledge_samples["id"])
        if not prior_knowledge_samples.empty
        else set()
    )
    exclude_df = None
    if exclude_path is not None:
        try:
            exclude_df = pd.read_csv(exclude_path, header=0)
        except FileNotFoundError:
            print(f"The df with path {exclude_path} does not exist. Skipping it.")

    exclude_ids = set(exclude_df["id"]) if exclude_df is not None else set()
    ids_to_filter = prior_ids.union(exclude_ids)

    df = df[~df["id"].isin(ids_to_filter)]
    if verbose:
        print(f"Main dataframe size after filtering: {len(df)}")

    if not prior_knowledge_samples.empty:
        if verbose:
            print("Counting tags from prior dataset before filtering:")
            count_tags(
                prior_knowledge_samples, output_path="tag_counts_prior_knowledge.csv"
            )

            print("\nProcessing prior knowledge dataset for tiering...")
        # the artists might have always high quality but the characters not so we need to create tiers as well
        masterpiece_prior, good_prior, bad_prior, worse_prior = create_quality_tiers(
            prior_knowledge_samples,
            single_characters=single_characters,
            artist_tags=artist_tags,
            verbose=verbose,
        )

        # only generate tags counts for each tier if verbose
        if verbose:
            files_to_process = [
                os.path.join(reports_dir, "tag_counts_masterpiece.csv"),
                os.path.join(reports_dir, "tag_counts_good.csv"),
                os.path.join(reports_dir, "tag_counts_bad.csv"),
                os.path.join(reports_dir, "tag_counts_worse.csv"),
            ]

            for name, df in zip(
                files_to_process,
                [masterpiece_prior, good_prior, bad_prior, worse_prior],
            ):
                count_tags(df, output_path=name)
            final_distribution_df = analyze_tag_distribution(files_to_process)

            if not final_distribution_df.empty:
                output_filename = os.path.join(reports_dir, "merged_tag_counts.csv")
                final_distribution_df.to_csv(output_filename, index=False)
                print(f"Successfully merged tag counts into '{output_filename}'.")
            else:
                print("Analysis resulted in an empty dataframe.")
        # Split each tier into Artists and Characters ---
        if verbose:
            print(
                "  - Splitting prior knowledge tiers into Artists and "
                "Characters subsets..."
            )

        # Ensure artist_tags are available and the necessary column exists
        if artist_tags and "tag_string_artist" in masterpiece_prior.columns:
            # Use regex instead of .isin() to handle space-separated artists
            escaped_artists = [re.escape(t) for t in artist_tags]
            artist_pattern = r"(?:^|\s)(" + "|".join(escaped_artists) + r")(?:$|\s)"

            # 1. Split the Masterpiece Tier
            is_artist_mask_mp = masterpiece_prior["tag_string_artist"].str.contains(
                artist_pattern, regex=True, na=False
            )
            masterpiece_prior_artists = masterpiece_prior[is_artist_mask_mp].copy()
            masterpiece_prior_artists["quality_tier"] = masterpiece_prior_artists[
                "quality_label"
            ]
            masterpiece_prior_chars = masterpiece_prior[~is_artist_mask_mp].copy()
            masterpiece_prior_chars["quality_tier"] = masterpiece_prior_chars[
                "quality_label"
            ]

            # 2. Split the Good Score Tier
            is_artist_mask_good = good_prior["tag_string_artist"].str.contains(
                artist_pattern, regex=True, na=False
            )
            good_prior_artists = good_prior[is_artist_mask_good].copy()
            good_prior_artists["quality_tier"] = good_prior_artists["quality_label"]
            good_prior_chars = good_prior[~is_artist_mask_good].copy()
            good_prior_chars["quality_tier"] = good_prior_chars["quality_label"]

            # 3. Split the Bad Score Tier
            is_artist_mask_bad = bad_prior["tag_string_artist"].str.contains(
                artist_pattern, regex=True, na=False
            )
            bad_prior_artists = bad_prior[is_artist_mask_bad].copy()
            bad_prior_chars = bad_prior[~is_artist_mask_bad].copy()
            bad_prior_chars["quality_tier"] = bad_prior_chars["quality_label"]

            # 4. Split the Worse Score Tier
            is_artist_mask_worse = worse_prior["tag_string_artist"].str.contains(
                artist_pattern, regex=True, na=False
            )
            worse_prior_artists = worse_prior[is_artist_mask_worse].copy()
            worse_prior_chars = worse_prior[~is_artist_mask_worse].copy()
            worse_prior_chars["quality_tier"] = worse_prior_chars["quality_label"]

            worse_prior_artists["quality_label"] = "bad_score"
            num_to_upgrade = len(worse_prior_artists)
            bad_prior_artists = pd.concat(
                [bad_prior_artists, worse_prior_artists], ignore_index=True
            )
            bad_prior_artists["quality_tier"] = bad_prior_artists["quality_label"]

            if verbose:
                print(
                    f"  - Upgraded {num_to_upgrade} artist samples "
                    "from 'worse_score' to 'bad_score'."
                )

            # Apply skip_tags logic with different probabilities ---
            if skip_tags:
                if verbose:
                    print("  - Applying skip_tags to prior knowledge subsets...")

                # --- Artists (half probability) ---
                artist_dfs = {
                    "Masterpiece Artists": masterpiece_prior_artists,
                    "Good Score Artists": good_prior_artists,
                    "Bad Score Artists": bad_prior_artists,
                }
                filtered_artist_dfs = {}
                for name, artist_df in artist_dfs.items():
                    before_count = len(artist_df)
                    filtered_df = _apply_skip_tags(
                        artist_df, skip_tags, rng, probability_multiplier=0.5
                    )
                    after_count = len(filtered_df)
                    filtered_artist_dfs[name] = filtered_df
                    if verbose and before_count > 0:
                        print(
                            f"    - {name}: Skipped "
                            f"{before_count - after_count} "
                            f"images (50% prob)."
                        )

                masterpiece_prior_artists = filtered_artist_dfs["Masterpiece Artists"]
                good_prior_artists = filtered_artist_dfs["Good Score Artists"]
                bad_prior_artists = filtered_artist_dfs["Bad Score Artists"]

            if verbose:
                # Optional: Print stats about the split for verification
                mp_a, mp_c = (
                    len(masterpiece_prior_artists),
                    len(masterpiece_prior_chars),
                )
                go_a, go_c = len(good_prior_artists), len(good_prior_chars)
                ba_a, ba_c = len(bad_prior_artists), len(bad_prior_chars)
                wo_c = len(worse_prior_chars)
                print(f"    Masterpiece: {mp_a} artists, {mp_c} chars")
                print(f"    Good Score:  {go_a} artists, {go_c} chars")
                print(f"    Bad Score:   {ba_a} artists, {ba_c} chars")
                print(f"    Worse Score: {wo_c} chars")

        else:
            # If no artist tags or column, assume all are characters
            if verbose:
                print(
                    "  - Warning: No artist tags found or 'tag_string_artist' "
                    "column missing. Treating all prior samples as characters."
                )
            masterpiece_prior_artists, good_prior_artists, bad_prior_artists = (
                pd.DataFrame(columns=masterpiece_prior.columns) for _ in range(3)
            )

            masterpiece_prior_chars = masterpiece_prior
            good_prior_chars = good_prior
            bad_prior_chars = bad_prior
            worse_prior_chars = worse_prior

        # Set already sampled counts based on the processed prior data.
        already_sampled_tiers = {
            "masterpiece": len(masterpiece_prior_artists),
            "good_score": len(good_prior_artists),
            "bad_score": len(bad_prior_artists),
            "worse_score": 0,
        }

        all_sampled_dfs.extend(
            [masterpiece_prior_artists, good_prior_artists, bad_prior_artists]
        )
        if verbose:
            print("  - Prior knowledge samples included:")
            for tier, count in already_sampled_tiers.items():
                print(f"    - {tier}: {count}")
    else:
        already_sampled_tiers = {
            "masterpiece": 0,
            "good_score": 0,
            "bad_score": 0,
            "worse_score": 0,
        }

    # Handle cases where prior artist knowledge is sufficient
    num_prior_samples = sum(already_sampled_tiers.values())
    if num_prior_samples >= total_samples:
        if verbose:
            print(
                f"Prior knowledge count ({num_prior_samples}) meets target {total_samples}. Sampling only from it."
            )
        final_df = pd.concat(all_sampled_dfs, ignore_index=True)
        final_df = final_df.sample(n=total_samples, random_state=rng)
        final_df["quality_tier"] = final_df["quality_label"]
        final_df[
            [
                "id",
                "quality_tier",
                "rating",
                "tag_string",
                "file_url",
                "large_file_url",
                "image_width",
                "image_height",
            ]
        ].to_csv(output_csv, index=False)
        return final_df, {"prior_knowledge_only": {"sampled": len(final_df)}}

    remaining_samples = total_samples - num_prior_samples
    characters_samples = sum(
        [
            len(masterpiece_prior_chars),
            len(good_prior_chars),
            len(bad_prior_chars),
            len(worse_prior_chars),
        ]
    )
    non_prior_samples = remaining_samples - characters_samples
    if verbose:
        print(
            f"Included {num_prior_samples} prior samples. Need to sample {remaining_samples} more."
            f"With {characters_samples} character samples avaible."
        )

    # --- 1. Initial Setup and Validation ---
    if not np.isclose(sum(quality_percentages.values()), 1.0):
        raise ValueError("quality_percentages must sum to 1.0")
    if not np.isclose(sum(ratings_percentage.values()), 1.0):
        raise ValueError("ratings_percentage must sum to 1.0")

    # handle case when we need more samples than the prior
    if non_prior_samples > 0:
        masterpiece_df, good_df, bad_df, worse_df = create_quality_tiers(
            df,
            single_characters=single_characters,
            reports_dir=reports_dir,
            artist_tags=artist_tags,
            verbose=verbose,
        )
        # always include first the characters and the rest of the
        # data can be sampled
        already_sampled_tiers["masterpiece"] += len(masterpiece_prior_chars)
        already_sampled_tiers["good_score"] += len(good_prior_chars)
        already_sampled_tiers["bad_score"] += len(bad_prior_chars)
        already_sampled_tiers["worse_score"] += len(worse_prior_chars)
        all_sampled_dfs.extend(
            [
                masterpiece_prior_chars,
                good_prior_chars,
                bad_prior_chars,
                worse_prior_chars,
            ]
        )

        num_prior_samples = sum(already_sampled_tiers.values())
        if verbose:
            print(
                f"Included character tags as already sampled. {num_prior_samples} already sampled."
            )

    elif non_prior_samples <= 0:
        masterpiece_df = masterpiece_prior_chars
        good_df = good_prior_chars
        bad_df = bad_prior_chars
        worse_df = worse_prior_chars

    quality_tier_dfs = {
        "masterpiece": masterpiece_df,
        "good_score": good_df,
        "bad_score": bad_df,
        "worse_score": worse_df,
    }

    # --- 2. Pre-filtering (Exclude and Skip Tags) ---
    if verbose:
        print("\nStep 2: Applying global filters (exclude_df, skip_tags)...")

    global_sampled_parents = set()
    for tier_name, tier_df in quality_tier_dfs.items():
        if tier_df.empty:
            continue

        # Create a working copy for this tier
        work_df = tier_df.copy()

        # Filter based on external exclude_df
        work_df = work_df[~work_df["id"].isin(exclude_ids)]

        # Probabilistic skip_tags filter
        if skip_tags:
            initial_count = len(work_df)
            # Replace the old loop with a single, clean function call
            work_df = _apply_skip_tags(
                work_df, skip_tags, rng, probability_multiplier=1.0
            )
            if verbose:
                print(
                    f"  - {tier_name}: Skipped "
                    f"{initial_count - len(work_df)} "
                    "images based on skip_tags."
                )

        # Define parent group: parent_id if it exists, otherwise its own id
        if "parent_id" in work_df.columns:
            work_df["parent_group"] = work_df["parent_id"].fillna(work_df["id"])

        quality_tier_dfs[tier_name] = work_df

    # --- 3. Orchestrate Sampling with Cascading Deficit ---
    if verbose:
        print("\nStep 3: Sampling from each quality tier...")

    # Define tier order to ensure deficit cascades from high to low quality.
    tier_order = ["masterpiece", "good_score", "bad_score", "worse_score"]
    tier_targets = {
        name: int(total_samples * perc) for name, perc in quality_percentages.items()
    }

    # Adjust initial targets based on what was already sampled from prior knowledge.
    for tier_name in tier_order:
        if tier_name in tier_targets:
            adjustment = already_sampled_tiers.get(tier_name, 0)
            tier_targets[tier_name] = max(0, tier_targets[tier_name] - adjustment)

    deficit = 0
    for tier_name in tier_order:
        # Add any deficit from the previous, higher-quality tier.
        current_target = tier_targets.get(tier_name, 0) + deficit
        tier_df = quality_tier_dfs.get(tier_name)

        if tier_df is None or tier_df.empty or current_target <= 0:
            stats[tier_name] = {
                "available": len(tier_df) if tier_df is not None else 0,
                "target": current_target,
                "sampled": 0,
            }
            deficit = current_target  # The full target becomes the deficit.
            continue

        # Filter out any parent groups that have already been sampled in
        # a higher-priority tier.
        tier_df = tier_df[~tier_df["parent_group"].isin(global_sampled_parents)]

        if verbose:
            print(
                f"  - Processing {tier_name}: "
                f"Target={current_target}, Available={len(tier_df)}"
            )

        # Call the efficient, vectorized sampling helper function
        tier_samples = _sample_from_tier(
            tier_df, current_target, ratings_percentage, local_include_tags, rng
        )
        num_sampled = len(tier_samples)
        deficit = current_target - num_sampled  # Update deficit for next tier.

        if not tier_samples.empty:
            # Update the set of parents we've now sampled
            newly_sampled_parents = set(tier_samples["parent_group"].unique())
            global_sampled_parents.update(newly_sampled_parents)

            tier_samples["quality_tier"] = tier_name
            if verbose:
                print(
                    f"{tier_name}  df repeated ids {sum(tier_samples['id'].value_counts() > 1)}"
                )

            all_sampled_dfs.append(tier_samples)

        stats[tier_name] = {
            "available": len(tier_df),
            "target": current_target,
            "sampled": num_sampled,
        }

    # --- 4. Finalize and Report ---
    if not all_sampled_dfs:
        print("\nWarning: No images were sampled. Returning empty DataFrame.")
        return pd.DataFrame(), stats

    final_df = pd.concat(all_sampled_dfs, ignore_index=True)
    if verbose:
        print(
            f"Final df quality_tier nans {sum(final_df['quality_tier'].isna()), sum(final_df['quality_tier'].isnull())}"
        )
        print(
            f"Final df quality_label nans {sum(final_df['quality_label'].isna()), sum(final_df['quality_label'].isnull())}"
        )
        print(f"Final df repeated ids {sum(final_df['id'].value_counts() > 1)}")

    # Final Fill-Up Pass
    # If we are still short of total_samples, fill from any remaining images.
    remaining_needed = total_samples - len(final_df)
    if remaining_needed > 0:
        if verbose:
            print(
                f"  - Main sampling got {len(final_df)}. "
                f"Filling remaining {remaining_needed} slots."
            )

        # Create a set of all IDs that have already been sampled to
        # prevent duplicates in the fill-up pass.
        already_sampled_ids = set(final_df["id"])

        # Create a pool of all available, unsampled images.
        leftover_dfs = []
        for tier_df in quality_tier_dfs.values():
            if not tier_df.empty and "parent_group" in tier_df.columns:
                unsampled = tier_df[
                    ~tier_df["parent_group"].isin(global_sampled_parents)
                ]
                if not unsampled.empty:
                    leftover_dfs.append(unsampled)

        if leftover_dfs:
            leftover_pool = pd.concat(leftover_dfs).drop_duplicates(subset=["id"])

            # Filter the pool to ensure we only sample from images that
            # have not already been selected. This is the core of the fix.
            if not leftover_pool.empty:
                leftover_pool = leftover_pool[
                    ~leftover_pool["id"].isin(already_sampled_ids)
                ]
                fill_samples = leftover_pool.sample(
                    n=min(remaining_needed, len(leftover_pool)), random_state=rng
                )
                # Assign quality_tier based on the original label.
                fill_samples["quality_tier"] = fill_samples["quality_label"]
                final_df = pd.concat([final_df, fill_samples], ignore_index=True)

    if verbose:
        print(f"Total images sampled: {len(final_df)}")
        print("--- Tier Statistics ---")
        for name, s in stats.items():
            # Add already sampled count for accurate reporting
            already_s = already_sampled_tiers.get(name, 0)
            total_s = s["sampled"] + already_s
            print(f"  {name}: Sampled {total_s} (from {s['available']} available)")

    count_tags(
        final_df,
        normalized_upsampled_tags=None,
        output_path=os.path.join(reports_dir, "tag_counts_sampled_ids.csv"),
    )

    # Save sampled IDs to CSV for tracking
    final_df[
        [
            "id",
            "quality_tier",
            "rating",
            "tag_string",
            "file_url",
            "large_file_url",
            "image_width",
            "image_height",
        ]
    ].to_csv(output_csv, index=False)
    if verbose:
        print(f"\nSaved {len(final_df)} sampled IDs to '{output_csv}'")

    return final_df, stats


def sample_face_dataset(
    df: pd.DataFrame, output_csv: str, verbose: bool = True
) -> pd.DataFrame:
    """
    Filters the dataset specifically for anime faces.
    Enforces 1girl/1boy, solo, and removes bad meta tags.
    """
    if verbose:
        print(f"Original DataFrame size: {len(df)}")

    # 2. Tag count character <= 1
    if "tag_count_character" not in df.columns:
        df["tag_count_character"] = (
            df["tag_string_character"]
            .fillna("")
            .str.split(" ")
            .apply(lambda tags: len(tags) if tags != [""] else 0)
        )
    df = df[df["tag_count_character"] <= 1]

    # 3. 1girl or 1boy, and solo
    mask_solo = df["tag_string_general"].str.contains(r"\bsolo\b", regex=True, na=False)
    mask_gender = df["tag_string_general"].str.contains(
        r"\b(1girl|1boy)\b", regex=True, na=False
    )
    df = df[mask_solo & mask_gender]

    # 4. Meta tags exclusion
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

    # 5. Resolution check (at least 256x256 to allow proper cropping)
    if "image_width" in df.columns and "image_height" in df.columns:
        df = df[(df["image_width"] >= 256) & (df["image_height"] >= 256)]

    if verbose:
        print(f"Filtered face dataset size: {len(df)}")

    df.to_csv(output_csv, index=False)
    return df
