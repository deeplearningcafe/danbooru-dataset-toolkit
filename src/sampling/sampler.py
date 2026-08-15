import os
import re
import duckdb
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from ..prompts.prompt_utils import (
    get_tags_from_file,
    count_tags,
)
from .imputation import impute_missing_metrics
from .filters import apply_skip_tags
from .tiering import create_quality_tiers
from ..utils.loader import _read_dataset_duckdb


def _sample_from_tier(
    tier_df: pd.DataFrame,
    target_count: int,
    ratings_percentage: Dict[str, float],
    include_tags: Optional[Dict[str, float]],
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """Performs weighted sampling on a single quality tier DataFrame."""
    if tier_df.empty or target_count == 0:
        return pd.DataFrame(columns=tier_df.columns)

    work_df = tier_df.copy()

    if work_df.empty:
        return pd.DataFrame(columns=work_df.columns)

    if include_tags:
        tag_cols = [
            "tag_string_character",
            "tag_string_copyright",
            "tag_string_artist",
            "tag_string",
        ]
        avail_cols = [c for c in tag_cols if c in work_df.columns]

        def _calc_weight(row: pd.Series) -> float:
            row_tags = set()
            for col in avail_cols:
                if isinstance(row[col], str):
                    row_tags.update(row[col].split())
            if not row_tags:
                return 1.0
            w = 1.0
            for tag in row_tags:
                if tag in include_tags:
                    w *= include_tags[tag]
            return w

        work_df["sampling_weight"] = work_df.apply(_calc_weight, axis=1)
    else:
        work_df["sampling_weight"] = 1.0

    rating_targets = {r: int(target_count * p) for r, p in ratings_percentage.items()}
    remainder = target_count - sum(rating_targets.values())
    for i in range(remainder):
        r_key = list(rating_targets.keys())[i % len(rating_targets)]
        rating_targets[r_key] += 1

    sampled_dfs = []
    for rating, n_samples in rating_targets.items():
        if n_samples == 0:
            continue
        group = work_df[work_df["rating"] == rating]
        if group.empty:
            continue
        sampled_dfs.append(
            group.sample(
                n=min(n_samples, len(group)),
                weights="sampling_weight",
                random_state=rng,
                replace=False,
            )
        )

    # If the first pass didn't meet target_count, this pass fills the
    # remainder from any available images, ignoring rating distribution.
    if sampled_dfs:
        first_pass = pd.concat(sampled_dfs)
    else:
        first_pass = pd.DataFrame(columns=work_df.columns)

    needed = target_count - len(first_pass)
    if needed > 0:
        leftover = work_df.drop(first_pass.index) if not first_pass.empty else work_df
        if not leftover.empty:
            fill = leftover.sample(
                n=min(needed, len(leftover)),
                weights="sampling_weight",
                random_state=rng,
                replace=False,
            )
            final_samples = pd.concat([first_pass, fill])
        else:
            final_samples = first_pass
    else:
        final_samples = first_pass

    if final_samples.empty:
        return pd.DataFrame(columns=work_df.columns)

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
    exclude_path: Optional[str] = None,
    is_lora: bool = False,
    random_seed: int = 42,
    output_csv: str = "sampled_ids.csv",
    reports_dir: str = "reports",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """Filters and samples dataset based on quality tiers and ratings.

    Uses DuckDB for fast reading, hierarchical imputation for missing metrics,
    and protects artist samples from skip tags and worse_score tiering.
    """
    if verbose:
        print(f"Sampling {total_samples} samples from dataset.")
    np.random.seed(random_seed)
    rng = np.random.RandomState(random_seed)

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

    prior_samples = pd.DataFrame()
    if prior_knowledge_path and os.path.exists(prior_knowledge_path):
        if verbose:
            print(f"\nStep 0: Loading prior knowledge from '{prior_knowledge_path}'...")

        try:
            prior_samples = _read_dataset_duckdb(prior_knowledge_path)
            prior_samples = prior_samples.drop_duplicates().reset_index(drop=True)

            if aes_scores_csv_path and os.path.exists(aes_scores_csv_path):
                aes_df = _read_dataset_duckdb(aes_scores_csv_path)
                aes_df = aes_df.rename(columns={"score": "aes_score"})
                prior_samples = pd.merge(
                    prior_samples,
                    aes_df[["id", "aes_score"]],
                    on="id",
                    how="left",
                )

            prior_samples = impute_missing_metrics(prior_samples, artist_tags)

        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load prior knowledge: {e}")
            prior_samples = pd.DataFrame()

    df = impute_missing_metrics(df, artist_tags)

    single_characters = is_lora
    has_char_filter = bool(character_list)
    has_artist_filter = bool(artist_list)

    if has_char_filter or has_artist_filter:
        if verbose:
            print("\n--- Filtering for specific targets ---")
            if has_char_filter:
                print(f"  Characters: {character_list}")
            if has_artist_filter:
                print(f"  Artists: {artist_list}")

        def apply_target_filters(temp_df: pd.DataFrame) -> pd.DataFrame:
            if temp_df.empty:
                return temp_df
            final_mask = pd.Series(False, index=temp_df.index)

            if has_char_filter and "tag_string_character" in temp_df.columns:
                escaped = [re.escape(t) for t in character_list]
                pat = r"(?:^|\s)(" + "|".join(escaped) + r")(?:$|\s)"
                final_mask |= temp_df["tag_string_character"].str.contains(
                    pat, regex=True, na=False
                )

            if has_artist_filter and "tag_string_artist" in temp_df.columns:
                escaped = [re.escape(t) for t in artist_list]
                pat = r"(?:^|\s)(" + "|".join(escaped) + r")(?:$|\s)"
                final_mask |= temp_df["tag_string_artist"].str.contains(
                    pat, regex=True, na=False
                )

            return temp_df[final_mask]

        # Filter Main DataFrame
        before_len = len(df)
        if not prior_samples.empty:
            prior_samples = apply_target_filters(prior_samples)
        df = apply_target_filters(df)
        if verbose:
            print(f"  - Main Dataset filtered from {before_len} to {len(df)} samples.")

    prior_ids = set(prior_samples["id"]) if not prior_samples.empty else set()
    exclude_df = None
    if exclude_path and os.path.exists(exclude_path):
        exclude_df = _read_dataset_duckdb(exclude_path)

    exclude_ids = (
        set(exclude_df["id"])
        if exclude_df is not None and "id" in exclude_df
        else set()
    )
    ids_to_filter = prior_ids.union(exclude_ids)
    df = df[~df["id"].isin(ids_to_filter)]
    if verbose:
        print(f"Main dataframe size after filtering: {len(df)}")

    if not prior_samples.empty:
        mp_p, go_p, ba_p, wo_p = create_quality_tiers(
            prior_samples,
            single_characters=single_characters,
            reports_dir=reports_dir,
            artist_tags=artist_tags,
            verbose=verbose,
        )

        # Initialize quality_tier from quality_label for all prior tiers
        for p_df in (mp_p, go_p, ba_p, wo_p):
            p_df["quality_tier"] = p_df["quality_label"]

        if artist_tags and "tag_string_artist" in mp_p.columns:
            escaped = [re.escape(t) for t in artist_tags]
            a_pat = r"(?:^|\s)(" + "|".join(escaped) + r")(?:$|\s)"

            mp_a_mask = mp_p["tag_string_artist"].str.contains(
                a_pat, regex=True, na=False
            )
            mp_prior_art = mp_p[mp_a_mask].copy()
            mp_prior_chr = mp_p[~mp_a_mask].copy()

            go_a_mask = go_p["tag_string_artist"].str.contains(
                a_pat, regex=True, na=False
            )
            go_prior_art = go_p[go_a_mask].copy()
            go_prior_chr = go_p[~go_a_mask].copy()

            ba_a_mask = ba_p["tag_string_artist"].str.contains(
                a_pat, regex=True, na=False
            )
            ba_prior_art = ba_p[ba_a_mask].copy()
            ba_prior_chr = ba_p[~ba_a_mask].copy()

            wo_a_mask = wo_p["tag_string_artist"].str.contains(
                a_pat, regex=True, na=False
            )
            wo_prior_art = wo_p[wo_a_mask].copy()
            wo_prior_chr = wo_p[~wo_a_mask].copy()
            num_to_upgrade = len(wo_prior_art)

            if not wo_prior_art.empty:
                wo_prior_art["quality_label"] = "bad_score"
                wo_prior_art["quality_tier"] = "bad_score"
                ba_prior_art = pd.concat(
                    [ba_prior_art, wo_prior_art], ignore_index=True
                )
            if verbose:
                print(
                    f"  - Upgraded {num_to_upgrade} artist samples "
                    "from 'worse_score' to 'bad_score'."
                )

            if verbose:
                mp_a, mp_c = len(mp_prior_art), len(mp_prior_chr)
                go_a, go_c = len(go_prior_art), len(go_prior_chr)
                ba_a, ba_c = len(ba_prior_art), len(ba_prior_chr)
                wo_c = len(wo_prior_chr)
                print(f"    Masterpiece: {mp_a} artists, {mp_c} chars")
                print(f"    Good Score:  {go_a} artists, {go_c} chars")
                print(f"    Bad Score:   {ba_a} artists, {ba_c} chars")
                print(f"    Worse Score: {wo_c} chars")

        else:
            mp_prior_art = go_prior_art = ba_prior_art = pd.DataFrame(
                columns=mp_p.columns
            )
            mp_prior_chr = mp_p.copy()
            go_prior_chr = go_p.copy()
            ba_prior_chr = ba_p.copy()
            wo_prior_chr = wo_p.copy()

        already_sampled_tiers = {
            "masterpiece": len(mp_prior_art),
            "good_score": len(go_prior_art),
            "bad_score": len(ba_prior_art),
            "worse_score": 0,
        }
        all_sampled_dfs.extend([mp_prior_art, go_prior_art, ba_prior_art])

    else:
        already_sampled_tiers = {
            "masterpiece": 0,
            "good_score": 0,
            "bad_score": 0,
            "worse_score": 0,
        }
        mp_prior_chr = go_prior_chr = ba_prior_chr = wo_prior_chr = pd.DataFrame()

    num_prior = sum(already_sampled_tiers.values())
    if num_prior >= total_samples:
        final_df = pd.concat(all_sampled_dfs, ignore_index=True).sample(
            n=total_samples, random_state=rng
        )
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

    rem_samples = total_samples - num_prior
    char_samples_count = sum(
        [len(mp_prior_chr), len(go_prior_chr), len(ba_prior_chr), len(wo_prior_chr)]
    )
    non_prior = rem_samples - char_samples_count

    if non_prior > 0:
        mp_df, go_df, ba_df, wo_df = create_quality_tiers(
            df,
            single_characters=single_characters,
            reports_dir=reports_dir,
            artist_tags=artist_tags,
            verbose=verbose,
        )
        already_sampled_tiers["masterpiece"] += len(mp_prior_chr)
        already_sampled_tiers["good_score"] += len(go_prior_chr)
        already_sampled_tiers["bad_score"] += len(ba_prior_chr)
        already_sampled_tiers["worse_score"] += len(wo_prior_chr)
        all_sampled_dfs.extend([mp_prior_chr, go_prior_chr, ba_prior_chr, wo_prior_chr])
    else:
        mp_df, go_df, ba_df, wo_df = (
            mp_prior_chr,
            go_prior_chr,
            ba_prior_chr,
            wo_prior_chr,
        )

    quality_tier_dfs = {
        "masterpiece": mp_df,
        "good_score": go_df,
        "bad_score": ba_df,
        "worse_score": wo_df,
    }

    global_sampled_parents = set()
    for tier_name, tier_df in quality_tier_dfs.items():
        if tier_df.empty:
            continue
        work_df = tier_df[~tier_df["id"].isin(exclude_ids)].copy()
        if skip_tags:
            work_df = apply_skip_tags(work_df, skip_tags, rng, 1.0, artist_tags)
        if "parent_id" in work_df.columns:
            work_df["parent_group"] = work_df["parent_id"].fillna(work_df["id"])
        quality_tier_dfs[tier_name] = work_df

    tier_order = ["masterpiece", "good_score", "bad_score", "worse_score"]
    tier_targets = {n: int(total_samples * p) for n, p in quality_percentages.items()}
    for t_name in tier_order:
        if t_name in tier_targets:
            adj = already_sampled_tiers.get(t_name, 0)
            tier_targets[t_name] = max(0, tier_targets[t_name] - adj)

    deficit = 0
    for tier_name in tier_order:
        curr_target = tier_targets.get(tier_name, 0) + deficit
        tier_df = quality_tier_dfs.get(tier_name)

        if tier_df is None or tier_df.empty or curr_target <= 0:
            stats[tier_name] = {
                "available": len(tier_df) if tier_df is not None else 0,
                "target": curr_target,
                "sampled": 0,
            }
            deficit = curr_target
            continue

        if "parent_group" in tier_df.columns:
            tier_df = tier_df[~tier_df["parent_group"].isin(global_sampled_parents)]

        tier_samples = _sample_from_tier(
            tier_df, curr_target, ratings_percentage, local_include_tags, rng
        )
        num_sampled = len(tier_samples)
        deficit = curr_target - num_sampled

        if not tier_samples.empty:
            if "parent_group" in tier_samples.columns:
                global_sampled_parents.update(tier_samples["parent_group"].unique())
            tier_samples["quality_tier"] = tier_name
            all_sampled_dfs.append(tier_samples)

        stats[tier_name] = {
            "available": len(tier_df),
            "target": curr_target,
            "sampled": num_sampled,
        }

    if not all_sampled_dfs:
        return pd.DataFrame(), stats

    final_df = pd.concat(all_sampled_dfs, ignore_index=True)
    rem_needed = total_samples - len(final_df)
    if rem_needed > 0:
        already_ids = set(final_df["id"])
        leftovers = []
        for tier_df in quality_tier_dfs.values():
            if not tier_df.empty and "parent_group" in tier_df.columns:
                unsampled = tier_df[
                    ~tier_df["parent_group"].isin(global_sampled_parents)
                ]
                if not unsampled.empty:
                    leftovers.append(unsampled)
        if leftovers:
            pool = pd.concat(leftovers).drop_duplicates(subset=["id"])
            pool = pool[~pool["id"].isin(already_ids)]
            if not pool.empty:
                fill = pool.sample(n=min(rem_needed, len(pool)), random_state=rng)
                fill["quality_tier"] = fill["quality_label"]
                final_df = pd.concat([final_df, fill], ignore_index=True)

    count_tags(
        final_df,
        normalized_upsampled_tags=None,
        output_path=os.path.join(reports_dir, "tag_counts_sampled_ids.csv"),
    )

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
