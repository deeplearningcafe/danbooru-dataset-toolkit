import pandas as pd
import os
import re
from tqdm import tqdm
import json
from transformers import CLIPTokenizer
from ..utils.loader import (
    load_all_parquets,
    parallel_scan_images,
    append_weight_to_json,
)
from ..prompts.prompt_utils import (
    format_danbooru_tag_inverse,
    format_danbooru_tag,
    normalize_tag_string,
    format_tag_string,
    extract_and_remove_person_tags,
    META_TAGS_TO_INCLUDE,
    RATINGS,
    QUALITY_TIER_AES,
    count_tags,
)
from ..prompts.shuffle_prompts import split_upsampled_tags
from pathlib import Path
from ..prompts.tag_weighter import TagWeighter
import random


class PromptGenerator:
    """
    Handles the final data merging, tier refinement, tag pruning,
    and prompt file generation.
    """

    def __init__(self, config: dict):
        self.config = config
        self.tokenizer = CLIPTokenizer.from_pretrained(config["tokenizer_path"])
        self.image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

    def _get_metadata_tag(self, row):
        """
        Determines the metadata tag based on image dimensions.
        """
        width = row["image_width"]
        height = row["image_height"]
        tags = []
        if width * height <= 500 * 500:
            return "lowres"
        elif width * height >= 1600 * 1200:
            tags.append("highres")
        elif width * height >= 3200 * 2400:
            tags.append("absurdres")
        elif width > 10000 or height > 10000:
            tags.append("incredibly_absurdres")
        if tags:
            return ", ".join(tags)
        return ""

    def _format_year(self, year):
        """
        Determines the metadata tag based on image dimensions.
        """
        choice = random.choice([0, 1])
        year_str = "newest"
        num_year = int(year[-2:])
        if num_year <= 17:
            year_str = "oldest"
        elif 17 < num_year <= 19:
            year_str = "old"
        elif 19 < num_year <= 20:
            year_str = "modern"
        elif 20 < num_year <= 22:
            year_str = "recent"

        # include 50% times the year to improve generalization
        if choice:
            year_str += f", {year}"
        return year_str

    def load_booru_df(
        self,
        parquet_path: str,
        classifier_labels: list[str],
        sampled_ids_csv: str = None,
        upsampled_prompts_csv: str = None,
        prior_knowledge_path: str = None,
        num_parquets: int = 100,
    ) -> pd.DataFrame:
        # --- Step 1: Load all necessary data ---
        print("Loading all data sources...")
        # Load the full original dataframe to get all tag strings
        full_df = load_all_parquets(
            parquet_path,
            True if prior_knowledge_path else False,
            num_parquets=num_parquets,
        )
        columns_to_keep = [
            "id",
            "created_at",
            "relative_path",
            "tag_string_general",
            "tag_string_character",
            "tag_string_copyright",
            "rating",
            "tag_string_artist",
            "final_tier",
            "quality_tier",
            "image_width",
            "image_height",
            "tag_string_meta",
        ]
        existing_columns = [col for col in columns_to_keep if col in full_df.columns]
        full_df = full_df[existing_columns]

        if prior_knowledge_path:
            prior_knowledge_samples = pd.read_csv(
                prior_knowledge_path, header=0, low_memory=True
            )
            prior_knowledge_samples = (
                prior_knowledge_samples.drop_duplicates().reset_index(drop=True)
            )
            existing_columns = [
                col for col in columns_to_keep if col in prior_knowledge_samples.columns
            ]
            prior_knowledge_samples = prior_knowledge_samples[existing_columns]
            if "tag_string_meta" not in prior_knowledge_samples.columns:
                if "tag_string_meta" in full_df.columns:
                    full_df = full_df.drop(columns=["tag_string_meta"])

                full_df["tag_string_meta"] = full_df.apply(
                    self._get_metadata_tag, axis=1
                )
                prior_knowledge_samples["tag_string_meta"] = (
                    prior_knowledge_samples.apply(self._get_metadata_tag, axis=1)
                )
            if "fav_count" in prior_knowledge_samples.columns:
                nan_count = prior_knowledge_samples["fav_count"].isna().sum()
                if nan_count > 0:
                    mean_aes = prior_knowledge_samples["fav_count"].mean()
                    prior_knowledge_samples["fav_count"] = prior_knowledge_samples[
                        "fav_count"
                    ].fillna(mean_aes)

                    print(
                        f"  - Filled {nan_count} missing 'fav_count' "
                        f"values with the mean ({mean_aes:.4f})."
                    )

            full_df = (
                pd.concat([full_df, prior_knowledge_samples])
                .drop_duplicates(subset=["id"], keep="last")
                .reset_index(drop=True)
            )

        # Load the new aesthetic labels from the classifier
        # only images classified will be used
        classifier_labels_df_list = []
        for cls_path in classifier_labels:
            if cls_path.endswith(".csv"):
                classifier_labels_df = pd.read_csv(cls_path)
                classifier_labels_df_list.append(classifier_labels_df)
            elif cls_path.endswith(".json"):
                with open(cls_path, "r") as f:
                    json_data = json.load(f)
                classifier_labels_df = pd.DataFrame(
                    list(json_data.items()),
                    columns=["relative_path", "aesthetic_label"],
                )
                label_to_class = {
                    0: "worse_score",
                    1: "bad_score",
                    2: "good_score",
                    3: "masterpiece",
                }
                # idx2label = {0:"worst", 1:"worse", 2:"better", 3:"best"} classifier format
                classifier_labels_df["aesthetic_label"] = classifier_labels_df[
                    "aesthetic_label"
                ].apply(lambda label: label_to_class[int(label)])
                classifier_labels_df_list.append(classifier_labels_df)
            else:
                raise ValueError(
                    f"Unsupported file format: {classifier_labels}. Only CSV and JSON are supported."
                )
        classifier_labels_df = pd.concat(classifier_labels_df_list, axis=0)

        # --- Step 2: Prepare the final DataFrame for processing ---
        print("Merging data sources...")
        # Extract ID from the classifier's relative_path
        classifier_labels_df["id"] = classifier_labels_df["relative_path"].apply(
            lambda p: int(
                os.path.splitext(os.path.basename(p.replace("\\", os.sep)))[0]
            )
        )
        classifier_labels_df = classifier_labels_df.rename(
            columns={"aesthetic_label": "new_aesthetic_label"}
        )[["id", "new_aesthetic_label"]]

        if sampled_ids_csv is not None:
            # Load the sampled IDs and their initial 'quality_tier'
            sampled_df = pd.read_csv(sampled_ids_csv)
            sampled_df.dropna(subset=["relative_path"], inplace=True)
            print(f"Len processed_df {len(sampled_df)} after droping rel_path nans")

            # Merge the sampled data with the new classifier labels
            # This gives us a dataframe with id, quality_tier, and new_aesthetic_label
            processed_df = pd.merge(
                sampled_df, classifier_labels_df, on="id", how="left"
            )
        else:
            # infer the quality_tier by the relative path
            # classifier_labels_df['quality_tier'] = classifier_labels_df['relative_path'].apply(
            #     lambda p: os.path.dirname(p.replace('\\', os.sep)).split(os.sep)[-1]
            # )
            classifier_labels_df["quality_tier"] = classifier_labels_df[
                "new_aesthetic_label"
            ]

            processed_df = classifier_labels_df
        print(f"Len processed_df {len(processed_df)}")

        columns_to_include = [
            "id",
            "quality_tier",
            "new_aesthetic_label",
            "relative_path",
        ]
        if upsampled_prompts_csv is not None and len(upsampled_prompts_csv) > 0:
            # Load the new aesthetic labels from the classifier
            upsampled_prompts_df = pd.read_csv(upsampled_prompts_csv)
            upsampled_prompts_df["upsampled_tags"] = upsampled_prompts_df[
                "upsampled_tags"
            ].fillna("")

            processed_df = pd.merge(
                processed_df, upsampled_prompts_df, on="id", how="left"
            )
            columns_to_include.append("upsampled_tags")

        # Now, merge with the *original full dataframe* to get all the detailed
        # tag strings needed for prompt construction.
        # We select only the rows that were in our sample.

        final_df = pd.merge(
            full_df, processed_df[columns_to_include], on="id", how="inner"
        )

        # the samples with nans in the new_aesthetic_label should be removed as they can't be downloaded
        final_df.dropna(subset=["new_aesthetic_label"], inplace=True)

        print(f"Loaded {len(final_df)} samples for final processing.")
        return final_df

    def load_booru_df_sampled(
        self,
        parquet_path: str,
        sampled_ids_csv: str = None,
        prior_knowledge_path: str = None,
        num_parquets: int = 100,
    ) -> pd.DataFrame:
        # --- Step 1: Load all necessary data ---
        print("Loading all data sources...")
        # Load the full original dataframe to get all tag strings
        full_df = load_all_parquets(
            parquet_path,
            True if prior_knowledge_path else False,
            num_parquets=num_parquets,
        )
        columns_to_keep = [
            "id",
            "created_at",
            "relative_path",
            "tag_string_general",
            "tag_string_character",
            "tag_string_copyright",
            "rating",
            "tag_string_artist",
            "final_tier",
            "quality_tier",
            "image_width",
            "image_height",
            "tag_string_meta",
        ]
        existing_columns = [col for col in columns_to_keep if col in full_df.columns]
        full_df = full_df[existing_columns]

        if prior_knowledge_path:
            prior_knowledge_samples = pd.read_csv(
                prior_knowledge_path, header=0, low_memory=False
            )
            existing_columns = [
                col for col in columns_to_keep if col in prior_knowledge_samples.columns
            ]
            prior_knowledge_samples = prior_knowledge_samples[existing_columns]
            if "tag_string_meta" not in prior_knowledge_samples.columns:
                if "tag_string_meta" in full_df.columns:
                    full_df = full_df.drop(columns=["tag_string_meta"])

                full_df["tag_string_meta"] = full_df.apply(
                    self._get_metadata_tag, axis=1
                )
                prior_knowledge_samples["tag_string_meta"] = (
                    prior_knowledge_samples.apply(self._get_metadata_tag, axis=1)
                )
            if "fav_count" in prior_knowledge_samples.columns:
                nan_count = prior_knowledge_samples["fav_count"].isna().sum()
                if nan_count > 0:
                    mean_aes = prior_knowledge_samples["fav_count"].mean()
                    prior_knowledge_samples["fav_count"] = prior_knowledge_samples[
                        "fav_count"
                    ].fillna(mean_aes)

                    print(
                        f"  - Filled {nan_count} missing 'fav_count' "
                        f"values with the mean ({mean_aes:.4f})."
                    )

            full_df = (
                pd.concat([full_df, prior_knowledge_samples])
                .drop_duplicates(subset=["id"], keep="last")
                .reset_index(drop=True)
            )

        processed_df = pd.read_csv(sampled_ids_csv)
        print(f"Len processed_df {len(processed_df)}")
        processed_df.dropna(subset=["relative_path"], inplace=True)
        print(f"Len processed_df {len(processed_df)} after droping rel_path nans")

        columns_to_include = ["id", "quality_tier", "relative_path"]

        final_df = pd.merge(
            full_df, processed_df[columns_to_include], on="id", how="inner"
        )

        final_df.dropna(subset=["quality_tier"], inplace=True)
        final_df["final_tier"] = final_df["quality_tier"]

        print(f"Loaded {len(final_df)} samples for final processing.")
        return final_df

    def refine_tiers_and_assign_final_class(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Refines quality tiers using a new classifier's labels.

        This function implements a "Veto and Rescue" system. It starts with
        the 'quality_tier' from the initial sampling and creates a new
        'final_tier' column. It then uses the 'new_aesthetic_label' to
        correct potential misclassifications.

        - Veto (Demotion): High-quality images ('masterpiece', 'good_score')
        are demoted to 'worse_score' if the new classifier labels them 'worst'.
        - Rescue (Promotion): Low-quality images ('bad_score', 'worse_score')
        are promoted to 'good_score' if the new classifier labels them 'best'.

        Args:
            df (pd.DataFrame): DataFrame containing 'quality_tier' and
                            'new_aesthetic_label' columns.

        Returns:
            pd.DataFrame: The DataFrame with a new 'final_tier' column.
        """
        print("Refining quality tiers with new classifier labels...")
        if "quality_tier" not in df.columns or "new_aesthetic_label" not in df.columns:
            raise ValueError(
                "Input DataFrame must contain 'quality_tier' and "
                "'new_aesthetic_label' columns."
            )

        # Initialize the final tier with the original tier
        df["final_tier"] = df["quality_tier"]

        # Rule 1: Veto (Demotion)
        demotion_mask = df["quality_tier"].isin(
            ["masterpiece", "good_score", "bad_score"]
        ) & (df["new_aesthetic_label"] == "worst")
        num_demoted = demotion_mask.sum()
        if num_demoted > 0:
            df.loc[demotion_mask, "final_tier"] = "worse_score"
            print(f"  - Vetoed and demoted {num_demoted} images to 'worse_score'.")

        demotion_mask = df["quality_tier"].isin(
            [
                "masterpiece",
            ]
        ) & (df["new_aesthetic_label"] == "worse")
        num_demoted = demotion_mask.sum()
        if num_demoted > 0:
            df.loc[demotion_mask, "final_tier"] = "bad_score"
            print(f"  - Vetoed and demoted {num_demoted} images to 'bad_score'.")

        # Rule 2: Rescue (Promotion)
        # as the num masterpiece is too low, we need to rescue more
        promotion_mask = df["quality_tier"].isin(["bad_score", "good_score"]) & (
            df["new_aesthetic_label"] == "best"
        )
        num_promoted = promotion_mask.sum()
        if num_promoted > 0:
            df.loc[promotion_mask, "final_tier"] = "masterpiece"
            print(f"  - Rescued and promoted {num_promoted} images to 'masterpiece'.")

        promotion_mask = df["quality_tier"].isin(["worse_score"]) & (
            df["new_aesthetic_label"] == "better"
        )
        num_promoted = promotion_mask.sum()
        if num_promoted > 0:
            df.loc[promotion_mask, "final_tier"] = "good_score"
            print(f"  - Rescued and promoted {num_promoted} images to 'good_score'.")

        print("Tier refinement complete.")
        try:
            tiers_df = df[["id", "final_tier"]]
            tiers_df.to_csv(self.config["final_tiers_csv"], index=False)
        except IOError as e:
            print(
                f"Error: Could not write to file {self.config['final_tiers_csv']}. Reason: {e}"
            )

        return df

    # TODO: read all the datasets txt prompts and count the whole dataset
    def prune_and_report_tags(
        self,
        df: pd.DataFrame,
        tags_csv_path: str,
        min_general_count: int = 100,
        min_artist_count: int = 20,
    ) -> pd.DataFrame:
        """
        Prunes tags using efficient, vectorized operations and reports counts.

        This function is optimized to handle large datasets by avoiding
        row-by-row .apply() calls in favor of vectorized string methods.

        Args:
            df (pd.DataFrame): The dataframe with tag columns.
            tags_csv_path (str): Path to the CSV with allowed tags by category.
            artists_txt_path (str): Path to the TXT file with famous artists.
            min_general_count (int): Min count for general/character tags.
            min_artist_count (int): Min count for an artist if not famous.

        Returns:
            pd.DataFrame: The DataFrame with pruned tag columns.
        """
        print("Starting optimized tag pruning and reporting process...")

        # --- 1. Load external resources into sets for fast lookups ---
        tags_df = pd.read_csv(tags_csv_path)
        allowed_characters = set(tags_df[tags_df["category"] == 4]["name"])
        with open(self.config.get("artists_txt"), "r", encoding="utf-8") as f:
            famous_artists = {
                format_danbooru_tag_inverse(line) for line in f if line.strip()
            }

        # --- 2. Perform highly efficient, vectorized tag counting ---
        print("Counting tag occurrences with vectorized operations...")

        # **NEW**: Normalize 'upsampled_tags' to match the standard format
        # of 'tag_string_general' (space-separated with underscores) before
        # counting to ensure data consistency.
        print("Normalizing upsampled tags for consistent counting...")
        normalized_upsampled_tags = df["upsampled_tags"].apply(normalize_tag_string)

        def get_counts(series: pd.Series) -> pd.Series:
            """Uses explode and value_counts for fast counting."""
            return series.dropna().str.split(" ").explode().value_counts()

        # Combine general and upsampled tags for a unified count
        general_tags_series = pd.concat(
            [df["tag_string_general"], normalized_upsampled_tags]
        )
        general_counts = get_counts(general_tags_series)
        character_counts = get_counts(df["tag_string_character"])
        artist_counts = get_counts(df["tag_string_artist"])

        # --- 3. Generate and save the tag count report ---
        reports_dir = self.config.get("reports_dir", "reports")
        report_path = os.path.join(reports_dir, "tag_counts_report.csv")

        print("Saving tag counts to 'tag_counts_report.csv'...")
        report_dfs = []
        for counts, name in [
            (general_counts, "general"),
            (character_counts, "character"),
            (artist_counts, "artist"),
        ]:
            report_df = counts.reset_index()
            report_df.columns = ["tag", "count"]
            report_df["category"] = name
            report_dfs.append(report_df)

        full_report = pd.concat(report_dfs, ignore_index=True)
        full_report[["category", "tag", "count"]].to_csv(report_path, index=False)
        print(f"Saved {len(full_report)} unique tag counts to report.")

        # --- 4. Determine which tags to REMOVE for vectorized filtering ---
        # A tag is removed if it fails to meet the inclusion criteria.

        # General tags are removed if their count is too low.
        remove_general = set(general_counts[general_counts < min_general_count].index)

        # Characters are removed if count is too low OR not in the allowed list.
        remove_character = set(
            character_counts[character_counts < min_artist_count].index
        ) | (set(character_counts.index) - allowed_characters)

        # Artists are removed if they are NOT famous AND their count is too low.
        remove_artist = {
            tag
            for tag, count in artist_counts.items()
            if tag not in famous_artists and count < min_artist_count
        }

        # --- 5. Filter columns using a single, fast, vectorized regex ---
        print("Applying vectorized filters to DataFrame...")

        def build_removal_regex(tags_to_remove: set) -> str:
            """Builds a regex to find and remove specific whole tags."""
            if not tags_to_remove:
                return ""
            # Escape special characters in tags and join with '|' (OR)
            # \b ensures we match whole words only (e.g., 'cat' not 'catgirl')
            return r"\b(" + "|".join(re.escape(t) for t in tags_to_remove) + r")\b"

        # **MODIFIED**: The 'upsampled_tags' column is now also filtered.
        # We must re-normalize the tags to be removed into the comma-separated
        # format before applying the regex to the upsampled_tags column.
        upsampled_remove_set = {format_danbooru_tag(tag) for tag in remove_general}
        upsampled_regex = build_removal_regex(upsampled_remove_set)

        for col, removal_set in [
            ("tag_string_general", remove_general),
            ("tag_string_character", remove_character),
            ("tag_string_artist", remove_artist),
        ]:
            if col not in df.columns or not removal_set:
                continue

            print(f"Filtering {len(removal_set)} tags from '{col}'...")
            regex_pattern = build_removal_regex(removal_set)

            # Vectorized removal of all targeted tags at once
            df[col] = df[col].str.replace(regex_pattern, "", regex=True)
            # Clean up leftover double spaces from the removal
            df[col] = df[col].str.replace(r"\s+", " ", regex=True).str.strip()

        # Filter the upsampled_tags column using its specific format
        if "upsampled_tags" in df.columns and upsampled_regex:
            print(f"Filtering tags from 'upsampled_tags'...")
            df["upsampled_tags"] = (
                df["upsampled_tags"]
                .str.replace(upsampled_regex, "", regex=True)
                .str.replace(
                    r",\s*,",
                    ",",
                    regex=True,  # Clean up empty tags
                )
                .str.replace(
                    r"(^,\s*|\s*,$)",
                    "",
                    regex=True,  # Clean leading/trailing commas
                )
                .str.strip()
            )

        print("Tag pruning complete.")
        return df

    def count_and_weight_tags(
        self,
        df: pd.DataFrame,
        root_dir: str,
        metadata_path: str,
        num_workers: int = 4,
    ) -> None:
        """
        Counts tags from cached .txt files, computes a weight for each sample
        based on tag rarity and category using the provided TagWeighter, and
        appends this weight to each sample's .json metadata file.

        This function operates directly on the file system to handle large
        datasets efficiently. It now uses a left merge to ensure all tags
        from the dataset are included, even if they are not in tags_csv_path.

        Args:
            root_dir (str): The root directory where cached latents and
                            prompt files (.txt, .json) are stored.
            metadata_path (str): Path to the metadata.json file that lists
                                all valid samples for the current dataset.
            tags_csv_path (str): Path to the 'selected_tags.csv' file, which
                                maps tags to their categories.
            artists_txt_path (str): Path to a TXT file of prioritized artists,
                                    required by the TagWeighter.
        """
        print("Starting tag counting and sample weighting process...")
        root_path = Path(root_dir)

        # --- 1. Load valid sample IDs from metadata ---
        print(f"Loading metadata from '{metadata_path}'...")
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            valid_booru_ids = {
                str(item["booru_id"]) for item in metadata.get("sample_mapping")
            }
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Fatal: Could not load or parse metadata. {e}")
            return
        print(f"Found {len(valid_booru_ids)} valid samples in metadata.")

        # --- 2. Discover all .txt files and filter them ---
        print(f"Scanning for all prompt files in '{root_dir}'...")
        all_txt_files = parallel_scan_images(
            root_path, num_workers=num_workers, prompts=True
        )
        valid_txt_paths = {
            p.stem: p for p in all_txt_files if p.stem in valid_booru_ids
        }

        print(f"Found {len(valid_txt_paths)} matching .txt files.")
        if not valid_txt_paths:
            print("No prompt files found for the given metadata. Aborting.")
            return

        # --- 2. Perform highly efficient, vectorized tag counting ---
        print("Counting tag occurrences with vectorized operations...")

        # **NEW**: Normalize 'upsampled_tags' to match the standard format
        # of 'tag_string_general' (space-separated with underscores) before
        # counting to ensure data consistency.
        print("Normalizing upsampled tags for consistent counting...")
        normalized_upsampled_tags = df["upsampled_tags"].apply(normalize_tag_string)

        reports_dir = self.config.get("reports_dir", "reports")
        report_path = os.path.join(reports_dir, "tag_counts_report.csv")

        count_tags(
            df,
            normalized_upsampled_tags=normalized_upsampled_tags,
            output_path=report_path,
        )

        print("Initializing TagWeighter...")
        tag_weighter = TagWeighter(
            tag_counts_csv_path=report_path,
            artists_txt_path=self.config.get("artists_txt"),
            characters_txt_path=self.config.get("characters_txt"),
        )

        print("Computing and appending weights to JSON files...")
        for booru_id in tqdm(valid_txt_paths, desc="Appending weights"):
            txt_path = valid_txt_paths[booru_id]
            json_path = txt_path.with_suffix(".json")

            if not json_path.exists():
                continue

            try:
                caption = None
                # Attempt to read the .txt file with utf-8, but fall back
                # to utf-16 if a UnicodeDecodeError occurs. This handles
                # files that were incorrectly saved with a UTF-16 BOM (0xff).
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        caption = f.read()
                except UnicodeDecodeError:
                    # If utf-8 fails, try utf-16.
                    with open(txt_path, "r", encoding="utf-16") as f:
                        caption = f.read()
                weight = tag_weighter.get_caption_weight(caption)
                append_weight_to_json(str(json_path), weight)
            except Exception as e:
                print(
                    f"Warning: Failed to update {json_path} with txt file {txt_path}. Reason: {e}"
                )

        print("Finished appending tag weights to all valid JSON files.")

    def construct_prompt_string(self, row: pd.Series) -> str:
        """
        Constructs a structured prompt string from a DataFrame row.

        The format is:
        person count ||| character names ||| rating ||| general tags |||
        artist ||| score range based rating ||| year modifier

        Args:
            row (pd.Series): A row from the DataFrame.

        Returns:
            str: The formatted prompt string.
        """

        # 1. Person Count
        general_tags_raw = row.get("tag_string_general", "")
        # upsampled tags is already in formatted correctly with ","
        upsampled_tags_raw = str(row.get("upsampled_tags", ""))
        # general_tags_raw = general_tags_raw + " " + upsampled_tags_raw
        # This new function handles both finding all tags and cleaning the source
        person_count, remaining_general_tags = extract_and_remove_person_tags(
            general_tags_raw
        )

        # 2. Character Names
        characters = format_tag_string(row.get("tag_string_character", ""))
        copyright = format_tag_string(row.get("tag_string_copyright", ""))

        # 3. Rating
        rating = RATINGS[row.get("rating", "")]

        # 4. General Tags
        general_tags = format_tag_string(remaining_general_tags)
        # if upsampled_tags_raw contains solo can be a problem
        # only append them if the length is not too high or the prompt is truncated
        general_tags_upsampled = general_tags
        if len(upsampled_tags_raw) > 0:
            general_tags_upsampled = general_tags + ", " + upsampled_tags_raw

        # 5. Artist
        artist = format_tag_string(row.get("tag_string_artist", ""))

        # 6. Score Range Based Rating (Aesthetic Class + Meta Tags)
        aesthetic_class = format_tag_string(row.get("final_tier", "unknown_tier"))
        # we will keep the nai-v2 quality tags with 50% prob
        # as we won't shuffle the aes tags it is better to uncondition some times
        choice = random.choice([0, 1])
        if choice:
            nai_quality_tags = QUALITY_TIER_AES.get(aesthetic_class)
            aesthetic_class += ", " + random.choice(nai_quality_tags[:-1])
            # the last tag is the aesthetic
            aesthetic_class += ", " + nai_quality_tags[-1]

        meta_tags_raw = row.get("tag_string_meta", "")
        if pd.notna(meta_tags_raw):
            # Meta tags are also formatted for consistency.
            formatted_meta = format_tag_string(meta_tags_raw)

            # We only include the specific high-quality meta tags we care about.
            present_meta_tags = [
                tag
                for tag in META_TAGS_TO_INCLUDE
                if tag.replace("_", " ") in formatted_meta
            ]
            if present_meta_tags:
                aesthetic_class += ", " + ", ".join(present_meta_tags)

        # 7. Year Modifier
        year = f"year {row.get('year', '')}"
        year = self._format_year(year)

        if self.tokenizer is None:
            # Assemble the final prompt string
            prompt_parts = [
                str(person_count),
                str(characters),
                str(copyright),
                str(rating),
                str(general_tags_upsampled),
                str(artist),
                str(aesthetic_class),
                str(year),
            ]
            prompt_parts = [prompt for prompt in prompt_parts if len(prompt) > 0]
            return ", ".join(prompt_parts)

        prompt_prefix = [
            str(person_count),
            str(characters),
            str(copyright),
            str(rating),
        ]
        prompt_prefix = [part for part in prompt_prefix if len(part) > 0]
        lengths = []
        prompt_prefix_str = ", ".join(prompt_prefix)
        prompt_prefix_tokens = self.tokenizer(
            prompt_prefix_str + ", ",
            padding=False,  # Don't pad here, just count tokens
            truncation=False,  # Don't truncate as we want length
            return_length=True,
        )
        lengths.append(prompt_prefix_tokens["length"] - 2)
        # remove last token EOS
        prompt_prefix_tokens = prompt_prefix_tokens["input_ids"][:-1]

        prompt_suffix = [str(artist), str(aesthetic_class), str(year)]
        prompt_suffix = [part for part in prompt_suffix if len(part) > 0]
        prompt_suffix_str = ", ".join(prompt_suffix)
        prompt_suffix_tokens = self.tokenizer(
            prompt_suffix_str,
            padding=False,  # Don't pad here, just count tokens
            truncation=False,  # Don't truncate as we want length
            return_length=True,
        )
        lengths.append(prompt_suffix_tokens["length"] - 2)
        # remove the first token BOS
        prompt_suffix_tokens = prompt_suffix_tokens["input_ids"][1:]

        free_tokens = 225 - sum(lengths)

        general_tags_tokens = self.tokenizer(
            general_tags,
            padding=False,  # Don't pad here, just count tokens
            truncation=False,  # Don't truncate as we want length
            return_length=True,
        )
        # remove the 2 commas before and after the general tags
        general_tags_len = general_tags_tokens["length"] - 4

        free_tokens -= general_tags_len
        decoded_sliced_input_ids = ""
        if free_tokens > 0 and len(upsampled_tags_raw) > 0:
            decoded_sliced_input_ids, last_token = split_upsampled_tags(
                upsampled_tags_raw, free_tokens, self.tokenizer, return_last_token=True
            )

        general_tokens = self.tokenizer(
            ", ".join([general_tags, decoded_sliced_input_ids]),
            padding=False,  # Don't pad here, just count tokens
            truncation=False,  # Don't truncate as we want length
            return_length=True,
        )["input_ids"][1:-1]

        # 4. Assemble the final prompt using the defined separator
        all_parts = [
            prompt_prefix_str,
            general_tags,
            decoded_sliced_input_ids,
            prompt_suffix_str,
        ]
        all_parts = [prompt for prompt in all_parts if len(prompt) > 0]
        final_prompt = ", ".join(all_parts)

        return final_prompt, [
            prompt_prefix_tokens,
            general_tokens,
            prompt_suffix_tokens,
        ]

    def create_prompt_files(
        self,
        df: pd.DataFrame,
        output_dir: str,
        model_path: str = "model",
        create_json_files: bool = True,
        faces_mode: bool = False,
        character_list: list = None,
        artist_list: list = None,
    ):
        """
        Generates and saves a .txt prompt file for each row in the DataFrame.

        Args:
            df (pd.DataFrame): The final, processed DataFrame.
            output_dir (str): The directory to save the .txt files.
            model_path (str): Path to the models directory.
            create_json_files (bool): Whether to save tokenized JSON files.
            faces_mode (bool): If True, uses simple conditional prompts.
            character_list (list): Allowed characters from config.
            artist_list (list): Allowed artists from config.
        """
        print(f"Creating prompt files in '{output_dir}'...")
        os.makedirs(output_dir, exist_ok=True)

        # Pre-calculate year for efficiency
        if "created_at" in df.columns:
            df["year"] = (
                pd.to_datetime(df["created_at"], utc=True, errors="coerce")
                .dt.year.fillna("")
                .astype(str)
            )
        else:
            df["year"] = ""  # Ensure column exists

        # --- Masking Characters and Artists ---
        if character_list:
            allowed_chars = set(character_list)
            df["tag_string_character"] = df["tag_string_character"].apply(
                lambda x: " ".join([t for t in str(x).split() if t in allowed_chars])
                if pd.notna(x)
                else ""
            )

        if artist_list:
            allowed_artists = set(artist_list)
            df["tag_string_artist"] = df["tag_string_artist"].apply(
                lambda x: " ".join([t for t in str(x).split() if t in allowed_artists])
                if pd.notna(x)
                else ""
            )

        tokenizer_path = f"{model_path}tokenizer"
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)

        # Use tqdm for a progress bar, as this can be a slow I/O operation
        created_txts = 0
        for _, row in tqdm(
            df.iterrows(), total=df.shape[0], desc="Writing prompt files"
        ):
            prompt_text, tokens = self.construct_prompt_string(row)
            relative_path = os.path.splitext(
                row["relative_path"].replace("\\", os.sep)
            )[0]
            exists = False
            if os.path.exists(f"{output_dir}/{row['relative_path']}"):
                file_path = os.path.join(output_dir, f"{relative_path}.txt")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(prompt_text)

                if create_json_files:
                    base_filename = os.path.basename(relative_path)
                    # Split the filename from its extension to get the clean ID.
                    img_id = os.path.splitext(base_filename)[0]
                    data = {
                        "prefix_tokens": tokens[0],
                        "general_tokens": tokens[1],
                        "suffix_tokens": tokens[2],
                    }
                    file_path = os.path.join(output_dir, f"{relative_path}.json")
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)
                exists = True
                # continue

            if not exists:
                print(
                    f"The image with relative path {row['relative_path']} doesn't exists"
                )
            created_txts += int(exists)

        print(f"Created {created_txts} files")
