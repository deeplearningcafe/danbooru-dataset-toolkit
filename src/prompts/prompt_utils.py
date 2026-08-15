import pandas as pd
import re
from functools import reduce
from typing import List
import os

META_TAGS_TO_INCLUDE = {
    "highres",
    "absurdres",
    "ultra-detailed",
    "official_art",
    "novel_illustration",
    "official_wallpaper",
}
RATINGS = {"e": "explicit", "q": "questionable", "s": "sensitive", "g": "general"}
# It looks for a word boundary, one or more digits, 'boy' or 'girl',
# and an optional 's', all followed by another word boundary.
PERSON_COUNT_REGEX = re.compile(r"\b\d+\+?(?:boy|girl)s?\b")

QUALITY_TIER_AES = {
    "masterpiece": ["best quality", "very aesthetic"],
    "good score": ["amazing quality", "great quality", "aesthetic"],
    "bad score": ["normal quality", "bad quality", "displeasing"],
    "worse score": ["worst quality", "very displeasing"],
}

# --- Validation Constants ---
HAIR_LENGTH_TAGS = {
    "long hair",
    "medium hair",
    "short hair",
    "very long hair",
    "very short hair",
    "absurdly long hair",
}

EYE_COLOR_TAGS = {
    "blue eyes",
    "red eyes",
    "brown eyes",
    "green eyes",
    "purple eyes",
    "yellow eyes",
    "pink eyes",
    "black eyes",
    "aqua eyes",
    "orange eyes",
    "grey eyes",
    "silver eyes",
    "white eyes",
    "multicolored eyes",
}

HAIR_COLOR_TAGS = {
    "blonde hair",
    "brown hair",
    "black hair",
    "blue hair",
    "purple hair",
    "pink hair",
    "red hair",
    "white hair",
    "green hair",
    "grey hair",
    "orange hair",
    "silver hair",
    "aqua hair",
    "multicolored hair",
    "two-tone hair",
}

# Structural conflicts: If Key is present, remove Values.
# We check these against the "base" concept (e.g. "black pants" -> "pants")
CLOTHING_CONFLICTS = {
    "bikini": {"panties", "shorts", "bra"},
    "swimsuit": {"panties", "shorts"},
    "completely nude": {
        "shirt",
        "t-shirt",
        "pants",
        "jeans",
        "skirt",
        "dress",
        "bikini",
        "swimsuit",
        "jacket",
        "coat",
        "sweater",
        "kimono",
        "bra",
        "panties",
    },
    "bra": {"bikini"},
}


def format_danbooru_tag_inverse(formatted_tag: str) -> str:
    """
    Inverse of format_danbooru_tag.

    Transformations (reversed):
    1. Un-escape backslashed parentheses (r'\(' → '(' and r'\)' → ')')
    2. Replace spaces with underscores

    Args:
        formatted_tag (str): The human-readable tag, e.g. "star \\(symbol\\)"

    Returns:
        str: The original Danbooru tag, e.g. "star_(symbol)"

    Examples:
        >>> format_danbooru_tag_inverse("looking at viewer")
        'looking_at_viewer'
        >>> format_danbooru_tag_inverse("star \\(symbol\\)")
        'star_(symbol)'
    """
    # Un-escape parentheses
    unescaped = formatted_tag.replace(r"\(", "(").replace(r"\)", ")")
    # Replace spaces with underscores
    original_tag = unescaped.replace(" ", "_")
    return original_tag


def format_danbooru_tag(tag: str) -> str:
    """
    Format a Danbooru-style tag into more readable text.

    Transformations:
    1. Replace underscores with spaces
    2. Escape parentheses with backslashes

    Args:
        tag (str): The original Danbooru tag

    Returns:
        str: The formatted tag

    Examples:
        >>> format_danbooru_tag("looking_at_viewer")
        'looking at viewer'
        >>> format_danbooru_tag("star_(symbol)")
        'star \(symbol\)'
    """
    # Replace underscores with spaces
    formatted_tag = tag.replace("_", " ")

    # Escape parentheses with backslashes
    formatted_tag = formatted_tag.replace("(", r"\(").replace(")", r"\)")

    return formatted_tag


def normalize_tag_string(prompt: str) -> str:
    """
    Inverses the 'format_tag_string' function.

    Converts a comma-separated prompt with spaces (e.g., "long hair,
    blue eyes") back to a space-separated prompt with underscores
    (e.g., "long_hair blue_eyes"). This is used to standardize the
    upsampled tags for accurate counting.

    Args:
        prompt (str): The comma-separated tag string.

    Returns:
        str: The space-separated, underscore-formatted tag string.
    """
    if pd.isna(prompt) or not prompt:
        return ""

    # Split by comma, strip whitespace, replace internal spaces with
    # underscores, and join with a single space.
    tags = [
        format_danbooru_tag_inverse(t.strip()) for t in prompt.split(",") if t.strip()
    ]
    return " ".join(tags)


def format_tag_string(tag_string: str) -> str:
    """
    Formats a space-separated Danbooru tag string into a comma-separated,
    human-readable format in a highly optimized way.

    This function avoids splitting the string into a list, which is slow.
    Instead, it uses a chain of optimized `replace` calls. The order of
    operations is critical:
    1. Escape parentheses to prevent them from being misinterpreted.
    2. Replace the space separators between tags with ", ".
    3. Replace underscores within tags with spaces.

    Args:
        tag_string (str): The raw, space-separated tag string.
                          e.g., "1girl long_hair star_(symbol)"

    Returns:
        str: A formatted, comma-separated string.
             e.g., "1girl, long hair, star \(symbol\)"
    """
    if pd.isna(tag_string) or not tag_string:
        return ""

    # The sequence of replacements is optimized for speed and correctness.
    return (
        tag_string.replace("(", r"\(")
        .replace(")", r"\)")
        .replace(" ", ", ")
        .replace("_", " ")
    )


def extract_and_remove_person_tags(general_tags_raw: str):
    """
    Finds all person count tags, formats them, and removes them from the
    original string.

    Args:
        general_tags_raw (str): The raw, space-separated general tags.

    Returns:
        A tuple containing:
        - str: A comma-separated string of all found person count tags.
        - str: The original tag string with the person count tags removed.
    """
    if not isinstance(general_tags_raw, str):
        return "", ""

    # Find all non-overlapping matches for person count tags
    found_tags = PERSON_COUNT_REGEX.findall(general_tags_raw)

    # If no digit-based tags are found, check for 'solo' or 'duo'
    if not found_tags:
        if "solo" in general_tags_raw.split(" "):
            found_tags = ["solo"]

    if not found_tags:
        return "", general_tags_raw

    # Create the removal pattern by joining found tags with '|' (OR)
    # e.g., r'\b(1boy|3girls)\b'
    # The word boundaries (\b) are crucial to avoid partial matches
    # like removing 'boy' from 'cowboy'.
    removal_pattern = r"\b(" + "|".join(found_tags) + r")\b"

    # Remove the tags from the original string
    remaining_tags = re.sub(removal_pattern, "", general_tags_raw)

    # Clean up any resulting extra whitespace
    remaining_tags = re.sub(r"\s+", " ", remaining_tags).strip()

    # Join the found tags into a clean, comma-separated string
    person_count_str = ", ".join(found_tags)

    return person_count_str, remaining_tags


def count_tags(
    df: pd.DataFrame,
    normalized_upsampled_tags: pd.Series = None,
    output_path: str = "tag_counts_report.csv",
):
    """
    Creates a csv file containing the counts of each general, artists and character
    tags inside the provided dataframe.

    Args:
        df (pd.DataFrame): The dataframe to count from, it must contain columns
        ['tag_string_general', 'tag_string_character', 'tag_string_artist'].
        normalized_upsampled_tags (pd.Series): Optional series containing the
        upsampled tags of the dataframe.
        output_path (str): Path to save the csv file.

    Returns:
        None
    """

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
    print(f"Saving tag counts to '{output_path}'...")
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

    full_report[["category", "tag", "count"]].to_csv(output_path, index=False)
    print(f"Saved {len(full_report)} unique tag counts to report.")


def analyze_tag_distribution(csv_files: List[str]) -> pd.DataFrame:
    """
    Loads multiple tag count CSVs and merges them to analyze distribution.

    This function takes a list of CSV file paths, each corresponding to a
    quality bucket (e.g., masterpiece, good). It merges them based on
    'category' and 'tag', renaming the count columns to reflect their
    source bucket. This provides a unified view of how tags are
    distributed across different quality tiers.

    Args:
        csv_files (List[str]): A list of paths to the CSV files.
                               Expected format: 'tag_counts_{bucket}.csv'.

    Returns:
        pd.DataFrame: A merged DataFrame with columns like:
                      ['category', 'tag', 'count_masterpiece', ...].
    """
    if not csv_files:
        print("Warning: No CSV files provided for analysis.")
        return pd.DataFrame()

    # Maps the suffix from the filename to the desired column name suffix.
    bucket_map = {
        "masterpiece": "masterpiece",
        "good": "good",
        "bad": "bad",
        "worse": "worse",
    }

    all_dfs = []
    for file_path in csv_files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found, skipping: {file_path}")
            continue

        try:
            # Extracts bucket name, e.g., 'tag_counts_good.csv' -> 'good'
            base_name = os.path.basename(file_path)
            bucket_name = base_name.replace("tag_counts_", "").replace(".csv", "")
            if bucket_name not in bucket_map:
                print(
                    f"Warning: Unrecognized bucket '{bucket_name}' in "
                    f"file {file_path}. Skipping."
                )
                continue
        except Exception:
            print(f"Warning: Could not parse bucket name from {file_path}. Skipping.")
            continue

        df = pd.read_csv(file_path)
        # Rename 'count' to 'count_{bucket_name}' for clarity after merge
        count_col_name = f"count_{bucket_map[bucket_name]}"
        df.rename(columns={"count": count_col_name}, inplace=True)

        if "category" in df.columns and "tag" in df.columns:
            all_dfs.append(df[["category", "tag", count_col_name]])
        else:
            print(f"Warning: Required columns missing in {file_path}.")

    if not all_dfs:
        print("No valid dataframes were loaded. Returning empty DataFrame.")
        return pd.DataFrame()

    # Use reduce to iteratively merge all dataframes on category and tag.
    # 'outer' join ensures that tags present in any file are included.
    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on=["category", "tag"], how="outer"),
        all_dfs,
    )

    # Replace NaN (for tags not in a bucket) with 0 and cast to integer.
    count_cols = [col for col in merged_df if col.startswith("count_")]
    merged_df[count_cols] = merged_df[count_cols].fillna(0).astype(int)

    return merged_df


def get_tags_from_file(file_path: str) -> List[str]:
    """Extracts tags from a single text file in Danbooru format."""
    tags = []
    if not os.path.exists(file_path):
        print(f"Warning: File not found at '{file_path}'. Skipping.")
        return tags
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                tag = format_danbooru_tag_inverse(line.strip())
                if tag:
                    tags.append(tag)
    except Exception as e:
        print(f"Error processing file '{file_path}': {e}")
    return tags


def get_tags_from_knowledge_bases(knowledge_bases_paths: List[str]) -> List[str]:
    """Extracts tags from all txt knowledge base files."""
    tags = set()
    for path in knowledge_bases_paths:
        tags.update(get_tags_from_file(path))
    return list(tags)


def validate_upsampled_batch(upsampled_tags: str, row: pd.DataFrame) -> str:
    """
    Validates and cleans upsampled tags based on Danbooru metadata and heuristics.

    Args:
        upsampled_tags: The string of tags generated by the upsampler.
        row: A dictionary or Series containing metadata (original prompt, counts).
             Expected keys: 'tag_string_general', 'tag_count_character', 'original_prompt'.

    Returns:
        A cleaned string of upsampled tags.
    """

    # 1. Determine Character Count
    # We need the max count to limit attributes (e.g., 1 girl = 1 hair color).
    char_count = 0

    # Try to get from explicit count column
    if "tag_count_character" in row and row["tag_count_character"] is not None:
        try:
            char_count = int(row["tag_count_character"])
        except (ValueError, TypeError):
            char_count = 0

    # If 0 or missing, try to find "1girl", "2boys" in the general tags
    if char_count == 0:
        gen_tags = str(row.get("tag_string_general", "")).split()
        count_tags = [
            "1girl",
            "1boy",
            "2girls",
            "2boys",
            "3girls",
            "3boys",
            "4girls",
            "4boys",
        ]
        found_counts = []
        for t in gen_tags:
            if t in count_tags:
                # Extract the number.
                num_str = t[0]
                if num_str.isdigit():
                    found_counts.append(int(num_str))

        if found_counts:
            # we must sum as 3 chars can be 2girls, 1boy
            char_count = sum(found_counts)

    # Default to 1 if we couldn't find a count but there is a prompt,
    # to allow at least one set of attributes.
    if char_count == 0:
        char_count = 1

    # 2. Analyze Original Prompt (Ground Truth)
    original_prompt = str(row.get("original_prompt", "")).lower()
    original_tags_list = [t.strip() for t in original_prompt.split(",") if t.strip()]

    # Count existing attributes in the original prompt
    orig_hair_len_count = sum(1 for t in original_tags_list if t in HAIR_LENGTH_TAGS)
    orig_hair_col_count = sum(1 for t in original_tags_list if t in HAIR_COLOR_TAGS)
    orig_eye_col_count = sum(1 for t in original_tags_list if t in EYE_COLOR_TAGS)

    has_heterochromia = "heterochromia" in original_tags_list
    has_multicolored = "multicolored hair" in original_tags_list

    # 3. Define Limits
    # The limit is the number of characters.
    # Exception: Heterochromia allows +1 eye color per character (simplified).
    max_hair_len = char_count
    max_hair_col = char_count + (1 if has_multicolored else 0)
    max_eye_col = char_count + (1 if has_heterochromia else 0)

    # 4. Filter Upsampled Tags
    upsampled_list = [t.strip() for t in upsampled_tags.split(",") if t.strip()]
    cleaned_tags = []

    # Track how many we have added from upsampled
    added_hair_len = 0
    added_hair_col = 0
    added_eye_col = 0

    for tag in upsampled_list:
        tag_clean = tag.strip()

        # Hair Length
        if tag_clean in HAIR_LENGTH_TAGS:
            # If original already has enough info, skip upsampled
            if orig_hair_len_count >= max_hair_len:
                continue
            # If we explicitly added enough from upsampled, skip
            if (orig_hair_len_count + added_hair_len) >= max_hair_len:
                continue
            added_hair_len += 1

        # Hair Color
        elif tag_clean in HAIR_COLOR_TAGS:
            if orig_hair_col_count >= max_hair_col:
                continue
            if (orig_hair_col_count + added_hair_col) >= max_hair_col:
                continue
            added_hair_col += 1

        # Eye Color
        elif tag_clean in EYE_COLOR_TAGS:
            # Check for heterochromia in upsampled if not in original
            if tag_clean == "heterochromia":
                max_eye_col += 1

            if orig_eye_col_count >= max_eye_col:
                continue
            if (orig_eye_col_count + added_eye_col) >= max_eye_col:
                continue
            added_eye_col += 1

        # --- Clothing Heuristics ---
        # Check for conflicts with the Original Prompt, if true don't append it
        conflict_found = False
        for original_tag in original_tags_list:
            # Check if any word in original tag matches a conflict key
            # We do a simple check: if original has "black pants", key "pants" triggers
            for key, conflicting_set in CLOTHING_CONFLICTS.items():
                # split to take into account colors
                if key in original_tag.split(" "):
                    for tag in tag_clean.split(" "):
                        if tag in conflicting_set:
                            conflict_found = True
                            break
            if conflict_found:
                break

        if conflict_found:
            continue

        # If passed all checks, keep the tag
        cleaned_tags.append(tag_clean)

    return ", ".join(cleaned_tags)


def sanitize_prompt(prompt_str: str) -> str:
    """Sanitizes a Danbooru-style comma-separated prompt string.

    Args:
        prompt_str: The raw prompt string.
    """
    # Split by comma
    tags = prompt_str.split(",")

    cleaned_tags = []
    for tag in tags:
        # Strip leading/trailing whitespace and collapse internal duplicate spaces
        cleaned = " ".join(tag.strip().split())
        # Only keep non-empty tags (filters out leftover commas like ", ,")
        if cleaned:
            cleaned_tags.append(cleaned)

    # Join with standard single-comma and single-space
    return ", ".join(cleaned_tags)
