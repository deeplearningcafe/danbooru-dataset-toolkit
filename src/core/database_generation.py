"""Generates and incrementally updates a local database of Danbooru metadata.

This module handles the creation and updating of a dataset containing image
metadata from the Danbooru API. It is designed to scale efficiently by using
a combination of a crash-recovery text log and the existing dataset CSV as
the absolute source of truth for state tracking.

Key Features:
    * Incremental Updates: By scanning the existing CSV at startup, the module
      identifies the newest downloaded ID for each tag. It then only fetches
      newer posts, drastically reducing API and I/O overhead.
    * Time-Window Overlap: During updates, it fetches posts up to the last
      known ID *and* a specified time window (e.g., 7 days prior). This ensures
      that changing metadata (like upvotes or tags) on recently uploaded
      images is captured and deduplicated cleanly.
    * Crash Recovery: A lightweight text file logs fully processed tags to
      prevent starting from scratch if the script is interrupted.

This architecture makes the module highly robust for maintaining an up-to-date
prior knowledge dataset for generative model training.
"""

import requests
import time
import pandas as pd
import os
from typing import List, Set, Dict
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

from ..prompts.prompt_utils import get_tags_from_knowledge_bases

RETRY_ATTEMPTS = 3
INITIAL_RETRY_DELAY = 1  # seconds


def load_completed_tags(log_path: str) -> set:
    """Loads previously processed tags from a log file.

    Acts as a crash recovery mechanism to avoid re-fetching fully
    completed tags if the pipeline is interrupted.

    Args:
        log_path (str): Path to the text file containing logged tags.

    Returns:
        set: A set of tag strings that have already been completely processed.
    """
    if not os.path.exists(log_path):
        return set()
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            # Read each line, strip whitespace, and add to a set
            return {line.strip() for line in f if line.strip()}
    except Exception as e:
        print(f"Warning: Could not read completed tags log '{log_path}': {e}")
        return set()


def fetch_page_with_retry(session, tag, page):
    """Fetches a single page of JSON results for a tag with retries.

    Implements exponential backoff.

    Args:
        session (requests.Session): The active requests session.
        tag (str): The Danbooru tag to search.
        page (int): The API page number to fetch.

    Returns:
        list or None: A list of post dictionaries if successful, or None
        if all retry attempts fail.
    """
    url = f"https://danbooru.donmai.us/posts.json?tags={tag}&page={page}"
    # headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36'} # Add User-Agent
    cookies = {
        "cf_clearance": "YOUR-COOKIE",
    }
    delay = INITIAL_RETRY_DELAY
    for attempt in range(RETRY_ATTEMPTS):
        try:
            with session.get(url, headers=None, cookies=cookies, timeout=10) as resp:
                resp.raise_for_status()
                return resp.json()
        except (requests.RequestException, requests.Timeout) as e:
            print(
                f"Attempt {attempt + 1}/{RETRY_ATTEMPTS} failed for "
                f"fetching page {page} for tag '{tag}': {e}. Retrying in "
                f"{delay}s..."
            )
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                return None  # Return None after all retries fail


def get_newest_ids_for_tags(
    dataset_path: str,
    tags: Set[str],
) -> Dict[str, int]:
    """Scans the existing dataset to find the newest post ID per tag.

    This acts as the ground-truth watermark for incremental updates,
    preventing the script from re-downloading the entire history of a tag.
    It uses a memory-efficient strategy by dropping rows once evaluated.

    Args:
        dataset_path (str): Path to the existing CSV dataset.
        tags (Set[str]): Set of tags to evaluate against the dataset.

    Returns:
        Dict[str, int]: Mapping of tags to their maximum downloaded ID.
    """

    if not os.path.exists(dataset_path) or not tags:
        return {}

    print("Scanning existing dataset to find newest post IDs for each tag...")
    try:
        # Read only the necessary columns to minimize memory usage.
        # We now target the specific character and artist columns.
        df = pd.read_csv(
            dataset_path,
            usecols=["id", "tag_string_character", "tag_string_artist"],
            header=0,
            low_memory=True,
        )
        df.fillna("", inplace=True)

    except (FileNotFoundError, ValueError):
        # Handle cases where the file doesn't exist or is empty/invalid.
        return {}

    # Separate tags into characters and artists. We assume any tag present
    # in the artist column is an artist tag. This is crucial for the logic.
    all_artist_tags_in_df = set(" ".join(df["tag_string_artist"]).split())
    artist_tags_to_keep = tags.intersection(all_artist_tags_in_df)
    character_tags_to_process = tags - artist_tags_to_keep

    # Pre-calculate the number of characters for each sample. This avoids
    # recalculating it inside the loop.
    df["tag_count_character"] = (
        df["tag_string_character"]
        .str.split(" ")
        .apply(lambda x: len(x) if x != [""] else 0)
    )

    # Pre-generate a boolean mask for rows containing any priority artist.
    # These rows should never be removed, as they are valuable samples.
    if artist_tags_to_keep:
        # The regex is updated to correctly handle tags with special
        # characters by defining boundaries as start/end of string or space.
        artist_regex = "|".join(
            rf"(?:^|\s){re.escape(tag)}(?:$|\s)" for tag in artist_tags_to_keep
        )
        df["has_priority_artist"] = df["tag_string_artist"].str.contains(
            artist_regex, na=False
        )
    else:
        df["has_priority_artist"] = False

    newest_ids = {}
    # Iterating through tags is more memory-efficient than exploding a large
    # DataFrame, especially when the number of tags is much smaller than
    # the number of rows in the dataset.
    print(f"Initial dataset size: {len(df)} rows.")
    # Process character tags first to leverage the DataFrame reduction.
    for tag in character_tags_to_process:
        if df.empty:
            break  # Stop if all rows have been processed

        # Use regex with word boundaries for exact tag matching.
        regex_pattern = rf"(?:^|\s){re.escape(tag)}(?:$|\s)"

        mask = df["tag_string_character"].str.contains(regex_pattern, na=False)

        if mask.any():
            newest_ids[tag] = df.loc[mask, "id"].max()

            # Define conditions for removing a row after it has been used:
            # 1. It contains the character tag we just processed.
            # 2. It contains ONLY ONE character.
            # 3. It does NOT contain any of our priority artists.
            removal_mask = (
                mask & (df["tag_count_character"] == 1) & (~df["has_priority_artist"])
            )

            rows_to_drop = df.index[removal_mask]
            if not rows_to_drop.empty:
                df.drop(rows_to_drop, inplace=True)

    # --- Final Step: Process Artists on the Reduced DataFrame ---
    # Artist samples are not used for removal logic, so they are
    # processed last on the already-reduced DataFrame.
    print(f"Length for the artist search {len(df)}")
    for tag in artist_tags_to_keep:
        if df.empty:
            break

        regex_pattern = rf"(?:^|\s){re.escape(tag)}(?:$|\s)"
        mask = df["tag_string_artist"].str.contains(regex_pattern, na=False)

        if mask.any():
            # In case the ID was already found via the character search,
            # take the maximum of the two.
            newest_ids[tag] = max(newest_ids.get(tag, 0), df.loc[mask, "id"].max())

    print(
        f"Found newest IDs for {len(newest_ids)} tags. "
        f"Final dataset size for scan: {len(df)} rows."
    )
    return newest_ids


def fetch_all_for_tag(session, tag, stop_at_id: int = None, time_window_days: int = 7):
    """
    Fetches all posts for a single tag by paginating.

    Stops fetching if a post ID is encountered that is <= `stop_at_id`
    AND the post's creation time is older than the time window.
    """
    all_posts = []
    page = 1
    stop_cutoff_time = None

    while True:
        posts = fetch_page_with_retry(session, tag, page)
        if not posts:  # An empty list indicates the last page was reached
            break

        page_valid_posts = []
        stop_fetching = False

        for p in posts:
            post_id = int(p.get("id", 0))

            # CHANGED: Safely parse created_at to handle the time window
            created_at_str = p.get("created_at")
            try:
                post_time = datetime.fromisoformat(created_at_str)
            except Exception:
                post_time = datetime.now()

            if stop_at_id is not None and post_id <= int(stop_at_id):
                # CHANGED: Set cutoff to time_window_days before stop_id
                # only updated once
                if stop_cutoff_time is None:
                    stop_cutoff_time = post_time - timedelta(days=time_window_days)

            # CHANGED: Continue fetching until we surpass the time window
            if stop_cutoff_time is not None:
                if post_time < stop_cutoff_time:
                    stop_fetching = True
                    break

            page_valid_posts.append(p)

        all_posts.extend(page_valid_posts)

        if stop_fetching:
            break

        page += 1
        time.sleep(0.5)

    return all_posts


def create_prior_knowledge_dataset(
    knowledge_bases_paths: List[str],
    output_csv_path: str = "prior_knowledge.parquet",
    max_workers: int = 10,
    batch_size: int = 100,
    time_window_days: int = 7,
):
    """
    Creates a dataset by downloading posts from the Danbooru API,
    saving progress incrementally and allowing the script to be resumed.

    Args:
        knowledge_bases_paths (List[str]): Paths to files with tags.
        output_csv_path (str): Path to save the final CSV file.
        max_workers (int): Max concurrent threads for API requests.
        batch_size (int): Number of tags to process before saving progress.
    """
    print("--- Starting Prior Knowledge Dataset Creation ---")
    print(f"Using {max_workers} workers and {batch_size} batch size")
    completed_tags_log_path = "completed_tags.txt"

    all_tags = get_tags_from_knowledge_bases(knowledge_bases_paths)
    completed_tags = load_completed_tags(completed_tags_log_path)
    tags_to_fetch = [tag for tag in all_tags if tag not in completed_tags]

    # Get the newest post IDs for tags that have already been downloaded.
    newest_ids_map = get_newest_ids_for_tags(output_csv_path, completed_tags)

    # Combine new tags and existing tags to process all of them.
    # New tags are prioritized to ensure they are downloaded first.
    tags_to_process = tags_to_fetch + list(completed_tags)

    if not tags_to_process:
        print("All tags have already been downloaded. Exiting.")
        return
    print(f"Found {len(tags_to_process)} unique tags to download.")

    print(f"Found {len(all_tags)} total tags.")
    print(f"{len(completed_tags)} tags will be checked for updates.")
    print(f"{len(tags_to_fetch)} new tags will be downloaded.")
    print(f"{tags_to_fetch}")

    completed_tags_without_id = [
        tag for tag in completed_tags if newest_ids_map.get(tag) is None
    ]
    print(f"The following tags stop id wasn't found {completed_tags_without_id}")

    # Use a requests.Session for connection pooling and efficiency
    with requests.Session() as session:
        for i in range(0, len(tags_to_process), batch_size):
            batch_tags = tags_to_process[i : i + batch_size]
            total_batches = (len(tags_to_process) + batch_size - 1) // batch_size
            print(f"\n--- Processing Batch {i // batch_size + 1}/{total_batches} ---")

            batch_posts = []
            successfully_downloaded_tags = []

            # Use ThreadPoolExecutor to manage a pool of worker threads
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit a task to the executor for each tag
                future_to_tag = {}
                for tag in batch_tags:
                    # Get the newest ID for the tag if it exists.
                    # This will be None for new tags.
                    stop_id = newest_ids_map.get(tag)
                    future = executor.submit(
                        fetch_all_for_tag,
                        session,
                        tag,
                        stop_at_id=stop_id,
                        time_window_days=time_window_days,
                    )
                    future_to_tag[future] = tag

                # Process results as they are completed
                for future in as_completed(future_to_tag):
                    tag = future_to_tag[future]
                    try:
                        result = future.result()
                        if result:
                            batch_posts.extend(result)
                            successfully_downloaded_tags.append(tag)
                            print(
                                f"Successfully fetched {len(result)} posts for tag: '{tag}'"
                            )
                    except Exception as e:
                        print(f"Error fetching posts for tag '{tag}': {e}")

            if not batch_posts:
                print("No posts were downloaded. Exiting.")
                return

            # Convert the current batch to a DataFrame and clean it
            df_batch = pd.DataFrame(batch_posts)
            df_batch.drop_duplicates(subset=["id"], keep="first", inplace=True)

            # Define a fixed order of columns to ensure consistency.
            # Using a list instead of a set guarantees the column order.
            columns_to_keep = [
                "id",
                "created_at",
                "score",
                "source",
                "md5",
                "rating",
                "image_width",
                "image_height",
                "tag_string",
                "fav_count",
                "file_ext",
                "parent_id",
                "is_deleted",
                "is_banned",
                "file_url",
                "large_file_url",
                "preview_file_url",
                "tag_string_general",
                "tag_string_character",
                "tag_string_copyright",
                "tag_string_artist",
                "tag_count_character",
                "tag_count_copyright",
                "tag_string_meta",
            ]

            # Filter for columns that exist in the DataFrame, but maintain
            # the predefined order from the list.
            existing_columns = [
                col for col in columns_to_keep if col in df_batch.columns
            ]
            df_batch = df_batch[existing_columns]

            try:
                # Write header only if the file is new/empty
                header = not os.path.exists(output_csv_path)
                df_batch.to_csv(output_csv_path, mode="a", index=False, header=header)
                print(f"Saved {len(df_batch)} new unique posts to '{output_csv_path}'")

                # Log only the newly completed tags to allow resuming.
                # Updated tags are already in the completed log.
                newly_completed_tags = [
                    tag for tag in successfully_downloaded_tags if tag in tags_to_fetch
                ]
                if newly_completed_tags:
                    with open(completed_tags_log_path, "a", encoding="utf-8") as f:
                        for tag in newly_completed_tags:
                            f.write(f"{tag}\n")
                    print(f"Logged {len(newly_completed_tags)} new tags.")

            except Exception as e:
                print(f"Error saving batch to file: {e}")

    # CHANGED: Global deduplication added at the end of the update process
    print("\n--- Deduplicating database to keep updated metadata ---")
    try:
        if os.path.exists(output_csv_path):
            print(f"Loading {output_csv_path} for deduplication...")
            final_df = pd.read_csv(output_csv_path, low_memory=True)
            initial_len = len(final_df)

            # Keep 'last' to retain the newly appended rows with updated
            # metadata from the 1-week time window.
            final_df.drop_duplicates(subset=["id"], keep="last", inplace=True)
            final_df.to_csv(output_csv_path, index=False)
            print(
                f"Global deduplication complete: Reduced from "
                f"{initial_len} to {len(final_df)} rows."
            )
    except Exception as e:
        print(f"Error during final dataset deduplication: {e}")

    print("\n--- All batches processed. Dataset creation complete. ---")
