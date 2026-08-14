"""Generates and incrementally updates a local database of Danbooru metadata.

Uses DuckDB for vectorized queries, zero-copy Pandas conversions, and fast
incremental updates via native upserting.
"""

import os
import time
from typing import List, Set, Dict
from datetime import datetime, timedelta
import requests
import duckdb
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..prompts.prompt_utils import get_tags_from_knowledge_bases

HAS_CURL = False
try:
    from curl_cffi import requests as cffi_requests

    HAS_CURL = True
except ImportError:
    print("curl_cffi couldn't be imported")

RETRY_ATTEMPTS = 3
INITIAL_RETRY_DELAY = 1


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
        session (requests.Session | cffi_requests.Session): The active requests session.
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
            resp = session.get(url, headers=None, cookies=cookies, timeout=10)
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
                delay *= 2
            else:
                return None


def get_newest_ids_for_tags(
    csv_path: str,
    tags: Set[str],
) -> Dict[str, int]:
    """Finds newest post ID per tag by querying CSV directly via DuckDB."""
    if not os.path.exists(csv_path) or not tags:
        return {}

    conn = duckdb.connect()  # In-memory transient connection
    tag_list = list(tags)
    query = """
        SELECT t.tag, MAX(pk.id)
        FROM UNNEST(?::VARCHAR[]) AS t(tag)
        JOIN read_csv_auto(?) pk
          ON list_contains(
                 string_split(pk.tag_string_character, ' '), t.tag
             )
          OR list_contains(
                 string_split(pk.tag_string_artist, ' '), t.tag
             )
        GROUP BY t.tag
    """
    try:
        results = conn.execute(query, [tag_list, csv_path]).fetchall()
        conn.close()
        return {row[0]: row[1] for row in results if row[1] is not None}
    except Exception as e:
        print(f"DuckDB error in get_newest_ids_for_tags: {e}")
        conn.close()
        return {}


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
        if not posts:
            break

        page_valid_posts = []
        stop_fetching = False

        for p in posts:
            post_id = int(p.get("id", 0))

            created_at_str = p.get("created_at")
            try:
                post_time = datetime.fromisoformat(created_at_str)
            except Exception:
                post_time = datetime.now()

            if stop_at_id is not None and post_id <= int(stop_at_id):
                if stop_cutoff_time is None:
                    stop_cutoff_time = post_time - timedelta(days=time_window_days)

            if stop_cutoff_time is not None:
                if post_time < stop_cutoff_time:
                    stop_fetching = True
                    break

            page_valid_posts.append(p)

        all_posts.extend(page_valid_posts)

        if stop_fetching:
            break

        page += 1
        time.sleep(1.0)

    return all_posts


def create_prior_knowledge_dataset(
    knowledge_bases_paths: List[str],
    output_csv_path: str = "prior_knowledge.csv",
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

    # TODO: optimize this, as when using lists it's repetitive
    tags_to_fetch = [tag for tag in all_tags if tag not in completed_tags]

    newest_ids_map = get_newest_ids_for_tags(output_csv_path, completed_tags)

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

    session_class = cffi_requests.Session if HAS_CURL else requests.Session
    session_kwargs = {"impersonate": "chrome"} if HAS_CURL else {}

    with session_class(**session_kwargs) as session:
        for i in range(0, len(tags_to_process), batch_size):
            batch_tags = tags_to_process[i : i + batch_size]
            total_batches = (len(tags_to_process) + batch_size - 1) // batch_size
            print(f"\n--- Processing Batch {i // batch_size + 1}/{total_batches} ---")

            batch_posts = []
            successfully_downloaded_tags = []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_tag = {}
                for tag in batch_tags:
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

            df_batch = pd.DataFrame(batch_posts)
            df_batch.drop_duplicates(subset=["id"], keep="first", inplace=True)

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
                # "tag_count_character",
                # "tag_count_copyright",
                # "tag_string_meta",
            ]

            existing_columns = [
                col for col in columns_to_keep if col in df_batch.columns
            ]
            df_batch = df_batch[existing_columns]

            header = not os.path.exists(output_csv_path)
            df_batch.to_csv(output_csv_path, mode="a", index=False, header=header)
            print(f"Saved {len(df_batch)} new posts to '{output_csv_path}'")

            newly_done = [t for t in successfully_downloaded_tags if t in tags_to_fetch]
            if newly_done:
                with open(completed_tags_log_path, "a") as f:
                    for t in newly_done:
                        f.write(f"{t}\n")

    if os.path.exists(output_csv_path):
        print("\n--- Deduplicating CSV using DuckDB ---")
        conn = duckdb.connect()
        dedup_query = f"""
            COPY (
                SELECT * FROM read_csv_auto('{output_csv_path}')
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY id ORDER BY id
                ) = 1
            ) TO '{output_csv_path}' (HEADER, DELIMITER ',')
        """
        conn.execute(dedup_query)
        conn.close()
        print("CSV deduplication complete.")

    print("\n--- Dataset creation complete. ---")
