from glob import glob
import os
import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, Callable, Generator, List
import json
from PIL import Image
import logging
import multiprocessing

# Define image file extensions
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif"}
AESTHETIC_LABEL = {"worse_score": 0, "bad_score": 1, "good_score": 2, "masterpiece": 3}


def load_all_parquets(
    global_path: str, skip_aes: bool = False, num_parquets: int = 100
) -> pd.DataFrame:
    """Loads parquet files directly using DuckDB zero-copy integration."""
    parquet_glob = os.path.join(global_path, "train-*.parquet")
    aes_csv = os.path.join(global_path, "aes_2024.csv")

    conn = duckdb.connect()
    if not skip_aes and os.path.exists(aes_csv):
        query = f"""
            SELECT p.* EXCLUDE (
                source, up_score, down_score, last_commented_at,
                last_comment_bumped_at, last_noted_at, uploader_id,
                approver_id, pixiv_id, bit_flags
            ), a.score AS aes_score
            FROM read_parquet('{parquet_glob}') p
            LEFT JOIN read_csv_auto('{aes_csv}') a ON p.id = a.id
            LIMIT {num_parquets * 100000}
        """
    else:
        query = f"""
            SELECT * EXCLUDE (
                source, up_score, down_score, last_commented_at,
                last_comment_bumped_at, last_noted_at, uploader_id,
                approver_id, pixiv_id, bit_flags
            )
            FROM read_parquet('{parquet_glob}')
            LIMIT {num_parquets * 100000}
        """
    df = conn.execute(query).df()
    conn.close()
    return df


def load_prior_knowledge_df(
    prior_df_path: str,
    aes_scores_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """Loads prior knowledge dataset from DuckDB database or flat files."""
    conn = duckdb.connect()

    if prior_df_path.endswith(".duckdb"):
        source_query = f"SELECT * FROM '{prior_df_path}'.prior_knowledge"
    elif prior_df_path.endswith(".parquet"):
        source_query = f"SELECT * FROM read_parquet('{prior_df_path}')"
    else:
        source_query = f"SELECT * FROM read_csv_auto('{prior_df_path}')"

    if aes_scores_csv_path and os.path.exists(aes_scores_csv_path):
        query = f"""
            SELECT pk.*, aes.score AS aes_score
            FROM ({source_query}) pk
            LEFT JOIN read_csv_auto('{aes_scores_csv_path}') aes
              ON pk.id = aes.id
        """
    else:
        query = source_query

    df = conn.execute(query).df()
    conn.close()

    df.drop_duplicates(subset=["id"], inplace=True)
    if "aes_score" in df.columns and df["aes_score"].isna().any():
        df["aes_score"] = df["aes_score"].fillna(df["aes_score"].mean())
    if "fav_count" in df.columns and df["fav_count"].isna().any():
        df["fav_count"] = df["fav_count"].fillna(df["fav_count"].mean())

    return df


def dirwalk(
    path: Path, condition: Optional[Callable] = None
) -> Generator[Path, None, None]:
    """Walk through directory and yield files that meet the condition."""
    try:
        for p in path.iterdir():
            if p.is_dir():
                yield from dirwalk(p, condition)
            elif condition is None or condition(p):
                yield p
    except OSError as e:
        logging.error(f"Could not access directory {path}: {e}")


def append_weight_to_json(json_path: str, weight: float):
    """
    Reads a JSON file, appends the 'tag_weight' key, and writes it back.
    This operation is designed to be safe against race conditions if run
    in parallel, though this implementation is serial.

    Args:
        json_path (str): The full path to the JSON file.
        weight (float): The tag weight to append.
    """
    primary_encoding = "utf-8"
    # Fallback to 'utf-16' to handle the 0xff error, which is common
    # with UTF-16 BOMs (0xfffe).
    fallback_encoding = "utf-16"
    data = None

    try:
        with open(json_path, "r", encoding=primary_encoding) as f:
            data = json.load(f)
    except UnicodeDecodeError as e:
        # 2. If decoding fails (e.g., due to 0xff), try the fallback
        print(f"Retrying {json_path} with {fallback_encoding} due to: {e}")
        try:
            with open(json_path, "r", encoding=fallback_encoding) as f:
                data = json.load(f)
        except Exception as fe:
            # 3. If fallback fails, log and exit this sample
            print(f"Warning: Fallback failed for {json_path}. Reason: {fe}")
            return
    except (FileNotFoundError, json.JSONDecodeError) as e:
        # This handles cases where the JSON is missing or corrupted.
        print(f"Warning: Could not update {json_path}. Reason: {e}")

    # 4. If data was loaded successfully, update and write back
    if data is not None:
        data["tag_weight"] = weight

        try:
            with open(json_path, "w", encoding=primary_encoding) as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Warning: Failed to write back {json_path}. Reason: {e}")


def is_valid_image(path: Path) -> bool:
    """
    Check if a file is a valid and non-corrupted image.
    It tries to open the image and verifies it.
    """
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError) as e:
        logging.warning(f"Corrupted image found and skipped: {path} - {e}")
        return False


def _scan_and_validate_worker(directory: Path) -> List[Path]:
    """
    Worker function for multiprocessing. Scans a single directory
    and returns a list of valid image paths.
    """
    return list(dirwalk(directory, is_valid_image))


def _scan_worker(directory: Path) -> List[Path]:
    """
    Worker function for multiprocessing. Scans a directory recursively
    for files with valid image extensions.

    This is much faster as it only checks file names and avoids opening files.
    """
    # Use the highly optimized rglob for recursive file searching and
    # filter by checking the file suffix against a set of known image
    # extensions. This avoids the costly I/O of opening each file.
    return [p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]


def _scan_worker_prompt(directory: Path) -> List[Path]:
    """
    Worker function for multiprocessing. Scans a directory recursively
    for files with valid image extensions.

    This is much faster as it only checks file names and avoids opening files.
    """
    # Use the highly optimized rglob for recursive file searching and
    # filter by checking the file suffix against a set of known image
    # extensions. This avoids the costly I/O of opening each file.
    return [p for p in directory.rglob("*") if p.suffix.lower() == ".txt"]


def parallel_scan_images(
    root_dir: Path,
    num_workers: Optional[int] = None,
    prompts: Optional[bool] = False,
) -> List[Path]:
    """
    Scans a directory for valid images in parallel by distributing
    subdirectories among multiple worker processes.
    """
    # Find subdirectories to distribute as tasks.
    try:
        sub_dirs = [p for p in root_dir.iterdir() if p.is_dir()]
    except FileNotFoundError:
        logging.error(f"Root directory not found: {root_dir}")
        return []

    # If no subdirectories, scan the root directory itself.
    if not sub_dirs:
        logging.info("No subdirectories found. Scanning root directory...")
        return _scan_and_validate_worker(root_dir)

    logging.info(f"Starting parallel scan of {len(sub_dirs)} subdirectories...")

    # If no subdirectories exist, scan the root directory directly in a
    # single-threaded manner.
    if not sub_dirs:
        logging.info("No subdirectories found. Scanning root directory...")
        return _scan_worker(root_dir)

    all_image_paths = []
    # Use a multiprocessing Pool to manage worker processes.
    with multiprocessing.Pool(processes=num_workers) as pool:
        # imap_unordered is used to process results as they complete, which
        # can be slightly more efficient if subdirectories are uneven in size.
        if not prompts:
            results = pool.imap_unordered(_scan_worker, sub_dirs)
        else:
            results = pool.imap_unordered(_scan_worker_prompt, sub_dirs)

        for image_paths in results:
            all_image_paths.extend(image_paths)

    return all_image_paths


def resolve_image_path(path: Path) -> Path:
    """Resolves image path by checking original path or .avif alternative.

    Args:
        path (Path): Target image Path object.

    Returns:
        Path: Existing file Path (original or .avif), or original if neither.
    """
    if path.exists():
        return path
    avif_path = path.with_suffix(".avif")
    if avif_path.exists():
        return avif_path
    return path


def build_id_path_map(root_dir: Path) -> dict[int, Path]:
    """Scans root_dir recursively and returns an ID-to-Path lookup map.

    Supports all extensions (.png, .jpg, .jpeg, .webp, .avif) and
    handles nested character and tier subdirectories in O(1) lookup.
    """
    id_map = {}
    if not root_dir.exists():
        return id_map

    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            try:
                img_id = int(p.stem)
                id_map[img_id] = p
            except ValueError:
                continue
    return id_map


def _read_dataset_duckdb(file_path: str) -> pd.DataFrame:
    """Reads CSV/Parquet file using DuckDB for zero-copy performance."""
    if not os.path.exists(file_path):
        return pd.DataFrame()

    conn = duckdb.connect()
    if file_path.endswith(".parquet"):
        df = conn.execute(f"SELECT * FROM read_parquet('{file_path}')").df()
    elif file_path.endswith(".csv"):
        df = conn.execute(f"SELECT * FROM read_csv_auto('{file_path}')").df()
    else:
        conn.close()
        raise ValueError(f"Unsupported file format: {file_path}")

    conn.close()
    return df
