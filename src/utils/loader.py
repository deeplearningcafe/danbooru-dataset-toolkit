from glob import glob
import os
import pandas as pd
from pathlib import Path
from typing import Optional, Callable, Generator, List
import json
from PIL import Image
import logging
import multiprocessing

# Define image file extensions
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

def load_all_parquets(global_path:str, skip_aes:bool=False, num_parquets:int=100):
    # Get all parquet files in the directory
    parquet_files = glob(f"{global_path}/train-*.parquet")
    # Create an empty list to store dataframes
    dfs = []

    # Read each parquet file and append to the list
    for i, file in enumerate(parquet_files):
        if i < num_parquets:
            print(f"Loading {os.path.basename(file)}...")
            df_chunk = pd.read_parquet(file)
            dfs.append(df_chunk)

    # Concatenate all dataframes vertically (axis=0)
    df = pd.concat(dfs, axis=0, ignore_index=True)
    columns_to_drop = ['source', 'up_score', 'down_score',
                   'last_commented_at', 'last_comment_bumped_at', 'last_noted_at',
                   'uploader_id', 'approver_id', 'pixiv_id', 'bit_flags', ]
    df.drop(columns=columns_to_drop, inplace=True)
    if not skip_aes:
        try:
            score_df = pd.read_csv(f"{global_path}/aes_2024.csv")
            score_df = score_df.rename(columns={'score': 'aes_score'})
            df = df.merge(score_df, on='id', how='left')
        except FileNotFoundError:
            raise FileNotFoundError("Aesthetic labels file not founded")
    return df

def load_prior_knowledge_df(
    prior_df_path: str,
    aes_scores_csv_path: str=None,
):
    """Loads the csv containing the downloaded dataset from the prior knowledge
    and merges it with the precomputed aesthetic scores csv.

    Args:
        prior_df_path (str): Path to the prior knowledge dataset.
        aes_scores_csv_path (str, optional): Path to the aesthetic scores. Defaults to None.

    Returns:
        Pandas.DataFrame: Prior knowledge dataframe.
    """
    prior_knowledge_samples = pd.read_csv(
        prior_df_path, header=0, low_memory=False
    )
    prior_knowledge_samples = prior_knowledge_samples.drop_duplicates().reset_index(drop=True)

    if aes_scores_csv_path and os.path.exists(aes_scores_csv_path):
        aes_df = pd.read_csv(aes_scores_csv_path)
        aes_df = aes_df.rename(columns={'score': 'aes_score'})
        prior_knowledge_samples = pd.merge(
            prior_knowledge_samples, aes_df[['id', 'aes_score']],
            on='id', how='left'
        )

    # Fill missing aes_score values with the mean to include new images.
    if 'aes_score' in prior_knowledge_samples.columns:
        nan_count = prior_knowledge_samples['aes_score'].isna().sum()
        if nan_count > 0:
            # the highest masterpiece number comes from applying the fillnan here and not after filtering
            mean_aes = prior_knowledge_samples['aes_score'].mean()
            prior_knowledge_samples['aes_score'] = prior_knowledge_samples[
                'aes_score'
            ].fillna(mean_aes)

            print(f"  - Filled {nan_count} missing 'aes_score' "
                    f"values with the mean ({mean_aes:.4f}).")
    if 'fav_count' in prior_knowledge_samples.columns:
        nan_count = prior_knowledge_samples['fav_count'].isna().sum()
        if nan_count > 0:
            mean_aes = prior_knowledge_samples['fav_count'].mean()
            prior_knowledge_samples['fav_count'] = prior_knowledge_samples[
                'fav_count'
            ].fillna(mean_aes)

            print(f"  - Filled {nan_count} missing 'fav_count' "
                    f"values with the mean ({mean_aes:.4f}).")
    return prior_knowledge_samples

def dirwalk(
    path: Path,
    condition: Optional[Callable] = None
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
    primary_encoding = 'utf-8'
    # Fallback to 'utf-16' to handle the 0xff error, which is common
    # with UTF-16 BOMs (0xfffe).
    fallback_encoding = 'utf-16'
    data = None

    try:
        with open(json_path, 'r', encoding=primary_encoding) as f:
            data = json.load(f)
    except UnicodeDecodeError as e:
        # 2. If decoding fails (e.g., due to 0xff), try the fallback
        print(f"Retrying {json_path} with {fallback_encoding} due to: {e}")
        try:
            with open(json_path, 'r', encoding=fallback_encoding) as f:
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
        data['tag_weight'] = weight

        try:
            with open(json_path, 'w', encoding=primary_encoding) as f:
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
    return [
        p for p in directory.rglob('*')
        if p.suffix.lower() in IMAGE_SUFFIXES
    ]

def _scan_worker_prompt(directory: Path) -> List[Path]:
    """
    Worker function for multiprocessing. Scans a directory recursively
    for files with valid image extensions.

    This is much faster as it only checks file names and avoids opening files.
    """
    # Use the highly optimized rglob for recursive file searching and
    # filter by checking the file suffix against a set of known image
    # extensions. This avoids the costly I/O of opening each file.
    return [
        p for p in directory.rglob('*')
        if p.suffix.lower() == ".txt"
    ]

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

    logging.info(
        f"Starting parallel scan of {len(sub_dirs)} subdirectories..."
    )

    # If no subdirectories exist, scan the root directory directly in a
    # single-threaded manner.
    if not sub_dirs:
        logging.info(
            "No subdirectories found. Scanning root directory..."
        )
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