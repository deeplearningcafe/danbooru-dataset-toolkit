import cv2
import faiss
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import re
from ..utils.loader import load_prior_knowledge_df, IMAGE_SUFFIXES
from ..prompts.prompt_utils import format_danbooru_tag_inverse

# Set up logging
logger = logging.getLogger(__name__)


def is_image_file(path: Path) -> bool:
    """Check if a file has a common image extension."""
    return path.suffix.lower() in IMAGE_SUFFIXES


def find_image_files(root_dir: Path) -> list[Path]:
    """Recursively find all image files in a directory."""
    image_paths = []
    for p in tqdm(root_dir.rglob("*")):
        if is_image_file(p):
            image_paths.append(p)
    return image_paths


def _move_file_worker(src_path: Path, dest_path: Path):
    """Worker to move a single file and create its parent directory."""
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_path), str(dest_path))
    except Exception as e:
        logger.error(f"Failed to move {src_path} to {dest_path}: {e}")


class DSU:
    """Disjoint Set Union (Union-Find) data structure."""

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_j] = root_i


def _hash_batch_worker(paths: list[Path]):
    """
    Worker function to read and hash a BATCH of images.
    This is more efficient as it reduces function call overhead.
    """
    phasher = cv2.img_hash.PHash_create()
    batch_hashes = []
    batch_valid_paths = []
    for path in paths:
        try:
            image = cv2.imread(str(path))
            if image is None:
                logger.warning(f"Could not read image, skipping: {path}")
                continue

            # The compute method returns a numpy array, we append it directly
            batch_hashes.append(phasher.compute(image)[0])
            batch_valid_paths.append(path)
        except Exception as e:
            logger.error(f"Error hashing {path}: {e}")

    return batch_hashes, batch_valid_paths


def _move_back_from_dedup(src_path: Path, dedup_root: Path, original_root: Path):
    """
    Moves a single file from a deduplication subfolder back to its
    inferred original location, along with its .txt file.
    This function uses the same path logic as the `recover_files` function.

    Args:
        src_path: The path to the source file to move.
        dedup_root: The root directory of the deduplication output.
        original_root: The root directory of the original dataset.
    """
    try:
        # Assumes structure: dedup_root/group_id/original_rel_path/file.ext
        # We determine the original path by stripping the group_id folder.
        relative_to_dedup = src_path.relative_to(dedup_root)
        original_relative_path = Path(*relative_to_dedup.parts[1:])
        dest_path = original_root / original_relative_path

        # Move the main file
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_path), str(dest_path))

        # Move corresponding prompt file if it exists
        prompt_src = src_path.with_suffix(".txt")
        if prompt_src.exists():
            prompt_dest = dest_path.with_suffix(".txt")
            shutil.move(str(prompt_src), str(prompt_dest))
    except (ValueError, IndexError):
        logger.warning(f"Could not determine original path for {src_path}. Skipping.")
    except Exception as e:
        logger.error(f"Failed to move {src_path}: {e}")


def automate_deduplication(
    deduplication_dir: str,
    root_dir: str,
    sampled_ids_csv: str,
    prior_df_path: str,
):
    """
    Automates cleaning of the deduplication directory based on heuristics.

    This function processes the output of `deduplicate_images` by applying
    a set of rules to automatically keep one image from a group and delete
    the rest. Groups that do not fit the rules are skipped for manual review.

    Args:
        deduplication_dir: The directory where duplicated files were moved.
        root_dir: The original dataset directory to move files back to.
        hash_threshold: Max pHash distance to consider images identical.
        large_file_mb: File size threshold in MB to define a "large" file.
    """
    dedup_path = Path(deduplication_dir)
    root_path = Path(root_dir)

    prior_knowledge_samples = load_prior_knowledge_df(prior_df_path, None)
    print(f"Loaded prior knowledge df of {len(prior_knowledge_samples)} samples.")
    try:
        sampled_df = pd.read_csv(sampled_ids_csv, header=0, low_memory=False)
    except FileNotFoundError:
        print("Error: Sampled IDs file not found. Run sampling first.")
        return
    # merge to get the quality_tier
    prior_knowledge_samples = pd.merge(
        prior_knowledge_samples, sampled_df[["id", "quality_tier"]], how="left", on="id"
    )
    # samples without quality_tier means not sampled
    prior_knowledge_samples.dropna(subset=["quality_tier"], inplace=True)
    # free memory
    del sampled_df
    prior_knowledge_samples["tag_count"] = (
        prior_knowledge_samples["tag_string"].str.split(" ").str.len()
    )

    if not dedup_path.is_dir():
        logger.error(f"Deduplication directory not found: {dedup_path}")
        return

    group_paths = [p for p in dedup_path.iterdir() if p.is_dir()]
    logger.info(f"Found {len(group_paths)} groups to process automatically.")
    skipped_folder = 0
    moved_back_folders = 0
    for group_path in tqdm(group_paths, desc="Automating Deduplication"):
        # Instead of assuming immediate subfolders are class IDs, we recursively
        # find all images and group them by their immediate parent folder.
        # This handles both 'group/0/img' and 'group/character/0/img' structures.

        all_images = [p for p in group_path.rglob("*") if is_image_file(p)]

        if not all_images:
            # Empty group folder, just remove it
            shutil.rmtree(group_path)
            continue

        # Group images by their parent directory (the Class ID folder)
        images_by_parent = defaultdict(list)
        for img in all_images:
            images_by_parent[img.parent].append(img)

        # These are the folders containing the actual images (e.g., "0", "3")
        subfolders = list(images_by_parent.keys())

        # Rule 1: Multiple subfolders (interpreted as "quality_tiers")
        if len(subfolders) > 1:
            try:
                # Keep the folder whose name is the highest number.
                subfolders.sort(key=lambda p: int(p.name), reverse=True)
                folder_to_keep = subfolders[0]
                folders_to_delete = subfolders[1:]

                # Move files from the best folder back to the root dataset
                files_to_recover = images_by_parent[folder_to_keep]

                for file_path in files_to_recover:
                    _move_back_from_dedup(file_path, dedup_path, root_path)

                # Clean up by deleting the entire processed group folder
                shutil.rmtree(group_path)
                moved_back_folders += 1
                continue
            except (ValueError, IndexError):
                # This occurs if folder names are not numeric, so we skip.
                logger.warning(
                    f"Skipping group {group_path.name} due to non-numeric "
                    "subfolder names."
                )
                continue

        # Rule 2: Single subfolder containing exactly two images
        if len(subfolders) == 1:
            # Get the list of images from the dictionary we built earlier
            image_files = images_by_parent[subfolders[0]]

            if len(image_files) < 2:
                # Skip if there is only one image or no images
                skipped_folder += 1
                continue

            try:
                # --- Apply new heuristics based on prior_knowledge_samples ---
                image_ids = [int(p.stem) for p in image_files]
                group_df = prior_knowledge_samples[
                    prior_knowledge_samples["id"].isin(image_ids)
                ].set_index("id")

                if len(group_df) != len(image_files):
                    logger.warning(
                        f"Skipping group {group_path.name} due to missing "
                        "metadata for some images."
                    )
                    skipped_folder += 1
                    continue

                ids_to_keep = set()

                if len(image_files) == 2:
                    # Case 1: Two images. Keep the best one.
                    sorted_df = group_df.sort_values(
                        by=["tag_count", "score", "fav_count"],
                        ascending=[False, False, False],
                    )
                    ids_to_keep.add(sorted_df.index[0])

                elif len(image_files) >= 3:
                    # Case 2: Three or more images.
                    # Keep the one with the longest prompt.
                    id_max_tags = group_df["tag_count"].idxmax()
                    ids_to_keep.add(id_max_tags)

                    # Keep the one with the highest score.
                    # If it's the same as the one with max tags,
                    # keep the second highest score.
                    sorted_by_score = group_df.sort_values(by="score", ascending=False)
                    if sorted_by_score.index[0] == id_max_tags:
                        if len(sorted_by_score) > 1:
                            ids_to_keep.add(sorted_by_score.index[1])
                    else:
                        ids_to_keep.add(sorted_by_score.index[0])

                # Determine which files to keep and delete
                files_to_keep = [p for p in image_files if int(p.stem) in ids_to_keep]

                # Perform file operations
                for file_to_keep in files_to_keep:
                    _move_back_from_dedup(file_to_keep, dedup_path, root_path)

                # Clean up the entire group folder
                shutil.rmtree(group_path)
                moved_back_folders += 1

            except Exception as e:
                logger.error(f"Error processing group {group_path.name}: {e}")
                skipped_folder += 1
    print(f"Moved back folders {moved_back_folders}\nSkipped folders {skipped_folder}")


def recover_files(output_path: Path, root_path: Path, max_workers: int = 8):
    """
    Moves files from the deduplication output directory back to their
    original locations in the root directory.

    This function assumes that the user has manually curated the files
    in the output_dir, leaving only the ones they wish to restore.

    Args:
        output_dir: The directory where duplicated files were moved.
        root_dir: The original dataset directory to move files back to.
        max_workers: The number of worker threads for parallel file moving.
    """

    if not output_path.is_dir():
        logger.error(f"Output directory not found: {output_path}")
        return

    logger.info(f"Scanning for files to recover from {output_path}...")
    # Find all files, not just images, to include .txt files etc.
    files_to_move = [p for p in output_path.rglob("*") if p.is_file()]

    if not files_to_move:
        logger.warning("No files found in the output directory to recover.")
        return

    logger.info(
        f"Found {len(files_to_move)} files. "
        f"Preparing to move them back to {root_path}..."
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for src_path in files_to_move:
            try:
                # Path relative to output_dir is e.g.:
                # 'group_name/original/relative/path/file.ext'
                relative_to_output = src_path.relative_to(output_path)

                # We skip the first part ('group_name')
                original_relative_path = Path(*relative_to_output.parts[1:])

                dest_path = root_path / original_relative_path

                futures.append(executor.submit(_move_file_worker, src_path, dest_path))
            except (ValueError, IndexError):
                logger.warning(
                    f"Could not determine original path for {src_path}. Skipping."
                )

        progress = tqdm(
            as_completed(futures), total=len(futures), desc="Recovering Files"
        )
        for future in progress:
            future.result()

    logger.info("File recovery process complete.")


def _scan_subdir_for_ids(subdir: Path) -> set[str]:
    """
    Scans a single directory recursively for image files and returns a set
    of their stems (filenames without extension).
    """
    # Recursively find all files and filter them using the is_image_file
    # helper to ensure only valid image types are processed.
    return {p.stem for p in subdir.rglob("*") if is_image_file(p)}


def create_exclusion_list(
    sampled_ids_csv: str,
    download_dir: str,
    start_index: int,
    max_downloads: int,
    output_csv_path: str,
    num_workers: int = 4,
):
    """
    Compares downloaded images against the original sampled list to find
    missing files (e.g., deleted during deduplication) and creates an
    exclusion list to prevent them from being sampled again.

    Args:
        sampled_ids_csv: Path to the CSV with all originally sampled IDs.
        download_dir: The root directory where images were downloaded.
        start_index: The starting row index from the sampled_ids_csv.
        max_downloads: The number of rows to process from the start_index.
        output_csv_path: Path to save the new exclusion CSV file.
        num_workers: Number of threads to use for scanning directories.
    """
    logger.info("Starting to create exclusion list for missing files...")
    root_path = Path(download_dir)

    # Step 1: Load the target subset of expected image IDs.
    try:
        source_df = pd.read_csv(sampled_ids_csv)
    except FileNotFoundError:
        logger.error(f"Source sampled IDs file not found: {sampled_ids_csv}")
        return

    end_index = (
        start_index + max_downloads if max_downloads is not None else len(source_df)
    )
    target_df = source_df.iloc[start_index:end_index]
    expected_ids = set(target_df["id"].astype(str))
    logger.info(
        f"Loaded {len(expected_ids)} expected IDs from "
        f"{sampled_ids_csv} (rows {start_index} to {end_index})."
    )

    # Step 2: Scan download directory in parallel to find actual image IDs.
    subdirs = [d for d in root_path.iterdir() if d.is_dir()]
    if not subdirs:
        logger.warning(f"No subdirectories in {root_path}. Scanning root.")
        subdirs = [root_path]

    logger.info(f"Scanning {len(subdirs)} subdirs with {num_workers} threads...")
    found_ids = set()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_subdir = {
            executor.submit(_scan_subdir_for_ids, subdir): subdir for subdir in subdirs
        }

        progress = tqdm(
            as_completed(future_to_subdir), total=len(subdirs), desc="Scanning Dirs"
        )
        for future in progress:
            found_ids.update(future.result())

    logger.info(f"Found {len(found_ids)} images in {download_dir}.")

    # Step 3: Determine which IDs are missing by set difference.
    missing_ids = expected_ids - found_ids
    logger.info(f"Identified {len(missing_ids)} missing image IDs.")

    if not missing_ids:
        logger.info("No missing images found. Exclusion list not needed.")
        return

    # Step 4: Create and save the exclusion DataFrame.
    exclude_df = pd.DataFrame(list(missing_ids), columns=["id"])
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    exclude_df.to_csv(output_csv_path, index=False)
    logger.info(f"Exclusion list saved to {output_csv_path}")


def deduplicate_images(
    root_dir: str,
    output_dir: str,
    threshold: int = 5,
    batch_size: int = 128,
    max_workers: int = 8,
    move_back: bool = False,
    artists_list: list = None,
    sampled_ids_csv: str = None,
):
    """
    Finds and moves similar images using an efficient, asynchronous, and
    batched perceptual hashing pipeline. Can also be used to move
    curated files back to the original directory.

    This function performs the following steps:
    1. Creates batches of image paths.
    2. Uses a ThreadPoolExecutor to process these batches asynchronously.
       Each worker hashes an entire batch of images.
    3. Uses Faiss to efficiently find all pairs of images within the hash
       distance threshold.
    4. Employs a Disjoint Set Union (DSU) data structure to efficiently
       group all transitively similar images.
    5. Moves the grouped images and their .txt prompt files to subfolders.

    Args:
        root_dir: The source directory containing the image dataset.
        output_dir: Directory where groups of similar images will be moved.
        threshold: Max Hamming distance to consider images similar.
        batch_size: Number of images to process in a single batch per worker.
        max_workers: The number of worker threads for parallel hashing.
        move_back: If True, reverses the process, moving files from
                   `output_dir` back to `root_dir`.
    """

    root_path = Path(root_dir)
    output_path = Path(output_dir)

    # --- Execute recovery logic if move_back is True ---
    if move_back:
        recover_files(output_path, root_path, max_workers)
        return

    output_path.mkdir(exist_ok=True)

    image_paths = find_image_files(root_path)

    # --- Filter out artist images from deduplication ---
    if artists_list and sampled_ids_csv:
        try:
            sampled_df = pd.read_csv(sampled_ids_csv, low_memory=False)
            formatted_artists = [format_danbooru_tag_inverse(a) for a in artists_list]
            if "tag_string" in sampled_df.columns:
                artist_ids = set()
                for artist in formatted_artists:
                    artist_pattern = r"(?:^|\s)" + re.escape(artist) + r"(?:$|\s)"
                    mask = sampled_df["tag_string"].str.contains(
                        artist_pattern, regex=True, na=False
                    )
                    artist_ids.update(sampled_df.loc[mask, "id"].astype(str).tolist())

                filtered_paths = [p for p in image_paths if p.stem not in artist_ids]
                logger.info(
                    f"Excluded {len(image_paths) - len(filtered_paths)} images "
                    f"belonging to protected artists from deduplication."
                )
                image_paths = filtered_paths
        except Exception as e:
            logger.error(f"Failed to filter artists in deduplication: {e}")
    if not image_paths:
        logger.warning("No images found. Deduplication step is complete.")
        return

    # --- 1. Asynchronous Batch Hashing ---
    logger.info(
        f"Computing hashes with {max_workers} workers, "
        f"batch size {batch_size}..."
        f"Founded {len(image_paths)} image paths"
    )
    # Create batches of file paths
    batches = [
        image_paths[i : i + batch_size] for i in range(0, len(image_paths), batch_size)
    ]

    hashes, valid_paths = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(_hash_batch_worker, batch): batch for batch in batches
        }
        progress = tqdm(
            as_completed(future_to_batch), total=len(batches), desc="Hashing Batches"
        )
        for future in progress:
            batch_hashes, batch_valid_paths = future.result()
            if batch_hashes:
                hashes.extend(batch_hashes)
                valid_paths.extend(batch_valid_paths)

    if not hashes:
        logger.warning("Could not compute any valid hashes. Aborting.")
        return

    hashes_np = np.array(hashes, dtype=np.uint8)
    num_images, hash_dim_bytes = hashes_np.shape
    hash_dim_bits = hash_dim_bytes * 8
    logger.info(f"Computed {num_images} hashes of {hash_dim_bits}-bit.")

    # --- 2. Find Similar Pairs with Faiss ---
    logger.info("Building Faiss index to find similar pairs...")
    index = faiss.IndexBinaryFlat(hash_dim_bits)
    index.add(hashes_np)
    # range_search finds all items within the given radius (threshold)
    lims, _, I = index.range_search(hashes_np, threshold)
    logger.info(f"Found {len(I) - num_images} potential similarity links.")

    # --- 3. Group Images with Disjoint Set Union (DSU) ---
    logger.info("Grouping similar images using DSU...")
    dsu = DSU(num_images)
    for i in range(num_images):
        # Get neighbors of image i from the Faiss result
        neighbors = I[lims[i] : lims[i + 1]]
        for j in neighbors:
            if i != j:
                dsu.union(i, j)

    # Consolidate groups
    groups = defaultdict(list)
    for i in range(num_images):
        root = dsu.find(i)
        groups[root].append(i)

    # Filter out single-member groups (images with no duplicates)
    duplicate_groups = [g for g in groups.values() if len(g) > 1]
    logger.info(f"Identified {len(duplicate_groups)} groups of similar images.")

    # --- 4. Move Files ---
    logger.info("Moving grouped images to output directory...")
    for group in tqdm(duplicate_groups, desc="Moving Files"):
        # Use the ID of the first image as the subfolder name
        first_image_path = valid_paths[group[0]]
        subfolder_name = first_image_path.stem

        for idx in group:
            src_path = valid_paths[idx]
            try:
                # Preserve relative path inside the new subfolder
                relative_path = src_path.relative_to(root_path)
                dest_path = output_path / subfolder_name / relative_path

                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_path), str(dest_path))

                # Move corresponding prompt file if it exists
                prompt_src = src_path.with_suffix(".txt")
                if prompt_src.exists():
                    shutil.move(str(prompt_src), str(dest_path.with_suffix(".txt")))
            except FileNotFoundError:
                logger.warning(f"File not found (already moved?): {src_path}")
            except Exception as e:
                logger.error(f"Failed to move {src_path}: {e}")

    logger.info("Image deduplication process complete.")
