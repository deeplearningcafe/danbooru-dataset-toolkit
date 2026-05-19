import os
import sys
import yaml
import pandas as pd
import glob
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.download import Downloader
from src.core.prompt_generator import PromptGenerator
from src.prompts.prompt_utils import format_danbooru_tag_inverse

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def scan_directory_for_images(dir_path: str) -> set:
    """Scans a directory recursively for images and returns a set of IDs."""
    found_ids = set()
    if not os.path.isdir(dir_path):
        return found_ids
    for root, _, files in os.walk(dir_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                file_id_str = os.path.splitext(file)[0]
                try:
                    found_ids.add(int(file_id_str))
                except ValueError:
                    pass
    return found_ids


def redownload_missing_latents():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    experiment_name = config.get("experiment_name", "experiment")
    report_dirs = glob.glob(f"reports/{experiment_name}_*")
    if not report_dirs:
        print(f"No reports directory found for '{experiment_name}'.")
        return
    latest_report_dir = sorted(report_dirs)[-1]

    download_dir = config.get("download_dir")

    pu_config = config.get("prompt_upsampling", {})
    upsampled_csv = os.path.join(
        latest_report_dir,
        os.path.basename(pu_config.get("upsampled_tags_path", "upsampled.csv")),
    )
    print("Generating prompts for the redownloaded images...")
    if os.path.exists(upsampled_csv):
        up_df = pd.read_csv(upsampled_csv)
    else:
        up_df = pd.DataFrame()

    s_config = config.get("sampling", {})
    sampled_csv = os.path.join(
        latest_report_dir,
        os.path.basename(s_config.get("sampled_ids_csv", "sampled_ids.csv")),
    )

    d_config = config.get("download", {})
    exclude_csv = os.path.join(
        latest_report_dir,
        os.path.basename(d_config.get("exclusion_list_csv", "exclude_ids.csv")),
    )

    print(f"Reading sampled IDs from {sampled_csv}...")
    sampled_df = pd.read_csv(sampled_csv, low_memory=False)

    expected_ids = set(up_df["id"].astype(int))
    print(f"Found {len(expected_ids)} expected images")

    print("Scanning download directory for existing images...")
    downloaded_ids = scan_directory_for_images(download_dir)

    missing_ids = expected_ids - downloaded_ids
    print(f"Missing IDs to redownload: {len(missing_ids)}")

    if not missing_ids:
        return

    missing_df = sampled_df[sampled_df["id"].astype(int).isin(missing_ids)].copy()

    missing_df["quality_tier"] = missing_df["quality_tier"].fillna("good_score")

    if os.path.exists(exclude_csv):
        exclude_df = pd.read_csv(exclude_csv)
        initial_len = len(exclude_df)
        exclude_df = exclude_df[~exclude_df["id"].astype(int).isin(missing_ids)]
        exclude_df.to_csv(exclude_csv, index=False)
        print(f"Removed {initial_len - len(exclude_df)} missing IDs from {exclude_csv}")

    downloader = Downloader(
        max_workers=config["download"].get("max_workers", 8),
        timeout=config["download"].get("timeout", 10),
        max_downloads=None,
    )

    dummy_csv = os.path.join(latest_report_dir, "redownloaded_missing.csv")

    print("Downloading missing images...")
    downloader.download_images(
        df=missing_df,
        output_dir=download_dir,
        csv_path=dummy_csv,
        output_csv_path=exclude_csv,
        start_index=0,
        character_list=[],
    )

    print("Redownload complete.")


if __name__ == "__main__":
    redownload_missing_latents()
