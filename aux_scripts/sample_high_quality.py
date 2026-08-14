import os
import shutil
import argparse
import pandas as pd
import yaml
from pathlib import Path
from src.utils.loader import resolve_image_path


def sample_high_quality(
    tiers_csv: str, labels_csv: str, output_dir: str, num_samples: int
):
    """
    Samples high quality images (masterpiece, good_score) and copies
    them to a specified directory for analysis.
    """
    config_path = "configs/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("sampling", {}).get("random_seed", 42)
    print(f"Using random seed from config: {seed}")

    print(f"Loading {tiers_csv} and {labels_csv}...")
    df_tiers = pd.read_csv(tiers_csv, header=0)
    df_labels = pd.read_csv(labels_csv, header=0)

    # Parse global paths to create booru_id
    def extract_id(path_str):
        try:
            return int(Path(str(path_str)).stem)
        except ValueError:
            return -1

    df_labels["id"] = df_labels["relative_path"].apply(extract_id)

    # Left join to get global paths and final tiers together
    df_merged = pd.merge(df_labels, df_tiers, on="id", how="left")

    target_tiers = ["masterpiece", "good_score"]
    out_path = Path(output_dir)

    copied_ids = set()

    for tier in target_tiers:
        df_tier = df_merged[df_merged["final_tier"] == tier]
        if df_tier.empty:
            print(f"No samples found for tier: {tier}")
            continue

        n_samples = min(num_samples, len(df_tier))
        df_sampled = df_tier.sample(n=n_samples, random_state=seed)

        tier_dir = out_path / tier
        tier_dir.mkdir(parents=True, exist_ok=True)

        print(f"Copying {n_samples} samples for '{tier}' to {tier_dir}...")

        copied = 0
        for _, row in df_sampled.iterrows():
            raw_path = Path(str(row["relative_path"]))
            src_file = resolve_image_path(raw_path)
            img_id = row["id"]
            if src_file.exists():
                shutil.copy2(src_file, tier_dir / src_file.name)
                copied += 1
                copied_ids.add(img_id)
            else:
                print(f"File not found: {src_file}")

        print(f"Successfully copied {copied} files for '{tier}'.\n")

    print("Processing 'best' aesthetic label tier...")
    df_best = df_merged[
        (df_merged["aesthetic_label"] == "best") & (~df_merged["id"].isin(copied_ids))
    ]

    if df_best.empty:
        print("No new samples found for aesthetic label 'best'.")
    else:
        n_samples = min(num_samples, len(df_best))
        df_sampled = df_best.sample(n=n_samples, random_state=seed)

        best_dir = out_path / "best"
        best_dir.mkdir(parents=True, exist_ok=True)

        print(f"Copying {n_samples} samples for 'best' to {best_dir}...")
        copied = 0
        for _, row in df_sampled.iterrows():
            raw_path = Path(str(row["relative_path"]))
            src_file = resolve_image_path(raw_path)
            if src_file.exists():
                shutil.copy2(src_file, best_dir / src_file.name)
                copied += 1
                copied_ids.add(row["id"])
            else:
                print(f"File not found: {src_file}")

        print(f"Successfully copied {copied} files for 'best'.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample high quality images for visual analysis."
    )
    parser.add_argument(
        "--tiers_csv", required=True, help="Path to the final_tiers.csv"
    )
    parser.add_argument(
        "--labels_csv", required=True, help="Path to image_aesthetic_labels.csv"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to save the samples"
    )
    parser.add_argument("--num_samples", type=int, default=64, help="Samples per tier")

    args = parser.parse_args()
    sample_high_quality(
        args.tiers_csv, args.labels_csv, args.output_dir, args.num_samples
    )
