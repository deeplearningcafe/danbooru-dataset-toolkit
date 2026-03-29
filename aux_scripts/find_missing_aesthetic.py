import os
import yaml
import pandas as pd
import glob


def find_missing_aesthetic(
    download_dir: str,
    aesthetic_csv: str,
    output_file: str = "missing_aesthetic_samples.txt",
):
    """
    Scans the download directory for images and compares them against
    the IDs present in the aesthetic CSV to find missing entries.
    """
    if not os.path.exists(download_dir):
        print(f"Error: Download directory '{download_dir}' not found.")
        return

    if not os.path.exists(aesthetic_csv):
        print(f"Error: Aesthetic CSV '{aesthetic_csv}' not found.")
        return

    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    image_paths = []

    print(f"Scanning '{download_dir}' for images...")
    for root, _, files in os.walk(download_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))

    print(f"Found {len(image_paths)} images in the download directory.")

    try:
        # Load the CSV, forcing 'id' to string to match image filenames
        df = pd.read_csv(aesthetic_csv, dtype={"id": str})
        aes_ids = set(df["id"].unique())
        print(f"Loaded {len(aes_ids)} unique IDs from aesthetic CSV.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    missing_paths = []
    for path in image_paths:
        img_id = os.path.splitext(os.path.basename(path))[0]
        if img_id not in aes_ids:
            missing_paths.append(path)

    print(f"Found {len(missing_paths)} images missing aesthetic labels.")

    if missing_paths:
        with open(output_file, "w") as f:
            for p in missing_paths:
                f.write(p + "\n")
        print(f"Saved missing paths to '{output_file}'")


if __name__ == "__main__":
    config_path = "configs/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    download_dir = config.get("download_dir")
    experiment_name = config.get("experiment_name", "experiment")

    # The pipeline dynamically creates a reports directory with the date.
    # We find the most recent reports directory for this experiment.
    report_dirs = glob.glob(f"reports/{experiment_name}_*")
    if not report_dirs:
        print(f"No reports directory found for '{experiment_name}'.")
        exit(1)

    latest_report_dir = sorted(report_dirs)[-1]

    # Construct the path to the aesthetic labels CSV
    csv_filename = os.path.basename(
        config.get("prompts", {}).get("final_tiers_csv", "final_tiers.csv")
    )
    aesthetic_csv = os.path.join(latest_report_dir, csv_filename)

    print(f"Using download directory: {download_dir}")
    print(f"Using aesthetic CSV: {aesthetic_csv}")
    output_file = os.path.join(latest_report_dir, "missing_aesthetic_samples.txt")

    find_missing_aesthetic(download_dir, aesthetic_csv, output_file)
