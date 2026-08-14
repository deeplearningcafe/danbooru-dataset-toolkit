import os
import shutil
import yaml
import pandas as pd
import argparse
import glob
from pathlib import Path
import concurrent.futures

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
QUALITY2ID = {"masterpiece": 3, "good_score": 2, "bad_score": 1, "worse_score": 0}


def scan_directory(dir_path: str) -> dict:
    """
    Scans a directory for images and returns a dict mapping id to path.
    Designed to run in a worker thread.
    """
    id_to_path = {}
    if not os.path.exists(dir_path):
        return id_to_path

    for root, _, files in os.walk(dir_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                file_id_str = os.path.splitext(file)[0]
                try:
                    file_id = int(file_id_str)
                    id_to_path[file_id] = os.path.join(root, file)
                except ValueError:
                    pass
    return id_to_path


def extract_id(path_str: str) -> int:
    """Safely extracts the integer ID from a file path."""
    try:
        return int(os.path.splitext(os.path.basename(str(path_str)))[0])
    except (ValueError, TypeError):
        return -1


def copy_previous_samples(
    old_dir: str, old_tiers_csv: str, old_aes_csv: str, old_up_csv: str
):
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    experiment_name = config.get("experiment_name", "experiment")
    report_dirs = glob.glob(f"reports/{experiment_name}_*")
    if not report_dirs:
        print(f"No reports directory found for '{experiment_name}'.")
        return
    latest_report_dir = sorted(report_dirs)[-1]

    new_download_dir = config.get("download_dir")

    s_config = config.get("sampling", {})
    new_sampled_csv = os.path.join(
        latest_report_dir,
        os.path.basename(s_config.get("sampled_ids_csv", "sampled_ids.csv")),
    )

    c_config = config.get("classification", {})
    new_aes_csv = os.path.join(
        latest_report_dir,
        os.path.basename(c_config.get("aesthetic_labels_csv", "labels.csv")),
    )

    pu_config = config.get("prompt_upsampling", {})
    new_up_csv = os.path.join(
        latest_report_dir,
        os.path.basename(pu_config.get("upsampled_tags_path", "upsampled.csv")),
    )

    if not os.path.exists(new_sampled_csv):
        print(f"Sampled IDs CSV not found: {new_sampled_csv}")
        return
    sampled_df = pd.read_csv(new_sampled_csv)
    to_download_ids = set(sampled_df["id"].astype(int))

    already_downloaded = set()
    if os.path.exists(new_up_csv):
        up_df = pd.read_csv(new_up_csv)
        if "id" in up_df.columns:
            already_downloaded.update(up_df["id"].astype(int))

    if os.path.exists(new_aes_csv):
        aes_df = pd.read_csv(new_aes_csv)
        if "id" in aes_df.columns:
            already_downloaded.update(aes_df["id"].astype(int))
        elif "relative_path" in aes_df.columns:
            ids = aes_df["relative_path"].apply(extract_id)
            already_downloaded.update(ids[ids != -1])

    ids_to_copy = to_download_ids - already_downloaded
    print(f"Target IDs: {len(to_download_ids)}")
    print(f"Already processed: {len(already_downloaded)}")
    print(f"Need to copy: {len(ids_to_copy)}")

    if not ids_to_copy:
        print("Nothing to copy.")
        return

    old_paths_dict = {}
    folders_to_scan = [os.path.join(old_dir, str(i)) for i in range(4)]

    print("Scanning old directory...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(scan_directory, folder) for folder in folders_to_scan
        ]
        for future in concurrent.futures.as_completed(futures):
            old_paths_dict.update(future.result())

    old_tiers_df = pd.read_csv(old_tiers_csv)
    old_tiers_dict = dict(
        zip(old_tiers_df["id"].astype(int), old_tiers_df["final_tier"])
    )

    copied_ids = set()
    print(f"Copying files... from the {len(old_paths_dict)} possible images")

    for i, img_id in enumerate(ids_to_copy):
        if img_id in old_paths_dict:
            old_img_path = Path(old_paths_dict[img_id])
            tier = old_tiers_dict.get(img_id, "bad_score")
            class_id = QUALITY2ID.get(tier, 0)

            dest_folder = Path(new_download_dir) / str(class_id)
            dest_folder.mkdir(parents=True, exist_ok=True)

            dest_img_path = dest_folder / old_img_path.name
            if dest_img_path.exists():
                continue
            shutil.copy2(old_img_path, dest_img_path)

            old_txt_path = old_img_path.with_suffix(".txt")
            new_txt_path = dest_folder / old_txt_path.name
            if old_txt_path.exists() and not new_txt_path.exists():
                shutil.copy2(old_txt_path, dest_folder / old_txt_path.name)

            old_up_name = f"{old_img_path.stem}_upsampled.txt"
            old_up_path = old_img_path.parent / old_up_name
            new_up_path = dest_folder / old_up_name
            if old_up_path.exists() and not new_up_path.exists():
                shutil.copy2(old_up_path, dest_folder / old_up_name)

            copied_ids.add(img_id)

        if i % 10000 == 0:
            print(f"Processed {i} images!")

    print(f"Copied {len(copied_ids)} samples.")

    if copied_ids:
        print("Updating CSVs...")
        if os.path.exists(old_up_csv):
            old_up_df = pd.read_csv(old_up_csv)
            mask = old_up_df["id"].astype(int).isin(copied_ids)
            copied_up_df = old_up_df[mask]

            if not copied_up_df.empty:
                up_mode = "a" if os.path.exists(new_up_csv) else "w"
                copied_up_df.to_csv(
                    new_up_csv, mode=up_mode, header=(up_mode == "w"), index=False
                )

        # Update Aesthetic labels
        if os.path.exists(old_aes_csv):
            old_aes_df = pd.read_csv(old_aes_csv)

            if "id" in old_aes_df.columns:
                mask = old_aes_df["id"].astype(int).isin(copied_ids)
            elif "relative_path" in old_aes_df.columns:
                extracted_ids = old_aes_df["relative_path"].apply(extract_id)
                mask = extracted_ids.isin(copied_ids)
            else:
                mask = pd.Series(False, index=old_aes_df.index)

            copied_aes_df = old_aes_df[mask].copy()

            if "relative_path" in copied_aes_df.columns:

                def update_rel_path(row):
                    img_id = (
                        row["id"] if "id" in row else extract_id(row["relative_path"])
                    )
                    tier = old_tiers_dict.get(img_id, "bad_score")
                    class_id = QUALITY2ID.get(tier, 0)
                    old_name = os.path.basename(str(row["relative_path"]))
                    return f"{class_id}/{old_name}"

                copied_aes_df["relative_path"] = copied_aes_df.apply(
                    update_rel_path, axis=1
                )

            if not copied_aes_df.empty:
                aes_mode = "a" if os.path.exists(new_aes_csv) else "w"
                copied_aes_df.to_csv(
                    new_aes_csv, mode=aes_mode, header=(aes_mode == "w"), index=False
                )

    print("Finished copying previous samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy previous samples to avoid redownloading."
    )
    parser.add_argument("--old_dir", required=True, help="Old dataset dir")
    parser.add_argument("--old_tiers", required=True, help="Old tiers CSV")
    parser.add_argument("--old_aes", required=True, help="Old aesthetic CSV")
    parser.add_argument("--old_up", required=True, help="Old upsampled CSV")
    args = parser.parse_args()

    copy_previous_samples(args.old_dir, args.old_tiers, args.old_aes, args.old_up)
