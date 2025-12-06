import os
import shutil
from pathlib import Path
from typing import Dict
import yaml
# valid image extensions to look for
valid_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

def generate_curated_id_list(cleaned_folder_path: str, output_txt_path: str):
    """
    Reads images from the cleaned folder and writes their IDs (filenames
    without extensions) to a text file.
    """
    cleaned_path = Path(cleaned_folder_path)

    ids = []

    # Iterate over files in the directory
    if not cleaned_path.exists():
        print(f"Error: Source folder {cleaned_path} does not exist.")
        return

    for file_path in cleaned_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in valid_exts:
            # stem returns filename without extension (e.g. '123.jpg' -> '123')
            ids.append(file_path.stem)

    # Sort for consistency
    ids.sort()

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for image_id in ids:
            f.write(f"{image_id}\n")

    print(f"Successfully wrote {len(ids)} IDs to {output_txt_path}")


def restore_curated_dataset(character_root_path: str, id_list_path: str):
    """
    Reads a list of IDs, finds the corresponding images and text files
    in the subfolders (0, 1, 2, etc.) of the character root, and copies
    them to a 'cleaned' folder.
    """
    root_path = Path(character_root_path)
    list_path = Path(id_list_path)
    dest_path = root_path / "cleaned"

    if not list_path.exists():
        print(f"Error: ID list file {list_path} not found.")
        return

    # Create destination folder if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)

    # 1. Build a map of ID -> File Path for the full dataset.
    # This avoids traversing the directory tree for every single ID.
    print("Indexing full dataset...")
    id_to_path_map: Dict[str, Path] = {}

    # Recursively find all files, excluding the destination 'cleaned' folder
    for file_path in root_path.rglob('*'):
        if file_path.is_file() and 'cleaned' not in file_path.parts and file_path.suffix.lower() in valid_exts:
            # We map the stem (ID) to the full path
            # If duplicates exist across folders, the last one found wins
            id_to_path_map[file_path.stem] = file_path

    # 2. Read the target IDs
    with open(list_path, 'r', encoding='utf-8') as f:
        target_ids = [line.strip() for line in f if line.strip()]

    copied_count = 0
    missing_count = 0

    print(f"Starting copy process for {len(target_ids)} samples...")

    # 3. Copy files
    for target_id in target_ids:
        # Check if the image exists in our map
        if target_id in id_to_path_map:
            img_src = id_to_path_map[target_id]

            # Construct expected text file path (same folder, same stem, .txt)
            txt_src = img_src.with_suffix('.txt')

            # Destination paths
            img_dest = dest_path / img_src.name
            txt_dest = dest_path / txt_src.name

            try:
                # Copy image
                shutil.copy2(img_src, img_dest)

                # Copy text prompt if it exists
                if txt_src.exists():
                    shutil.copy2(txt_src, txt_dest)
                else:
                    print(f"Warning: Prompt file missing for {target_id}")

                copied_count += 1
            except Exception as e:
                print(f"Error copying {target_id}: {e}")
        else:
            print(f"Missing: Image ID {target_id} not found in subfolders.")
            missing_count += 1

    print(f"Process complete. Copied: {copied_count}, Missing: {missing_count}")
    print(f"Curated dataset available at: {dest_path}")


if __name__ == "__main__":
    config_path = "configs/default_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    download_dir = config['download_dir']
    generate_curated_id_list(
        cleaned_folder_path=f"{download_dir}/yanami_anna/cleaned",
        output_txt_path="curated_yanami_anna.txt"
    )