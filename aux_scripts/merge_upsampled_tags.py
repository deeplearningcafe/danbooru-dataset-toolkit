import os
import yaml
from pathlib import Path


def merge_upsampled_tags(target_dir: str, suffix: str = "_upsampled"):
    """
    Scans the target directory for files ending with the specified suffix
    (e.g., '_upsampled.txt'), copies their contents to the corresponding
    original '.txt' file, and then deletes the upsampled file.

    Args:
        target_dir (str): The directory to scan for text files.
        suffix (str): The suffix appended to the upsampled text files.
    """
    target_path = Path(target_dir)
    if not target_path.exists():
        print(f"Error: Directory '{target_dir}' not found.")
        return

    print(f"\n--- Starting tag merge process in '{target_dir}' ---")

    # Use rglob to recursively find all files matching the suffix
    search_pattern = f"*{suffix}.txt"
    upsampled_files = list(target_path.rglob(search_pattern))

    if not upsampled_files:
        print(f"No '{search_pattern}' files found. Nothing to merge.")
        return

    print(f"Found {len(upsampled_files)} upsampled files. Merging...")

    processed_count = 0
    error_count = 0

    for upsampled_file in upsampled_files:
        try:
            # Determine the original file path by removing the suffix
            # e.g., "1234_upsampled.txt" -> "1234.txt"
            original_filename = upsampled_file.name.replace(f"{suffix}.txt", ".txt")
            original_file = upsampled_file.with_name(original_filename)

            # Read the upsampled tags
            with open(upsampled_file, "r", encoding="utf-8") as f:
                upsampled_tags = f.read()

            # Write the tags to the original file (overwriting the empty one)
            with open(original_file, "w", encoding="utf-8") as f:
                f.write(upsampled_tags)

            # Remove the upsampled file to clean up the directory
            upsampled_file.unlink()

            processed_count += 1

            # Print progress every 10,000 files to avoid console spam
            if processed_count % 10000 == 0:
                print(f"  ...processed {processed_count} files.")

        except Exception as e:
            print(f"Error processing {upsampled_file}: {e}")
            error_count += 1

    print(f"\n--- Merge Complete ---")
    print(f"Successfully merged and deleted {processed_count} files.")
    if error_count > 0:
        print(f"Encountered errors on {error_count} files.")


if __name__ == "__main__":
    config_path = "configs/default_config.yaml"

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at '{config_path}'.")
        exit(1)

    # Determine the target directory. It defaults to the face cropping output
    # directory, but falls back to the main download directory if not found.
    download_dir = config.get("download_dir", "StableDifussion/")
    default_faces_dir = f"{download_dir}_faces"

    faces_dir = config.get("face_cropping", {}).get("output_dir", default_faces_dir)

    # You can easily change this variable if you want to run it on the base download_dir
    target_directory = faces_dir

    print(f"Target directory resolved to: {target_directory}")

    # Run the merge
    merge_upsampled_tags(target_dir=target_directory)
