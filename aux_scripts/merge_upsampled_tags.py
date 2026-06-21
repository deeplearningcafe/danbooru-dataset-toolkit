import os
import yaml
from pathlib import Path

RATINGS = {"explicit", "questionable", "sensitive", "general"}


def merge_upsampled_tags(
    target_dir: str,
    suffix: str = "_upsampled",
    min_tag_samples: int = 50,
):
    """
    Scans the target directory for files ending with the specified suffix
    (e.g., '_upsampled.txt'), copies their contents to the corresponding
    original '.txt' file, and then deletes the upsampled file.
    Also generates a '_short.txt' prompt removing rare tags.

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

    tag_counts = {}
    file_to_tags = {}

    for upsampled_file in upsampled_files:
        try:
            original_filename = upsampled_file.name.replace(f"{suffix}.txt", ".txt")
            original_file = upsampled_file.with_name(original_filename)

            with open(upsampled_file, "r", encoding="utf-8") as f:
                upsampled_tags = f.read()

            upsampled_tags_list = [
                t.strip() for t in upsampled_tags.split(",") if t.strip()
            ]

            with open(original_file, "r", encoding="utf-8") as f:
                original_tags = f.read()

            original_tags_list = [
                t.strip() for t in original_tags.split(",") if t.strip()
            ]
            upsampled_tags_list = [
                tag
                for tag in upsampled_tags_list
                if (tag not in original_tags_list and tag not in RATINGS)
            ]
            # merge tags
            original_tags_list.extend(upsampled_tags_list)

            with open(original_file, "w", encoding="utf-8") as f:
                f.write(", ".join(original_tags_list))

            for tag in original_tags_list:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

            file_to_tags[original_file] = original_tags_list

            upsampled_file.unlink()

            processed_count += 1

            if processed_count % 10000 == 0:
                print(f"  ...processed {processed_count} files.")

        except Exception as e:
            print(f"Error processing {upsampled_file}: {e}")
            error_count += 1

    print(f"\n--- Generating Short Prompts ---")
    print(
        f"Filtering tags with < {min_tag_samples} occurrences from {len(tag_counts)} different tags..."
    )

    valid_tags = {tag for tag, count in tag_counts.items() if count >= min_tag_samples}
    print(
        f"Ended up with {len(valid_tags)} valid tags. Removed {len(tag_counts) - len(valid_tags)} tags."
    )

    # Second loop to generate id_short.txt for all files
    for original_file, tags in file_to_tags.items():
        try:
            short_tags = [tag for tag in tags if tag in valid_tags]
            short_file = original_file.with_name(f"{original_file.stem}_short.txt")
            with open(short_file, "w", encoding="utf-8") as f:
                f.write(", ".join(short_tags))
        except Exception as e:
            print(f"Error generating short prompt for {original_file}: {e}")
            error_count += 1

    print(f"\n--- Merge Complete ---")
    print(f"Successfully merged and deleted {processed_count} files.")
    if error_count > 0:
        print(f"Encountered errors on {error_count} files.")


if __name__ == "__main__":
    config_path = "configs/config.yaml"

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at '{config_path}'.")
        exit(1)

    download_dir = config.get("download_dir", "StableDifussion/")
    default_faces_dir = f"{download_dir}_faces"

    faces_dir = config.get("face_cropping", {}).get("output_dir", default_faces_dir)

    min_tag_samples = config.get("face_cropping", {}).get("min_tag_samples", 20)

    target_directory = faces_dir

    print(f"Target directory resolved to: {target_directory}")

    # Run the merge
    merge_upsampled_tags(
        target_dir=target_directory,
        min_tag_samples=min_tag_samples,
    )
