import csv
from pathlib import Path
from typing import List
from typing import Optional, Callable, Generator
import yaml

def dirwalk(
    path: Path,
    condition: Optional[Callable] = None
) -> Generator[Path, None, None]:
    """Walk through directory and yield files that meet the condition."""
    for p in path.iterdir():
        if p.is_dir():
            yield from dirwalk(p, condition)
        elif condition is None or condition(p):
            yield p


def create_upsampled_tags_csv(
    directory: Path,
    output_csv: Path
) -> None:
    """
    Reads all _upsampled.txt files within a directory, extracts the
    upsampled tags that appear after the year, and compiles them into a
    single CSV file.

    Args:
        directory (Path): The path to the directory containing the
                          _upsampled.txt files.
        output_csv (Path): The path where the output CSV file will be
                           saved.
    """
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the header of the CSV file.
        csv_writer.writerow(['id', 'upsampled_tags'])

        # Define a condition to filter for the upsampled text files.
        def is_upsampled_file(p: Path) -> bool:
            return p.name.endswith('_upsampled.txt')

        # Iterate through all files in the directory that match the
        # condition.
        for file_path in dirwalk(directory, is_upsampled_file):
            # The ID is extracted from the filename by removing the suffix.
            file_id = file_path.stem.replace('_upsampled', '')

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # Split the content into a list of tags.
                tags: List[str] = [
                    tag.strip() for tag in content.split(',')
                ]

                # Find the index of the year tag to separate original and
                # upsampled tags.
                year_index = -1
                for i, tag in enumerate(tags):
                    if tag in ('2023', '2024'):
                        year_index = i
                        break

                # If a year is found, the upsampled tags are all the tags
                # that follow.
                if year_index != -1:
                    upsampled_tags: str = ', '.join(
                        tags[year_index + 1:]
                    )
                    csv_writer.writerow([file_id, upsampled_tags])
if __name__ == "__main__":
    config_path = "configs/default_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    directory = Path(config['data_root'])
    output_csv = Path(config['prompt_upsampling']['upsampled_tags_path'])
    create_upsampled_tags_csv(directory, output_csv)