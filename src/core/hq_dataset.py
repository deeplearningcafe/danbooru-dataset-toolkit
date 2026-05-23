import shutil
import pandas as pd
from pathlib import Path
from typing import List


class HQDatasetPreparer:
    """
    Prepares a high-quality dataset by copying images, and their already
    computed .txt and .json prompt files to a temporary directory.
    Tracks samples that are missing their prompt files.
    """

    def __init__(self, temp_dir: str):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.missing_prompts: List[str] = []

    def prepare(self, csv_path: str) -> None:
        print(f"Preparing HQ dataset from {csv_path} into {self.temp_dir}")
        df_hq = pd.read_csv(csv_path)

        if "image_path" not in df_hq.columns:
            raise ValueError("CSV must contain 'image_path' column.")

        copied_count = 0
        for _, row in df_hq.iterrows():
            src_image = Path(str(row["image_path"]))
            if not src_image.exists():
                print(f"Image not found: {src_image}")
                continue

            src_txt = src_image.with_suffix(".txt")
            src_json = src_image.with_suffix(".json")

            # Track samples missing either text or json prompt files
            if not src_txt.exists() or not src_json.exists():
                self.missing_prompts.append(str(src_image))
                continue

            dst_image = self.temp_dir / src_image.name
            dst_txt = self.temp_dir / src_txt.name
            dst_json = self.temp_dir / src_json.name

            shutil.copy2(src_image, dst_image)
            shutil.copy2(src_txt, dst_txt)
            shutil.copy2(src_json, dst_json)

            copied_count += 1

        print(f"Prepared {copied_count} HQ samples in {self.temp_dir}")
        if self.missing_prompts:
            print(
                f"Warning: {len(self.missing_prompts)} samples were "
                "missing prompt files and were skipped."
            )
