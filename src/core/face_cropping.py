import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import logging
import pandas as pd
import os

from src.prompts.prompt_utils import format_tag_string, RATINGS

logger = logging.getLogger(__name__)


class FacePromptBuilder:
    """
    Builds the initial prompt for face crops using prior knowledge.
    Follows the Single Responsibility Principle by isolating prompt logic.
    """

    def __init__(self, allowed_artists: list[str] = None):
        self.allowed_artists = set(allowed_artists) if allowed_artists else set()

    def build_prompt(self, row: pd.Series) -> str:
        general_tags = str(row.get("tag_string_general", "")).split()
        gender_tags = [t for t in general_tags if t in ["1girl", "1boy"]]
        gender_str = ", ".join(gender_tags)

        char_str = format_tag_string(str(row.get("tag_string_character", "")))

        copy_str = format_tag_string(str(row.get("tag_string_copyright", "")))

        artist_raw = str(row.get("tag_string_artist", "")).split()
        artist_tags = [t for t in artist_raw if t in self.allowed_artists]
        artist_str = format_tag_string(" ".join(artist_tags))

        rating_val = str(row.get("rating", "g"))
        rating_str = RATINGS.get(rating_val, "general")

        parts = [gender_str, char_str, copy_str, artist_str, rating_str]
        prompt = ", ".join(p for p in parts if p)

        return prompt


class FaceCropper:
    """
    Detects and crops anime faces using YOLOv8, ensuring a fixed
    squared resolution.
    """

    def __init__(
        self,
        model_path: str,
        min_res: int = 256,
        padding: float = 0.2,
        top_padding: float = 0.4,
        allowed_artists: list[str] = None,
    ):
        self.model = YOLO(model_path)
        self.min_res = min_res
        self.padding = padding
        self.top_padding = top_padding
        self.prompt_builder = FacePromptBuilder(allowed_artists)

    def process_dataframe(
        self, df: pd.DataFrame, input_dir: str, output_dir: str, batch_size: int = 16
    ):
        """
        Processes images listed in a DataFrame, cropping faces and saving them.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if "relative_path" not in df.columns:
            logger.error("DataFrame must contain 'relative_path' column.")
            return

        # Construct full paths and check existence
        valid_rows = []
        for _, row in df.iterrows():
            rel_path = str(row["relative_path"]).replace("\\", os.sep)
            full_path = input_path / rel_path
            if full_path.exists():
                valid_rows.append((full_path, rel_path, row))

        logger.info(
            f"Found {len(valid_rows)} valid downloaded images for face cropping."
        )
        processed_count = 0

        for i in tqdm(range(0, len(valid_rows), batch_size), desc="Cropping"):
            batch_data = valid_rows[i : i + batch_size]
            batch_imgs = []
            batch_rel_paths = []
            batch_rows = []

            for full_path, rel_path, row in batch_data:
                img = cv2.imread(str(full_path))
                if img is not None:
                    batch_imgs.append(img)
                    batch_rel_paths.append(rel_path)
                    batch_rows.append(row)

            if not batch_imgs:
                continue

            results = self.model(batch_imgs, verbose=False)

            for idx, r in enumerate(results):
                boxes = r.boxes
                if len(boxes) == 0:
                    continue

                best_box_idx = boxes.conf.argmax().item()
                best_box = boxes.xyxy[best_box_idx].cpu().numpy()

                x1, y1, x2, y2 = map(int, best_box)
                w_orig, h_orig = x2 - x1, y2 - y1

                img = batch_imgs[idx]
                img_h, img_w = img.shape[:2]

                is_close_up = (w_orig * h_orig) > 0.6 * (img_w * img_h)

                if is_close_up:
                    # Ignore padding and YOLO output, resize original image
                    new_w, new_h = 256, 256

                    if new_w != 256 or new_h != 256:
                        print(
                            f"Close-up with incorrect dims with path: {rel_path} and w:{new_w}, h:{new_h}"
                        )
                        continue

                    final_crop = cv2.resize(
                        img, (new_w, new_h), interpolation=cv2.INTER_AREA
                    )
                else:
                    # Add padding to preserve hair and increase usable samples
                    pad_x = int(w_orig * self.padding)
                    pad_y = int(h_orig * self.padding)
                    top_pad = int(h_orig * self.top_padding)

                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - top_pad)
                    x2 = min(img_w, x2 + pad_x)
                    y2 = min(img_h, y2 + pad_y)

                    w, h = x2 - x1, y2 - y1

                    if w < self.min_res or h < self.min_res:
                        print(
                            f"Crop with incorrect dims with path: {rel_path} and w:{w}, h:{h}"
                        )
                        continue

                    face_crop = img[y1:y2, x1:x2]

                    # Resize the shortest edge to min_res to match aspect ratio
                    if w < h:
                        new_w = self.min_res
                        new_h = int(h * (self.min_res / w))
                    else:
                        new_h = self.min_res
                        new_w = int(w * (self.min_res / h))

                    if new_w != 256 or new_h != 256:
                        print(
                            f"Image with incorrect dims with path: {rel_path} and w:{new_w}, h:{new_h}"
                        )
                        continue

                    face_resized = cv2.resize(
                        face_crop, (new_w, new_h), interpolation=cv2.INTER_AREA
                    )

                    start_x = (new_w - self.min_res) // 2
                    start_y = (new_h - self.min_res) // 2
                    final_crop = face_resized[
                        start_y : start_y + self.min_res,
                        start_x : start_x + self.min_res,
                    ]

                out_file = output_path / batch_rel_paths[idx]
                out_file.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_file), final_crop)

                # Create the prompt file using the prior knowledge builder
                row = batch_rows[idx]
                prompt = self.prompt_builder.build_prompt(row)

                with open(out_file.with_suffix(".txt"), "w", encoding="utf-8") as f:
                    f.write(prompt)

                processed_count += 1

        logger.info(f"Successfully cropped {processed_count} faces.")
