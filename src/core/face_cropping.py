import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import logging
import pandas as pd
import os

logger = logging.getLogger(__name__)


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
    ):
        self.model = YOLO(model_path)
        self.min_res = min_res
        self.padding = padding
        self.top_padding = top_padding

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
                valid_rows.append((full_path, rel_path))

        logger.info(
            f"Found {len(valid_rows)} valid downloaded images for face cropping."
        )
        processed_count = 0

        for i in tqdm(range(0, len(valid_rows), batch_size), desc="Cropping"):
            batch_data = valid_rows[i : i + batch_size]
            batch_imgs = []
            batch_rel_paths = []

            for full_path, rel_path in batch_data:
                img = cv2.imread(str(full_path))
                if img is not None:
                    batch_imgs.append(img)
                    batch_rel_paths.append(rel_path)

            if not batch_imgs:
                continue

            # Run YOLO inference in batch mode
            results = self.model(batch_imgs, verbose=False)

            for idx, r in enumerate(results):
                boxes = r.boxes
                if len(boxes) == 0:
                    continue

                # Get the bounding box with the highest confidence
                best_box_idx = boxes.conf.argmax().item()
                best_box = boxes.xyxy[best_box_idx].cpu().numpy()

                x1, y1, x2, y2 = map(int, best_box)
                w_orig, h_orig = x2 - x1, y2 - y1

                img = batch_imgs[idx]
                img_h, img_w = img.shape[:2]

                # Add padding to preserve hair and increase usable samples
                pad_x = int(w_orig * self.padding)
                pad_y = int(h_orig * self.padding)
                top_pad = int(h_orig * self.top_padding)

                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - top_pad)
                x2 = min(img_w, x2 + pad_x)
                y2 = min(img_h, y2 + pad_y)

                w, h = x2 - x1, y2 - y1

                # Skip if the face is smaller than the target resolution
                # Evaluated after padding so more samples pass the threshold
                if w < self.min_res or h < self.min_res:
                    continue

                face_crop = img[y1:y2, x1:x2]

                # Resize the shortest edge to min_res to match aspect ratio
                if w < h:
                    new_w = self.min_res
                    new_h = int(h * (self.min_res / w))
                else:
                    new_h = self.min_res
                    new_w = int(w * (self.min_res / h))

                face_resized = cv2.resize(
                    face_crop, (new_w, new_h), interpolation=cv2.INTER_AREA
                )

                # Center crop to exact squared min_res x min_res
                start_x = (new_w - self.min_res) // 2
                start_y = (new_h - self.min_res) // 2
                final_crop = face_resized[
                    start_y : start_y + self.min_res, start_x : start_x + self.min_res
                ]

                # Save the cropped image
                out_file = output_path / batch_rel_paths[idx]
                out_file.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_file), final_crop)

                # Create an empty .txt file so the tagger can generate
                # fresh tags from scratch, avoiding hallucinated body parts.
                with open(out_file.with_suffix(".txt"), "w", encoding="utf-8") as f:
                    f.write("")

                processed_count += 1

        logger.info(f"Successfully cropped {processed_count} faces.")
