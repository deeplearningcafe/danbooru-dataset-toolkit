import os
import csv
import torch
from aesthetic.training.dataset.image_dataset import resize_with_padding
from aesthetic.training.models.tagger import load_feature_extractor
from aesthetic.training.models.four_cls import AestheticClassifier
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    is_scaled_image,
)
from transformers.image_transforms import (
    rescale,
    normalize,
)
from PIL import Image
import numpy as np
from tqdm import tqdm


class ImageClassifier:
    """
    Manages loading the aesthetic model and classifying images in a directory.
    """

    def __init__(self, batch_size: int = 32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.feature_extractor = None
        self.classifier = None
        print(f"Using device: {self.device}")

    def load_models(self, feature_extractor_path: str, classifier_path: str):
        """Loads the feature extractor and the aesthetic classifier."""
        print("Loading models...")
        self.feature_extractor, _, _ = load_feature_extractor(feature_extractor_path)
        self.feature_extractor.eval().to(self.device)

        checkpoint = torch.load(classifier_path, map_location=self.device)

        # Create a new state dict to hold the adjusted keys
        adjusted_state_dict = {}
        for key, value in checkpoint["model_state_dict"].items():
            if key.startswith("_orig_mod."):
                # Strip the prefix
                new_key = key[len("_orig_mod.") :]
                adjusted_state_dict[new_key] = value
            else:
                # Keep keys without the prefix as-is (e.g., if compile failed)
                adjusted_state_dict[key] = value

        # ... (logic from original load_model function)
        self.classifier = AestheticClassifier(
            feature_dim=checkpoint["feature_dim"],
            num_classes=checkpoint["num_classes"],
            hidden_dims=checkpoint.get("hidden_dims", 512),
            dropout_rate=checkpoint.get("dropout_rate", 0.1),
        )
        self.classifier.load_state_dict(adjusted_state_dict)
        self.classifier.eval().to(self.device)
        print("Models loaded successfully.")

    def _preprocess_image_for_batch(self, image_path: str):
        """
        Modified preprocessing function that returns tensor without batch dimension
        for batching multiple images together.
        """
        processor_config = {
            "size": {"height": 448, "width": 448},
            "color": [255, 255, 255],
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "rescale_factor": 1 / 255.0,
            "resample": PILImageResampling.BILINEAR,
        }

        size_dict = processor_config["size"]
        color_tuple = tuple(processor_config["color"])
        image_mean = processor_config["image_mean"]
        image_std = processor_config["image_std"]
        rescale_factor = processor_config["rescale_factor"]
        resample_filter = processor_config["resample"]

        try:
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image)
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return None

        # Preprocessing pipeline
        processed_image = resize_with_padding(
            image=image_array,
            size=(size_dict["height"], size_dict["width"]),
            color=color_tuple,
            resample=resample_filter,
            data_format=ChannelDimension.FIRST,
        )

        if not is_scaled_image(processed_image):
            processed_image = rescale(
                processed_image,
                scale=rescale_factor,
                data_format=ChannelDimension.FIRST,
            )

        processed_image = normalize(
            processed_image,
            mean=image_mean,
            std=image_std,
            data_format=ChannelDimension.FIRST,
        )

        # Return tensor without batch dimension for batching
        return torch.tensor(processed_image).float()

    @torch.no_grad()
    def _classify_batch(self, img_paths: list):
        """
        Classify a batch of images efficiently on GPU.
        Returns list of labels (or None for failed images).
        """
        idx2label = {0: "worst", 1: "worse", 2: "better", 3: "best"}

        # Preprocess all images in the batch
        batch_tensors = []
        valid_indices = []

        for i, img_path in enumerate(img_paths):
            tensor = self._preprocess_image_for_batch(img_path)
            if tensor is not None:
                batch_tensors.append(tensor)
                valid_indices.append(i)

        if not batch_tensors:
            return [None] * len(img_paths)

        try:
            # Stack tensors into batch and move to GPU
            batch_tensor = torch.stack(batch_tensors).to(self.device)

            # Forward pass through feature extractor
            features = self.feature_extractor.forward_features(batch_tensor)
            pooled_features = self.feature_extractor.head.global_pool(features)

            # Forward pass through classifier
            logits = self.classifier(pooled_features)
            predictions = torch.argmax(logits, dim=1)

            # Convert predictions to labels
            batch_labels = [idx2label[pred.item()] for pred in predictions]

            # Map results back to original order
            results = [None] * len(img_paths)
            for i, valid_idx in enumerate(valid_indices):
                results[valid_idx] = batch_labels[i]

            return results

        except Exception as e:
            print(f"Error in batch classification: {e}")
            return [None] * len(img_paths)

    def label_images_in_directory(self, root_dir: str, output_csv_path: str):
        """
        Walk through directory structure, classify all images in batches, and save
        results to CSV.

        Args:
            root_dir: Root directory containing images
            output_csv_path: Path for output CSV file
            feature_extractor: Loaded feature extractor model
            classifier: Loaded aesthetic classifier model
            batch_size: Number of images to process in each batch
        """
        if not self.feature_extractor or not self.classifier:
            raise RuntimeError("Models are not loaded. Call load_models() first.")

        # Common image extensions to process
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

        # Collect all image paths first
        all_image_paths = []
        print(f"Scanning for images in: {root_dir}")

        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, root_dir)
                    all_image_paths.append((full_path, relative_path))

        print(f"Found {len(all_image_paths)} images to process")

        results = []
        total_processed = 0
        total_failed = 0

        # Process images in batches
        for i in tqdm(range(0, len(all_image_paths), self.batch_size)):
            batch_paths = all_image_paths[i : i + self.batch_size]
            batch_full_paths = [path[0] for path in batch_paths]
            batch_relative_paths = [path[1] for path in batch_paths]

            print(
                f"Processing batch {i // self.batch_size + 1}/{(len(all_image_paths) + self.batch_size - 1) // self.batch_size}"
            )

            # Classify the entire batch
            batch_labels = self._classify_batch(batch_full_paths)

            # Collect results
            for rel_path, label in zip(batch_relative_paths, batch_labels):
                if label is not None:
                    results.append([rel_path, label])
                    total_processed += 1
                else:
                    total_failed += 1

            # Clear GPU cache periodically to prevent memory buildup
            if total_processed % 50 == 0:
                torch.cuda.empty_cache()

            print(f"Processed {total_processed} images, Failed: {total_failed}")

        # Write results to CSV file
        print(f"Writing results to: {output_csv_path}")
        with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(["relative_path", "aesthetic_label"])
            # Write all results
            writer.writerows(results)

        print(f"Labeling complete!")
        print(f"Total images processed: {total_processed}")
        print(f"Total images failed: {total_failed}")
        print(f"Results saved to: {output_csv_path}")
