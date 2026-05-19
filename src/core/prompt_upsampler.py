import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
from typing import List, Tuple, Optional, Any, Generator, Union, Callable
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, default_collate
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    is_scaled_image,
)
from transformers.image_transforms import (
    rescale,
    normalize,
)
import random
from src.prompts.tagger import Tagger, resize_with_padding
from ..utils.loader import load_all_parquets, parallel_scan_images
from ..prompts.prompt_utils import validate_upsampled_batch

import logging
import sys
import argparse
import re
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Define type hints for clarity
Resolution = Tuple[int, int]
# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
DEFAULT_IMG_SIZE = (448, 448)
PERSON_COUNT_REGEX = re.compile(r"\b\d+\+?(?:boy|girl)s?\b")


class TaggerDataset(Dataset):
    """
    Dataset specifically for loading and preprocessing images for the Tagger.
    Loads images, preprocesses them according to tagger requirements,
    and returns the tensor along with the original prompt and image path.
    """

    def __init__(
        self,
        root: Union[str, Path],
        tagger_input_size: Tuple[int, int],  # e.g., (448, 448) H, W
        tagger_means: List[float],  # e.g., [0.5, 0.5, 0.5]
        tagger_stds: List[float],  # e.g., [0.5, 0.5, 0.5]
        label_ext: str = ".txt",
        global_path: Optional[str] = None,
        booru_df_path: Optional[str] = None,
        num_workers: int = 4,
        resample_filter=PILImageResampling.BILINEAR,
        # Add other necessary params like padding color if needed by resize
        padding_color: Tuple[int, int, int] = (255, 255, 255),
    ):
        """
        Initializes the dataset for the tagger.

        Args:
            root: Root directory of the dataset.
            tagger_input_size: Target (height, width) for the tagger model.
            tagger_means: Normalization means for the tagger.
            tagger_stds: Normalization standard deviations for the tagger.
            label_ext: Extension for prompt files.
            global_path: Optional prefix for image paths.
            resample_filter: Resampling filter for resizing.
            padding_color: RGB tuple for padding.
        """
        self.root = Path(root)
        self.label_ext = label_ext
        self.global_path = global_path
        self.tagger_input_size = tagger_input_size
        self.tagger_means = tagger_means
        self.tagger_stds = tagger_stds
        self.resample_filter = resample_filter
        self.padding_color = padding_color
        self.rescale_factor = 1 / 255.0  # Standard rescale

        # Find all image paths (reuse dirwalk or similar logic)
        # Consider reusing _filter_valid_images if applicable
        logger.info(f"Scanning for images in {self.root}...")
        # Replace the slow, single-threaded call with the new parallel one.
        self.image_paths = parallel_scan_images(self.root, num_workers=num_workers)
        # Optional: Add filtering for valid images here if needed
        logger.info(f"Found {len(self.image_paths)} potential image files.")

        self.booru_df = None
        if booru_df_path is not None:
            self.booru_df = load_all_parquets(booru_df_path)

    def __len__(self) -> int:
        return len(self.image_paths)

    def filter_processed_images(self, processed_ids: set):
        """
        Filters out image paths that have already been processed.

        Args:
            processed_ids (set): A set of image IDs (stems) that have already been processed.
        """
        initial_count = len(self.image_paths)
        self.image_paths = [p for p in self.image_paths if p.stem not in processed_ids]
        filtered_count = initial_count - len(self.image_paths)
        logger.info(
            f"Filtered {filtered_count} already processed images. {len(self.image_paths)} remaining."
        )

    def _preprocess_for_tagger(self, image: Image.Image) -> torch.Tensor:
        """Preprocesses a PIL image for the tagger model."""
        # Convert PIL image (RGB) to numpy array (float32)
        image_array = np.asarray(image, dtype=np.float32)

        # --- Tagger-Specific Preprocessing Pipeline ---
        # 1. Resize with padding and convert to BGR numpy array
        #    Output format should be channels_first for PyTorch
        #    Ensure resize_with_padding handles BGR conversion correctly!
        processed_image = resize_with_padding(
            image=image_array,
            size=self.tagger_input_size,  # Use tagger's size (H, W)
            color=self.padding_color,
            resample=self.resample_filter,
            data_format=ChannelDimension.FIRST,  # PyTorch expects channels first
        )

        # 2. Rescale values to [0, 1]
        #    resize_with_padding might handle this depending on implementation,
        #    but explicit rescale ensures it.
        if not is_scaled_image(processed_image):
            processed_image = rescale(
                processed_image,
                scale=self.rescale_factor,
                data_format=ChannelDimension.FIRST,
            )

        # 3. Normalize with tagger's mean and std
        processed_image = normalize(
            processed_image,
            mean=self.tagger_means,
            std=self.tagger_stds,
            data_format=ChannelDimension.FIRST,
        )

        # 4. Convert final processed numpy array to PyTorch tensor
        img_tensor = torch.tensor(processed_image).float()
        return img_tensor

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, str, Path]]:
        """
        Returns (tagger_image_tensor, prompt, img_path, metadata_dict) or None on error.
        """
        if idx >= self.__len__():
            raise IndexError("Dataset index out of range.")

        img_path = self.image_paths[idx]
        prompt = ""

        # Metadata dictionary to pass to validation
        metadata = {
            "tag_string_general": "",
            "tag_count_character": None,
            "original_prompt": "",
        }

        try:
            # Attempt to read the corresponding prompt file
            prompt_path = img_path.with_suffix(self.label_ext)
            if prompt_path.exists():
                with prompt_path.open("r", encoding="utf-8") as f:
                    prompt = f.read().strip()
            elif self.booru_df is not None:
                try:
                    # Assuming filename is ID
                    img_id = int(os.path.splitext(os.path.basename(img_path))[0])
                    row = self.booru_df[self.booru_df["id"] == img_id]
                    if not row.empty:
                        prompt = row["tag_string"].iloc[0]
                        # Populate metadata for validation
                        metadata["tag_string_general"] = row.get(
                            "tag_string_general", pd.Series([""])
                        ).iloc[0]
                        metadata["tag_count_character"] = row.get(
                            "tag_count_character", pd.Series([None])
                        ).iloc[0]
                except Exception:
                    pass  # Fallback to file reading
            else:
                logger.warning(
                    f"Prompt file not found for {img_path}, using empty prompt."
                )

        except Exception as e:
            logger.warning(
                f"Error reading prompt file {prompt_path}: {e}. Using empty prompt."
            )
            prompt = ""  # Use empty prompt on error

        metadata["original_prompt"] = prompt

        # Construct full image path if global_path is set
        img_path_open = img_path
        if self.global_path:
            # Ensure correct path separators if needed
            img_path_corrected = str(img_path).replace("\\", os.sep)
            img_path_open = os.path.join(self.global_path, img_path_corrected)
            img_path_open = Path(img_path_open)  # Convert back to Path if needed

        if not img_path_open.exists():
            logger.warning(f"Image file not found: {img_path_open}. Skipping.")
            return None  # Signal error

        try:
            # Load image using PIL
            image = Image.open(img_path_open).convert("RGB")

            # Preprocess the image specifically for the tagger
            image_tensor = self._preprocess_for_tagger(image)

            # Return tensor, original prompt, original relative path and metadata and metadata
            return image_tensor, prompt, img_path, metadata

        except (IOError, OSError, Image.DecompressionBombError) as e:
            logger.warning(
                f"Error loading/processing image {img_path_open}: {e}. Skipping."
            )
            return None  # Signal error
        except Exception as e:
            # Catch unexpected errors during preprocessing or loading
            logger.error(
                f"Unexpected error for image {img_path_open}: {e}", exc_info=True
            )
            return None  # Signal error


# Define the custom collate function
def custom_collate_with_paths(
    batch: List[Optional[Any]],
) -> Optional[Tuple[torch.Tensor, List[str], List[Path]]]:
    """
    Custom collate function that handles batches containing
    (Tensor, str, Path) tuples, filtering out None values.

    Args:
        batch: A list of items fetched from the Dataset. Each item is
               expected to be either None or a tuple like
               (image_tensor, prompt_string, image_path_object).

    Returns:
        A tuple containing:
        - A batch of image tensors (stacked).
        - A list of prompt strings.
        - A list of Path objects.
        Returns None if the batch is empty after filtering Nones or if
        an error occurs during collation.
    """
    # 1. Filter out None values (items where __getitem__ failed)
    #    This handles potential errors during image loading/processing.
    filtered_batch = [item for item in batch if item is not None]

    # 2. Handle empty batch after filtering
    #    If all items in the original batch failed, return None.
    if not filtered_batch:
        logger.warning("Received an empty batch after filtering None items.")
        return None

    # 3. Separate components and perform collation
    try:
        # Use zip(*) to transpose the list of tuples:
        # e.g., [(t1, s1, p1), (t2, s2, p2)] -> [(t1, t2), (s1, s2), (p1, p2)]
        # This groups tensors, prompts, and paths together.
        components = list(zip(*filtered_batch))

        # Unpack the components
        image_tensors = components[0]  # Tuple of tensors
        prompts = components[1]  # Tuple of strings
        img_paths = components[2]  # Tuple of Path objects
        metadata = components[3]  # Tuple of Dicts

        # Collate tensors using default_collate:
        # This stacks the individual tensors into a single batch tensor.
        collated_tensors = default_collate(image_tensors)

        # Prompts (strings) and img_paths (Path objects) are kept as lists.
        # default_collate would fail on Path objects, and returning a list
        # is the standard way to batch non-tensor data.
        collated_prompts = list(prompts)
        collated_paths = list(img_paths)
        collated_metadata = list(metadata)

        # Return the collated batch in the desired structure
        return collated_tensors, collated_prompts, collated_paths, collated_metadata

    except Exception as e:
        # Catch potential errors during the separation or collation process.
        logger.error(f"Error during custom collation: {e}", exc_info=True)
        # Log details about the batch structure if helpful for debugging
        if filtered_batch:
            first_item = filtered_batch[0]
            # Check if the first item is a tuple as expected
            if isinstance(first_item, tuple):
                item_types = [type(comp) for comp in first_item]
                logger.error(
                    f"Problematic batch structure (first item types): {item_types}"
                )
            else:
                logger.error(
                    f"Problematic batch structure (first item type): {type(first_item)}"
                )
        return None  # Return None on error


def combine_prompts_intelligently(
    original_prompt: str,
    new_tags: List[str],
) -> str:
    """
    Combines an original prompt with new tags, intelligently avoiding
    concept conflicts like multiple colors for the same item or different
    person counts.

    Args:
        original_prompt: The original comma-separated prompt string.
        new_tags: A list of new tags from the tagger (using underscores).

    Returns:
        A combined, cleaned prompt string.
    """
    # 1. Normalize original tags for robust comparison.
    #    We preserve the original list to maintain user's formatting.
    original_tags_list = [t.strip() for t in original_prompt.split(",") if t.strip()]
    # 2. Parse original tags ONCE to identify existing concepts and counts.
    #    This is far more efficient and robust than repeated checks.
    existing_concepts = set()
    has_person_count = False
    # Regex to precisely match tags like '1girl', '2boys', '10girls', etc.
    # person_count_regex = re.compile(r'^\d+boy|^\d+girl')

    for tag in original_tags_list:
        if PERSON_COUNT_REGEX.match(tag):
            has_person_count = True
        # An attribute tag typically has an underscore.
        if " " in tag:
            # The concept is the part after the last underscore.
            concept = tag.rsplit(" ", 1)[-1]
            existing_concepts.add(concept)
    # 3. Filter new tags to avoid conflicts and duplicates.
    tags_to_add = []
    for new_tag in new_tags:
        # Normalize the new tag for comparison.
        normalized_new_tag = new_tag.lower()

        # Skip if tag is already present in the original prompt.
        if normalized_new_tag in original_tags_list:
            continue

        # Check for person count conflict.
        if has_person_count and PERSON_COUNT_REGEX.match(normalized_new_tag):
            continue

        # Check for concept conflict (e.g., 'hair', 'eyes', 'skirt').
        if " " in normalized_new_tag:
            concept = normalized_new_tag.rsplit(" ", 1)[-1]
            if concept in existing_concepts:
                continue

        # If no conflicts, add the tag (formatted with spaces).
        tags_to_add.append(new_tag)  # .replace('_', ' '))

    # 4. Combine and return the final prompt.
    #    This preserves the original tags and appends the new, filtered ones.
    final_tags = original_tags_list + tags_to_add
    return ", ".join(filter(None, final_tags)), ", ".join(filter(None, tags_to_add))


def upsample_prompts_batch(
    dataset_root: str,
    tagger_model_path: str,
    output_csv_path: str,
    tagger_device: str = "cuda",
    tag_threshold: float = 0.35,
    batch_size: int = 32,
    num_workers: int = 4,
    output_suffix: str = "_upsampled",
    dataloader_prefetch_factor: Optional[int] = 2,
    dataloader_persistent_workers: bool = True,
    save_original_on_error: bool = False,
    global_path: Optional[str] = None,
    label_ext: str = ".txt",
    skip_rating: bool = False,
    only_general_tags: bool = False,
    booru_df_path: Optional[str] = None,
):
    """
    Upsamples prompts for images using a tagger model via batch processing.

    Creates a dedicated TaggerDataset, uses DataLoader for efficiency,
    and saves new prompts to separate files.

    Args:
        dataset_root: Root directory of the image dataset.
        tagger_model_path: Path to the directory containing the tagger model.
        output_csv_path: Path to save the final CSV file.
        tagger_device: Device for the tagger model ('cuda' or 'cpu').
        tag_threshold: Confidence threshold for accepting tags.
        batch_size: Number of samples per batch for DataLoader.
        num_workers: Number of worker processes for DataLoader.
        output_suffix: Suffix for the new prompt file (before .txt).
        dataloader_prefetch_factor: prefetch_factor for DataLoader.
        dataloader_persistent_workers: persistent_workers for DataLoader.
        save_original_on_error: If True, save the original prompt to the
                                output file if tagging fails for an image.
        global_path: Optional path prefix for dataset images.
        label_ext: Extension for prompt files.
        skip_rating: Not include the rating prediction.
    """
    if Tagger is None:
        logger.error("Tagger class not imported. Cannot proceed.")
        return

    logger.info("Initializing Tagger...")
    try:
        # Initialize the Tagger class (loads model once)
        tagger = Tagger(
            model_dir=tagger_model_path, device=tagger_device, threshold=tag_threshold
        )
        # Get tagger's expected input details for the dataset
        tagger_input_size = tagger.input_size  # (H, W)
        # Assuming standard mean/std, adjust if tagger config specifies otherwise
        tagger_means = tagger.expected_processor_config["image_mean"]
        tagger_stds = tagger.expected_processor_config["image_std"]

    except Exception as e:
        logger.error(f"Failed to initialize Tagger: {e}", exc_info=True)
        return

    logger.info(f"Initializing TaggerDataset for root: {dataset_root}")
    try:
        # Create the dedicated dataset for the tagger
        dataset = TaggerDataset(
            root=dataset_root,
            tagger_input_size=tagger_input_size,
            tagger_means=tagger_means,
            tagger_stds=tagger_stds,
            label_ext=label_ext,
            global_path=global_path,
            booru_df_path=booru_df_path,
            # Add other params like resample_filter if needed
        )
    except Exception as e:
        logger.error(f"Failed to initialize TaggerDataset: {e}", exc_info=True)
        return

    if len(dataset) == 0:
        logger.warning("Dataset is empty. No prompts to upsample.")
        return

    logger.info(
        f"Setting up DataLoader with batch_size={batch_size}, num_workers={num_workers}"
    )
    pin_memory = (tagger_device == "cuda") and torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": custom_collate_with_paths,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    # Add prefetch_factor and persistent_workers if supported
    # Use getattr for safer check across PyTorch versions
    if getattr(DataLoader, "prefetch_factor", None) is not None:
        loader_kwargs["prefetch_factor"] = dataloader_prefetch_factor
    loader_kwargs["persistent_workers"] = (
        dataloader_persistent_workers if num_workers > 0 else False
    )

    data_loader = DataLoader(dataset, **loader_kwargs)

    logger.info(f"Starting prompt upsampling for {len(dataset)} images...")
    processed_count = 0
    error_count = 0
    skipped_batches = 0
    results = []

    for batch in tqdm(data_loader, desc="Upsampling Prompts", leave=False, ascii=True):
        if batch is None:
            logger.warning("Skipping an empty or invalid batch (collate error).")
            skipped_batches += 1
            error_count += batch_size  # Estimate errors for the skipped batch
            continue

        # Unpack batch - custom_collate_with_paths yields (tensor, [prompt], [path], [metadata])
        try:
            # The first element is the batch of preprocessed tensors
            # The second is a list of original prompts (strings)
            # The third is a list of Path objects
            image_tensors, prompts, img_paths, metadata_batch = batch
            # img_paths are already Path objects from the custom collate function
        except (ValueError, TypeError) as e:
            logger.error(f"Error unpacking batch: {e}. Skipping batch.")
            # Log details about the batch structure that caused the error
            if isinstance(batch, (list, tuple)) and len(batch) > 0:
                logger.error(f"Batch structure: {[type(item) for item in batch]}")
            else:
                logger.error(f"Batch type: {type(batch)}")
            # Estimate errors based on expected batch size
            error_count += batch_size
            continue

        # Get scores for the batch of tensors
        # tagger.tag_batch returns Optional[np.ndarray] of shape [B, num_tags]
        batch_scores_np = tagger.tag_batch(image_tensors)  # Pass tensors now

        # Check if the entire batch inference failed
        if batch_scores_np is None:
            logger.warning(f"Tagging failed for the entire batch. Check Tagger logs.")
            error_count += len(prompts)
            # save original prompts if flag is set
            if save_original_on_error:
                for i in range(len(prompts)):
                    img_path = img_paths[i]
                    output_filename = img_path.stem + output_suffix + ".txt"
                    output_path = img_path.parent / output_filename
                    try:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(output_path, "w", encoding="utf-8") as f:
                            f.write(prompts[i])  # Save original prompt
                    except Exception as e:
                        logger.error(
                            f"Error writing original prompt on batch fail"
                            f" for {output_path}: {e}"
                        )
            continue  # Move to the next batch

        # Process each item in the batch
        for i in range(len(prompts)):
            original_prompt = prompts[i]
            img_path = img_paths[i]
            row_metadata = metadata_batch[i]

            # Get scores for the current image from the batch results
            image_scores = batch_scores_np[i]

            # Check if 'original' is in the prompt to determine if we
            # should skip character tags.
            normalized_original_tags = {
                t.strip() for t in original_prompt.lower().split(",")
            }
            # character and copyrights should be omitted and trust danbooru
            # Force skip characters if only_general_tags is True
            if only_general_tags:
                skip_chars = True
            else:
                skip_chars = "original" in normalized_original_tags

            # Convert scores to tags based on threshold
            new_tags = tagger.get_tags_from_scores(
                image_scores, skip_rating=skip_rating, skip_character_tags=skip_chars
            )

            final_prompt, upsampled_tags = combine_prompts_intelligently(
                original_prompt, new_tags
            )

            # Validate and clean the upsampled tags using metadata and heuristics
            # This filters out hallucinations (e.g. extra hair colors, conflicting clothes)
            upsampled_tags = validate_upsampled_batch(upsampled_tags, row_metadata)

            results.append({"id": img_path.stem, "upsampled_tags": upsampled_tags})

            # Define the output path
            output_filename = img_path.stem + output_suffix + ".txt"
            output_path = img_path.parent / output_filename

            # Save the upsampled prompt (even if empty, might be intended)
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(upsampled_tags)
                processed_count += 1
            except IOError as e:
                logger.error(f"Error writing prompt file {output_path}: {e}")
                error_count += 1
            except Exception as e:
                logger.error(
                    f"Unexpected error saving {output_path}: {e}", exc_info=True
                )
                error_count += 1

        # Optional: Clear tensor variables to potentially free GPU memory sooner
        del image_tensors, batch_scores_np, batch
        if tagger_device == "cuda":
            torch.cuda.empty_cache()  # Use with caution, can slow things down

    logger.info(f"Prompt upsampling finished.")
    logger.info(f"Successfully processed and saved: {processed_count} prompts.")
    logger.info(f"Errors encountered (individual file/tagging errors): {error_count}")
    if skipped_batches > 0:
        logger.warning(f"Skipped {skipped_batches} batches due to collation errors.")

    # Create a pandas DataFrame and save all results to a single CSV file.
    if results:
        logger.info(f"Saving {len(results)} upsampled tags to {output_csv_path}...")
        try:
            df = pd.DataFrame(results)
            df.to_csv(output_csv_path, index=False)
            logger.info("Successfully saved the CSV file.")
        except Exception as e:
            logger.error(f"Failed to save CSV file: {e}", exc_info=True)
    else:
        logger.warning("No results were generated to save to CSV.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upsample dataset prompts using a tagger."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of the dataset images and prompts.",
    )
    parser.add_argument(
        "--output_csv_path",
        type=str,
        required=True,
        help="Path to save the output CSV file.",
    )
    parser.add_argument(
        "--tagger_model_dir",
        type=str,
        default="model/wd_swinv2",
        help="Directory containing the tagger model files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for tagger inference ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for processing."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of DataLoader workers."
    )
    parser.add_argument(
        "--tag_threshold",
        type=float,
        default=0.35,
        help="Confidence threshold for tags.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_upsampled",
        help="Suffix for output prompt filenames.",
    )
    parser.add_argument(
        "--save_original_on_error",
        action="store_true",
        help="Save original prompt if tagging fails for an image.",
    )
    parser.add_argument(
        "--global_path",
        type=str,
        default=None,
        help="Optional global path prefix for dataset images.",
    )
    parser.add_argument(
        "--label_ext", type=str, default=".txt", help="Extension for prompt files."
    )
    parser.add_argument(
        "--prefetch_factor", type=int, default=2, help="DataLoader prefetch factor."
    )
    parser.add_argument(
        "--persistent_workers",
        action=argparse.BooleanOptionalAction,  # Use new action for bool flags
        default=True,
        help="Use persistent DataLoader workers.",
    )
    parser.add_argument(
        "--skip_rating",
        action=argparse.BooleanOptionalAction,  # Use new action for bool flags
        default=True,
        help="Skip the rating prediction",
    )
    parser.add_argument("--booru_df_path", type=str, help="Path to the booru parquets")

    args = parser.parse_args()

    # Ensure CUDA is available if selected
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA selected but not available. Switching to CPU.")
        args.device = "cpu"

    # Run the upsampling process
    upsample_prompts_batch(
        dataset_root=args.data_root,
        tagger_model_path=args.tagger_model_dir,
        output_csv_path=args.output_csv_path,
        tagger_device=args.device,
        tag_threshold=args.tag_threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_suffix=args.output_suffix,
        dataloader_prefetch_factor=args.prefetch_factor,
        dataloader_persistent_workers=args.persistent_workers,
        save_original_on_error=args.save_original_on_error,
        global_path=args.global_path,
        label_ext=args.label_ext,
        skip_rating=args.skip_rating,
        booru_df_path=args.booru_df_path,
    )

    logger.info("Script finished.")
