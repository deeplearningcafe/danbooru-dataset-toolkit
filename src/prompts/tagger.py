import os
import json
import argparse
import logging
from typing import Tuple, Dict, Optional, List, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import timm
from collections import defaultdict

from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
)
from transformers.image_transforms import (
    rescale,
    to_channel_dimension_format,
    normalize,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_IMG_SIZE = (448, 448)
DEFAULT_MODEL_DIR = 'model/swinv2'

def format_danbooru_tag(tag: str) -> str:
    """
    Format a Danbooru-style tag into more readable text.
    
    Transformations:
    1. Replace underscores with spaces
    2. Escape parentheses with backslashes
    
    Args:
        tag (str): The original Danbooru tag
        
    Returns:
        str: The formatted tag
    
    Examples:
        >>> format_danbooru_tag("looking_at_viewer")
        'looking at viewer'
        >>> format_danbooru_tag("star_(symbol)")
        'star \(symbol\)'
    """
    # Replace underscores with spaces
    formatted_tag = tag.replace('_', ' ')
    
    # Escape parentheses with backslashes
    formatted_tag = formatted_tag.replace('(', r'\(').replace(')', r'\)')
    
    return formatted_tag

def load_feature_extractor(
    model_dir: str
) -> Tuple[Optional[nn.Module], Optional[Dict], Optional[Tuple[int, int]]]:
    """
    Loads the Swin V2 feature extractor model and its config.

    Args:
        model_dir: Directory containing model.safetensors and config.json.

    Returns:
        Tuple of (model, config, input_size_hw) or (None, None, None).
    """
    checkpoint_path = os.path.join(model_dir, 'model.safetensors')
    config_path = os.path.join(model_dir, 'config.json')

    if not all(os.path.exists(p) for p in [model_dir, checkpoint_path, config_path]):
        logger.error(f"Model directory, checkpoint, or config not found in {model_dir}")
        return None, None, None

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded config from {config_path}")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {config_path}")
        return None, None, None
    except IOError as e:
        logger.error(f"Error reading config file {config_path}: {e}")
        return None, None, None

    model_name = config.get('architecture', 'swinv2_base_window8_256')
    num_classes = config.get('num_classes', 0) # Original classes, not ours
    model_kwargs = config.get('model_args', {})
    input_size_cfg = config.get('pretrained_cfg', {}).get('input_size')

    if not isinstance(input_size_cfg, (list, tuple)) or len(input_size_cfg) != 3:
        logger.warning(f"Invalid input_size in config: {input_size_cfg}. Using default.")
        input_size_hw = DEFAULT_IMG_SIZE
    else:
        # Assuming format is (C, H, W)
        input_size_hw = (input_size_cfg[1], input_size_cfg[2])

    logger.info(f"Attempting to load model: {model_name}")
    try:
        model = timm.create_model(
            model_name,
            checkpoint_path=checkpoint_path,
            num_classes=num_classes, # Load with original head, we only use features
            **model_kwargs
        )
        model.eval() # Set to evaluation mode
        model.requires_grad_(False)
        logger.info("Feature extractor model loaded successfully.")
        return model, config, input_size_hw
    except Exception as e:
        logger.error(f"Error creating model {model_name}: {e}", exc_info=True)
        return None, None, None

def resize_with_padding(
    image: np.ndarray,
    size: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    resample = PILImageResampling.BILINEAR,
    data_format: Optional[ChannelDimension] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
):
    """
    Resizes image while maintaining aspect ratio and padding to fill target size.
    Correctly handles BGR conversion expected by the model.
    
    Args:
        image: Input image as numpy array (assumed RGB initially if from PIL)
        size: Target size as (height, width)
        color: Background color for padding as RGB tuple
        resample: PIL resampling filter
        data_format: Output channel dimension format
        input_data_format: Input channel dimension format
        
    Returns:
        Resized, padded, and BGR-converted image as numpy array
    """
    # For transformations, keep same data format as input unless specified
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    data_format = input_data_format if data_format is None else data_format

    # Convert to PIL for resizing if it's a numpy array
    # Keep track if we need to rescale back later
    do_rescale_back = False
    if not isinstance(image, Image.Image):
        # Check if image needs rescaling for PIL conversion (0-1 range)
        if is_scaled_image(image):
            do_rescale_back = True
            image = image * 255
        # Ensure it's uint8 for PIL
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        # Create PIL image, inferring mode if necessary
        image = Image.fromarray(image)

    # Get original dimensions
    original_width, original_height = image.size
    height, width = size

    # Calculate aspect ratio for resize
    ratio = min(width / original_width, height / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    # Resize maintaining aspect ratio
    resized_image = image.resize(
        (new_width, new_height), resample=resample
    )

    # Create new image with solid background (using RGBA for paste)
    new_image_rgba = Image.new("RGBA", (width, height), color + (255,))

    # Paste resized image at the center
    offset = ((width - new_width) // 2, (height - new_height) // 2)
    # Ensure resized image is RGBA for pasting with alpha
    resized_image_rgba = resized_image.convert("RGBA")
    new_image_rgba.paste(resized_image_rgba, offset, resized_image_rgba)

    # Convert final image back to RGB (discard alpha)
    new_image_rgb = new_image_rgba.convert("RGB")

    # Convert PIL image (RGB) to numpy array (float32)
    image_array = np.asarray(new_image_rgb, dtype=np.float32)
    
    # *** CHANGE START: Added BGR conversion ***
    # Convert RGB to BGR as expected by the model's original preprocessing
    image_array = image_array[:, :, ::-1]
    # *** CHANGE END ***

    # Add channel dimension if needed (e.g., for grayscale - unlikely here)
    if image_array.ndim == 2:
        image_array = np.expand_dims(image_array, axis=-1)
    
    # Convert to desired channel format (e.g., channels_first for PyTorch)
    # Input is now channels_last after numpy conversion and BGR swap
    image_array = to_channel_dimension_format(
        image_array, data_format, input_channel_dim=ChannelDimension.LAST
    )
    
    # Restore original scale (0-1) if needed
    if do_rescale_back:
        image_array = image_array / 255.0

    return image_array

class Tagger:
    """
    Handles loading the tagging model and processing batches of tensors.
    Preprocessing is expected to be done externally (e.g., by DataLoader).
    """
    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL_DIR,
        device: str = 'cuda',
        threshold: float = 0.35,
        # Removed unused batch_size parameter
    ):
        """
        Initializes the Tagger by loading the model, config, and labels.

        Args:
            model_dir: Directory containing model files.
            device: Device to run the model on ('cuda' or 'cpu').
            threshold: Confidence threshold for filtering tags.
        """
        self.model_dir = model_dir
        self.device = torch.device(device)
        self.threshold = threshold

        # Load model and config
        self.model, self.config, input_size_hw = self._load_model()
        if self.model is None:
            raise RuntimeError("Failed to load the tagging model.")
        self.model.to(self.device)

        # Determine input size (use default if not found)
        self.input_size = input_size_hw or DEFAULT_IMG_SIZE # (H, W)

        # Load labels (tag names)
        self.tag_names = self._load_labels()
        if not self.tag_names:
            raise RuntimeError("Failed to load tag labels.")

        # Store expected processor config details for reference if needed
        # These are the parameters the input tensors SHOULD have been processed with
        self.expected_processor_config = {
            "size": {"height": self.input_size[0], "width": self.input_size[1]},
            "image_mean": [0.5, 0.5, 0.5], # Example, adjust if needed
            "image_std": [0.5, 0.5, 0.5],  # Example, adjust if needed
            # Other params like color, rescale_factor are part of preprocessing
        }
        logger.info(f"Tagger initialized for device {self.device}. Expects "
                    f"input tensors preprocessed for size {self.input_size}")

    def _load_model(
        self
    ) -> Tuple[Optional[nn.Module], Optional[Dict], Optional[Tuple[int, int]]]:
        """Loads the Swin V2 model and config (internal helper)."""
        # --- Logic remains the same as previous version ---
        # Omitted for brevity
        # --- Placeholder for the actual loading logic ---
        checkpoint_path = os.path.join(self.model_dir, 'model.safetensors')
        config_path = os.path.join(self.model_dir, 'config.json')
        if not all(os.path.exists(p) for p in [self.model_dir, checkpoint_path, config_path]):
            logger.error(f"Model dir, checkpoint, or config missing in {self.model_dir}")
            return None, None, None
        try:
            with open(config_path, 'r') as f: config = json.load(f)
        except Exception as e:
            logger.error(f"Error reading/parsing config {config_path}: {e}")
            return None, None, None
        model_name = config.get('architecture', 'swinv2_base_window8_256')
        num_classes = config.get('num_classes', 0)
        model_kwargs = config.get('model_args', {})
        input_size_cfg = config.get('pretrained_cfg', {}).get('input_size')
        input_size_hw = None
        if isinstance(input_size_cfg, (list, tuple)) and len(input_size_cfg) == 3:
            input_size_hw = (input_size_cfg[1], input_size_cfg[2]) # (H, W)
        else:
             logger.warning(f"Using default input size {DEFAULT_IMG_SIZE}")
             input_size_hw = DEFAULT_IMG_SIZE
        try:
            model = timm.create_model(model_name, checkpoint_path=checkpoint_path, num_classes=num_classes, **model_kwargs)
            model.eval(); model.requires_grad_(False)
            logger.info(f"Loaded model {model_name} successfully.")
            return model, config, input_size_hw
        except Exception as e:
            logger.error(f"Error creating model {model_name}: {e}", exc_info=True)
            return None, None, None

    def _load_labels(self) -> Optional[List[str]]:
        """Loads tag names from selected_tags.csv (internal helper)."""
        labels_path = os.path.join(self.model_dir, 'selected_tags.csv')
        if not os.path.exists(labels_path):
            logger.error(f"Labels file not found at {labels_path}")
            return None
        try:
            tags_df = pd.read_csv(labels_path)
            if 'name' not in tags_df.columns:
                 logger.error(f"'name' column not found in {labels_path}")
                 return None
            # Use zip to efficiently create a list of (name, category) tuples
            # This is more memory-efficient and faster than iterating row-by-row.
            tag_data = list(zip(
                tags_df["name"],
                tags_df["category"]
            ))
            logger.info(f"Loaded {len(tag_data)} tag names.")
            return tag_data
        except Exception as e:
            logger.error(f"Error loading tags from {labels_path}: {e}")
            return None

    @torch.no_grad() # Ensure no gradients are computed
    def tag_batch(
        self,
        batch_tensor: torch.Tensor
    ) -> Optional[np.ndarray]:
        """
        Tags a batch of preprocessed image tensors.

        Args:
            batch_tensor: A batch of image tensors, preprocessed and ready
                          for the model (e.g., shape [B, C, H, W]).

        Returns:
            A NumPy array of shape [B, num_tags] containing sigmoid scores,
            or None if inference fails.
        """
        if batch_tensor is None or batch_tensor.shape[0] == 0:
            logger.warning("Received empty or None tensor batch.")
            return None

        # 1. Move tensor to device
        try:
            batch_tensor = batch_tensor.to(self.device)
        except Exception as e:
             logger.error(f"Error moving batch tensor to device {self.device}: {e}")
             return None # Cannot proceed

        # 2. Perform model inference
        try:
            predictions = torch.sigmoid(self.model(batch_tensor))
            # Move predictions to CPU *before* returning for further processing
            predictions_np = predictions.cpu().numpy()
            return predictions_np
        except Exception as e:
            logger.error(f"Error during model inference: {e}", exc_info=True)
            # Return None if inference fails for the batch
            return None

    def get_tags_from_scores(
        self,
        scores: np.ndarray, # Shape (num_tags,)
        skip_rating: bool = False,
        skip_character_tags: bool = False, 
    ) -> List[str]:
         """
         Converts an array of scores for a single image into a list of tags
         above the threshold.
         """
         filtered_tags = []
         for tag_idx, score in enumerate(scores):
             if score > self.threshold:
                 if tag_idx < len(self.tag_names):
                     tag_name, category = self.tag_names[tag_idx]
                     if skip_rating and category == 9:
                         continue
                     if skip_character_tags and category == 4:
                        continue

                     filtered_tags.append(format_danbooru_tag(tag_name))
                 else:
                     # This should ideally not happen if model output matches labels
                     logger.warning(f"Prediction index {tag_idx} out of bounds"
                                    f" for tag names (len {len(self.tag_names)})")
         return filtered_tags


def tag_image(image_path: str, model_path: str, device: str = 'cpu',) -> dict:
    """
    Tags an image using the swinv2_base_window8_256 model.
    
    Args:
        image_path: Path to the image file
        model_path: Path to the model directory containing model.safetensors and config.json
        
    Returns:
        Dictionary with tags and their confidence scores
    """
    # Load model
    model, config, input_size = load_feature_extractor(model_path)
    if model is None:
        logger.error("Failed to load model")
        return {}
    model.to(device)
    
    # Load labels
    labels_path = os.path.join(model_path, 'selected_tags.csv')
    if not os.path.exists(labels_path):
        logger.error(f"Labels file not found at {labels_path}")
        return {}
    
    try:
        tags_df = pd.read_csv(labels_path)
        tag_names = tags_df["name"].tolist()
    except Exception as e:
        logger.error(f"Error loading tags: {e}")
        return {}
    
    processor_config = {
        "size": {"height": 448, "width": 448},
        "color": [255, 255, 255], # Padding color (RGB)
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "rescale_factor": 1 / 255.0,
        "resample": PILImageResampling.BILINEAR, # Use constant
    }

    size_dict = processor_config["size"]
    color_tuple = tuple(processor_config["color"])
    image_mean = processor_config["image_mean"]
    image_std = processor_config["image_std"]
    rescale_factor = processor_config["rescale_factor"]
    resample_filter = processor_config["resample"]

    # Load and preprocess image
    try:
        # Load image as PIL RGB
        image = Image.open(image_path).convert('RGB')
        # Convert to numpy array (still RGB, range 0-255)
        image_array = np.array(image)
    except Exception as e:
        logger.error(f"Error opening or converting image {image_path}: {e}")
        return {}
    
    # --- Preprocessing Pipeline ---
    # 1. Resize with padding and convert to BGR numpy array
    #    Output format will be channels_first for PyTorch
    processed_image = resize_with_padding(
        image=image_array,
        size=(size_dict["height"], size_dict["width"]),
        color=color_tuple,
        resample=resample_filter,
        data_format=ChannelDimension.FIRST # PyTorch expects channels first
    )
    
    # 2. Rescale values to [0,1] if they aren't already
    #    resize_with_padding handles rescaling back if input was 0-1,
    #    so here we ensure it's rescaled *from* 0-255 to 0-1 if needed.
    if not is_scaled_image(processed_image):
         processed_image = rescale(
            processed_image,
            scale=rescale_factor,
            data_format=ChannelDimension.FIRST
        )
    
    # 3. Normalize with mean and std
    processed_image = normalize(
        processed_image, 
        mean=image_mean,
        std=image_std,
        data_format=ChannelDimension.FIRST
    )
    
    # 4. Convert final processed numpy array to PyTorch tensor
    #    Add batch dimension and move to target device
    img_tensor = torch.tensor(processed_image).float().unsqueeze(0).to(device)

        
    # Get predictions
    with torch.no_grad():
        # Model expects BGR input, which `processed_image` now is
        predictions = torch.sigmoid(model(img_tensor)).squeeze(0).cpu().numpy()
    
    logger.info(
    f"Max CUDA memory allocated: "
    f"{torch.cuda.max_memory_allocated() / (1024**3):.2f} GB"
    )

    # Map predictions to tags
    results = {}
    for tag, score in zip(tag_names, predictions):
        results[tag] = float(score)
    
    # Sort results by confidence
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_results

def main():
    parser = argparse.ArgumentParser(description='Tag images using SwinV2 model')
    parser.add_argument('--image_path', type=str, required=True, 
                        help='Path to the image file')
    parser.add_argument(
        '--model_path', type=str, default=DEFAULT_MODEL_DIR,
        help="Directory containing SwinV2 model files (config.json, model.safetensors)."
    )
    parser.add_argument('--threshold', type=float, default=0.35,
                        help='Confidence threshold for tags (default: 0.35)')
    
    parser.add_argument(
        '--device', type=str, default='cuda', choices=['cuda', 'cpu'],
        help="Device to use ('cuda' or 'cpu')."
    )

    args = parser.parse_args()
    
    # Tag the image
    tags_with_scores = tag_image(args.image_path, args.model_path, args.device)
    
    if not tags_with_scores:
        print("Failed to tag the image")
        return
    
    # Print results
    print(f"Tags for {args.image_path}:")
    print("-" * 40)
    
    # Filter by threshold
    filtered_tags = {tag: score for tag, score in tags_with_scores.items() 
                    if score > args.threshold}
    
    # Group by categories if available
    try:
        tags_df = pd.read_csv(os.path.join(args.model_path, 'selected_tags.csv'))
        categories = defaultdict(list)
        
        for tag, score in filtered_tags.items():
            category_id = tags_df.loc[tags_df['name'] == tag, 'category'].values
            if len(category_id) > 0:
                cat_id = int(category_id[0])
                categories[cat_id].append((tag, score))
            else:
                categories[-1].append((tag, score))  # Unknown category
                
        for cat_id, tags in categories.items():
            category_name = f"Category {cat_id}"
            print(f"\n{category_name}:")
            for tag, score in sorted(tags, key=lambda x: x[1], reverse=True):
                print(f"  {tag}: {score:.4f}")
    except Exception as e:
        # If categorization fails, just show all tags
        print("All tags above threshold:")
        for tag, score in filtered_tags.items():
            print(f"  {tag}: {score:.4f}")

if __name__ == "__main__":
    main()
    """
    python tagger.py --image_path data/data-0000-cleaned/data-0000/6013718.jpg --model_path model/wd_swinv2 --device cpu
    python tagger.py --image_path data/data-0000-cleaned/data-0000/6013779.jpg --model_path model/wd_swinv2 --device cpu
    """
