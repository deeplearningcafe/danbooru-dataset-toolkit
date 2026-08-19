from pathlib import Path
from typing import Callable, Generator, Optional, List, Tuple, Set, Dict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PIL import Image
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset
import torch
from transformers import CLIPTokenizer
import h5py
import json
from collections import defaultdict
import math
import gc  # For garbage collection if needed
import pandas as pd
from ..utils.loader import parallel_scan_images, resolve_image_path, AESTHETIC_LABEL

import sys

sys.path.append("..")

import os

# Get the absolute path of the current file (module_B.py)
current_file_path = os.path.abspath(__file__)

# Get the directory of the current file (folder_B)
current_dir = os.path.dirname(current_file_path)

# Get the parent directory (common_parent)
common_parent_dir = os.path.dirname(os.path.dirname(current_dir))

# Construct the path to folder_A
folder_A_path = os.path.join(common_parent_dir, "models")
print(folder_A_path)
# Add folder_A's path to sys.path
sys.path.insert(0, folder_A_path)
sys.path.insert(0, common_parent_dir)
from clip import Clip, ClipConfig
from vae import Vae, VaeConfig

# Define type hints for clarity
Resolution = Tuple[int, int]


def process_image(
    path: Path, label_ext: str = ".txt"
) -> Tuple[Path, Tuple[int, int], Optional[Exception]]:
    """
    Try to open an image, check for a non-empty prompt file (.txt or .json),
    and return its dimensions or an appropriate error.
    """
    actual_path = resolve_image_path(path)
    prompt_path = path.with_suffix(label_ext)
    prompt_exists = False
    if prompt_path.exists() and prompt_path.stat().st_size > 0:
        prompt_exists = True

    if not prompt_exists:
        return (
            path,
            None,
            FileNotFoundError(f"Prompt not found or is empty for {path.name}"),
        )

    try:
        with Image.open(actual_path) as img:
            size = img.size
        return path, (size[0], size[1]), None  # (width, height) format
    except Exception as e:
        return path, None, e


def encode_text_batch(
    prompts: List[str], text_encoder, tokenizer, max_length: int, device: str = "cpu"
) -> torch.Tensor:
    """
    Encodes a batch of text prompts using tokenizer and text encoder.
    Handles long prompts by chunking.

    Args:
        prompts: List of text prompts to encode.
        text_encoder: CLIP text encoder model.
        tokenizer: Tokenizer for processing text.
        max_length: Maximum token length for text.
        device: Device for computation ('cuda' or 'cpu').

    Returns:
        Tensor of text embeddings for the batch [B, seq_len, embed_dim].
    """
    input_ids = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids

    tokenizer_max_length = tokenizer.model_max_length

    # Handle long prompts by chunking if necessary
    # Note: This assumes all prompts in the batch might need chunking if one does.
    # A more complex implementation could handle mixed lengths.
    if input_ids.shape[-1] > tokenizer_max_length:
        processed_ids = []
        # Process each prompt's IDs individually for chunking logic
        for iids_single in input_ids:  # Iterate over batch dimension
            z = []
            # Chunk the single prompt's IDs
            for i in range(
                1, max_length - tokenizer_max_length + 2, tokenizer_max_length - 2
            ):
                ids_chunk = (
                    iids_single[0].unsqueeze(0),  # BOS
                    iids_single[i : i + tokenizer_max_length - 2],
                    iids_single[-1].unsqueeze(0),  # PAD or EOS
                )
                ids_chunk = torch.cat(ids_chunk)

                # Fix special tokens at chunk boundaries
                if (
                    ids_chunk[-2] != tokenizer.eos_token_id
                    and ids_chunk[-2] != tokenizer.pad_token_id
                ):
                    ids_chunk[-1] = tokenizer.eos_token_id
                if ids_chunk[1] == tokenizer.pad_token_id:
                    ids_chunk[1] = tokenizer.eos_token_id

                z.append(ids_chunk)
            # Stack chunks for this single sample [num_chunks, 77]
            processed_ids.append(torch.stack(z))
        # Stack all processed samples [B, num_chunks, 77]
        processed_ids = torch.stack(processed_ids)

        batch_size = processed_ids.size(0)
        # Reshape for encoder [B*num_chunks, 77]
        input_ids_for_encoder = processed_ids.reshape((-1, tokenizer_max_length))
    else:
        # No chunking needed, use original input_ids
        input_ids_for_encoder = input_ids
        batch_size = input_ids.size(0)  # Get batch size

    # Encode the batch
    with torch.no_grad():
        # Ensure input is on the correct device
        encoder_output = text_encoder(input_ids_for_encoder.to(device))
        # Typically use the penultimate layer output
        text_embeddings = encoder_output[-1][-2]  # Check model output structure

    # Reshape and concatenate if chunking was done
    if input_ids.shape[-1] > tokenizer_max_length:
        # Reshape back: [B, num_chunks * 77, embed_dim]
        text_embeddings = text_embeddings.reshape(
            (batch_size, -1, text_embeddings.shape[-1])
        )

        # Remove redundant special tokens after concatenation
        states_list = [text_embeddings[:, 0].unsqueeze(1)]  # Keep <BOS>
        for i in range(1, max_length, tokenizer_max_length):
            # Add content tokens from each chunk
            states_list.append(text_embeddings[:, i : i + tokenizer_max_length - 2])
        # Keep final <EOS> (or whatever is at the end)
        states_list.append(text_embeddings[:, -1].unsqueeze(1))
        text_embeddings = torch.cat(states_list, dim=1)
        # Ensure final shape is [B, max_length, embed_dim]
        # This might truncate if max_length isn't perfectly divisible
        text_embeddings = text_embeddings[:, :max_length, :]

    # Else (no chunking), embeddings are already [B, 77, embed_dim]
    # or [B, max_length, embed_dim] if max_length <= 77

    return text_embeddings  # Shape: [B, seq_len, embed_dim]


def tokenize_text_batch(prompts, tokenizer, max_length=77):
    """
    Tokenize a batch of text prompts without encoding.

    Args:
        prompts: List of text strings
        tokenizer: The tokenizer
        max_length: Maximum sequence length

    Returns:
        numpy.ndarray: Tokenized prompts [batch_size, max_length]
    """
    tokens = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="np",
    )

    return tokens.input_ids  # Shape: [batch_size, max_length]


class LatentEncodingDataset(Dataset):
    """
    Dataset class for preparing image data for latent encoding.
    Handles loading, bucketing assignment, and transformation.
    Does NOT perform the actual VAE/CLIP encoding itself.
    """

    def __init__(
        self,
        root: str | Path,
        dtype=torch.float32,
        max_log_ar_diff: float = 0.3,
        label_ext: str = ".txt",
        transform=None,
        df_tokens_path: str = None,
        already_tokenized: bool = False,
        max_res_area: Tuple[int, int] = (768, 512),
        max_dim_limit: int = 1024,
        base_res: Tuple[int, int] = (512, 512),
    ):
        self.root = Path(root)
        self.dtype = dtype
        self.max_log_ar_diff = max_log_ar_diff
        self.label_ext = label_ext

        if transform is None:
            # Define a default transform if none provided
            from torchvision.transforms import v2

            self.transform = v2.Compose(
                [
                    v2.PILToTensor(),
                    v2.ToDtype(dtype, scale=True),
                    v2.Normalize([0.5], [0.5]),
                ]
            )
            print("Using default transform.")
        else:
            self.transform = transform

        self.df_tokens = None
        if df_tokens_path:
            print(f"Loading token delimiter data from {df_tokens_path}...")
            try:
                # CHANGED: Force the 'id' column to be read as a string.
                # This ensures the index type matches the string-based
                # 'img_id' used for lookups, preventing KeyErrors.
                df = pd.read_csv(df_tokens_path, dtype={"id": str})
                self.df_tokens = df.set_index("id")
                print(
                    f"Successfully loaded {len(self.df_tokens)} "
                    "token delimiter entries."
                )
            except FileNotFoundError:
                print(
                    f"Warning: Token delimiter file not found at "
                    f"{df_tokens_path}. Proceeding without it."
                )
            except Exception as e:
                print(
                    f"Warning: Failed to load or process token delimiter "
                    f"file: {e}. Proceeding without it."
                )
        self.already_tokenized = already_tokenized

        (self.paths, _raw_res, self.booru_id_to_idx) = self._filter_valid_images(
            Path(root), num_workers=4
        )

        self.raw_res = np.array(_raw_res, dtype=np.int32)
        print(
            f"Loaded {len(self.paths)} valid images and created mapping for "
            f"{len(self.booru_id_to_idx)} booru IDs."
        )

        min_size = 256
        divisible = 64
        vae_factor = 8

        _bucket_res, self.log_aspect_ratios = self.generate_buckets(
            max_size=max_res_area,
            min_size=min_size,
            divisible=divisible,
            base_res=base_res,
            vae_factor=vae_factor,
            dim_limit=max_dim_limit,
        )
        # Store resolutions as int32 numpy array
        self.bucket_resolutions = np.array(_bucket_res, dtype=np.int32)
        # print(self.bucket_resolutions)
        # print("*"*20)
        # print(self.log_aspect_ratios)
        # so the log_aspect_ratios is a list and is always the same order so no need for dict,
        # the same for the images, just store the idx of the path list and thats enough

        self.buckets = self.assign_buckets(
            self.paths, self.raw_res, self.log_aspect_ratios, self.max_log_ar_diff
        )

        # Print samples per bucket
        for i in range(len(self.buckets)):
            print(
                f"Bucket with res {_bucket_res[i]} : {self.buckets[i].shape} elements"
            )

        # Create flat index mapping for efficient access ONLY to assigned images
        self.index_mapping = np.array(self._create_index_mapping(), dtype=np.int32)

        print(f"Initialized dataset with {len(self.index_mapping)} assignable images.")

    def load_entry(self, p: Path, label_ext: str = ".txt"):
        """
        Loads an image file and its corresponding prompt from a text file.

        Args:
            p: Path to the image file
            label_ext: File extension for the prompt file (default: ".txt")

        Returns:
            Tuple of (PIL Image object, prompt string)
        """
        actual_path = resolve_image_path(p)
        _img = Image.open(actual_path)
        with p.with_suffix(label_ext).open("r") as f:
            if label_ext == ".txt":
                prompt = f.read()
            elif label_ext == ".json":
                # prompt becomes a dict with the tokens
                prompt = json.load(f)

        # Handle different image modes
        if _img.mode == "RGB":
            img = _img
        elif _img.mode == "RGBA":
            # Handle transparent images
            baimg = Image.new("RGB", _img.size, (255, 255, 255))
            baimg.paste(_img, (0, 0), _img)
            img = baimg
        else:
            img = _img.convert("RGB")

        return img, prompt

    # Helper method to load only the prompt efficiently
    def _load_prompt(self, p: Path, label_ext: str = ".txt") -> str:
        """Loads only the prompt string from the corresponding text file."""
        try:
            with p.with_suffix(label_ext).open("r") as f:
                # If the label file is a .json, parse it into a dictionary.
                # Otherwise, read it as plain text.
                if label_ext == ".json":
                    prompt = json.load(f)
                else:
                    prompt = f.read()
            return prompt
        except FileNotFoundError:
            print(f"Warning: Label file not found for {p}, using empty prompt.")
            return ""
        except Exception as e:
            print(
                f"Warning: Error reading label file for {p}: {e}, using empty prompt."
            )
            return ""

    def _filter_valid_images(
        self, data_dir, num_workers: int = os.cpu_count() // 2 or 4
    ) -> Tuple[List[Path], List[Tuple[int, int]]]:
        """
        Filter valid images from a directory and return paths, dimensions,
        and a mapping from image ID to index.
        An image is valid if it can be opened and has a corresponding
        non-empty prompt file.

        Args:
            data_dir: Directory containing image files.
            num_workers: Number of threads for parallel processing.

        Returns:
            A tuple of (valid_paths, dimensions, booru_id_to_idx_map).
        """
        print("Starting parallel scan to find all potential image files...")
        img_paths = parallel_scan_images(data_dir, num_workers=num_workers)
        print(f"Found {len(img_paths)} potential image files.")

        valid_paths = []
        dimensions = []
        booru_id_to_idx = {}

        # Process images in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(process_image, p, self.label_ext) for p in img_paths
            ]

            for future in tqdm(
                futures,
                desc="Processing images",
                total=len(img_paths),
                leave=False,
                ascii=True,
            ):
                path, size, error = future.result()
                if error is None:
                    booru_id = path.stem
                    current_idx = len(valid_paths)
                    booru_id_to_idx[booru_id] = current_idx

                    valid_paths.append(path)
                    dimensions.append(size)
                else:
                    if isinstance(error, FileNotFoundError):
                        # This handles the custom error from process_image for
                        # missing or empty prompt files.
                        print(f"\033[33mSkipped: {error}\033[0m")
                    else:
                        # This handles standard image processing errors.
                        print(
                            f"\033[33mSkipped: Error processing image "
                            f"{path.name}: {error}\033[0m"
                        )
        print(f"Found {len(valid_paths)} images with valid prompts.")
        return valid_paths, dimensions, booru_id_to_idx

    def generate_buckets(
        self,
        max_size: Resolution,
        min_size: int,
        divisible: int,
        base_res: Resolution,
        vae_factor: int = 8,
        dim_limit: int = 1024,  # Max dimension size from NovelAI readme
    ) -> Tuple[List[Resolution], List[float]]:
        """
        Generates bucket resolutions based on maximum latent area and constraints,
        inspired by NovelAI's approach.

        Args:
            max_size: Maximum resolution (width, height) defining max latent area.
            min_size: Minimum width or height allowed.
            divisible: Dimensions must be divisible by this value.
            base_res: Base resolution (width, height) tuple to include.
            vae_factor: VAE downsampling factor (usually 8).
            dim_limit: Absolute maximum dimension size allowed for any side.

        Returns:
            Tuple of (sorted resolutions list, log_aspect_ratios list)
        """
        # Calculate max latent tokens based on max_size allowed
        max_tokens = (max_size[0] / vae_factor) * (max_size[1] / vae_factor)

        possible_resolutions: Set[Resolution] = set()

        # --- Generate Landscape-dominant buckets ---
        w = min_size
        while (w / vae_factor) * (
            min_size / vae_factor
        ) <= max_tokens and w <= dim_limit:
            h = min_size
            # Find max height for this width within token limit and dim limit
            while (w / vae_factor) * ((h + divisible) / vae_factor) <= max_tokens and (
                h + divisible
            ) <= dim_limit:
                h += divisible
            # Add this resolution if valid
            if h >= min_size:
                possible_resolutions.add((w, h))
            w += divisible

        # --- Generate Portrait-dominant buckets ---
        h = min_size
        while (min_size / vae_factor) * (
            h / vae_factor
        ) <= max_tokens and h <= dim_limit:
            w = min_size
            # Find max width for this height within token limit and dim limit
            while ((w + divisible) / vae_factor) * (h / vae_factor) <= max_tokens and (
                w + divisible
            ) <= dim_limit:
                w += divisible
            # Add this resolution if valid
            if w >= min_size:
                possible_resolutions.add((w, h))
            h += divisible

        # Ensure base resolution is included if it meets constraints
        # (Check token count and dimension limits)
        if (
            (base_res[0] / vae_factor) * (base_res[1] / vae_factor) <= max_tokens
            and base_res[0] >= min_size
            and base_res[1] >= min_size
            and base_res[0] <= dim_limit
            and base_res[1] <= dim_limit
            and base_res[0] % divisible == 0
            and base_res[1] % divisible == 0
        ):
            possible_resolutions.add(base_res)
        else:
            print(
                f"Warning: Base resolution {base_res} is invalid or exceeds "
                f"max_tokens/dim_limit derived from max_size {max_size}. "
                f"It might not be added unless generated by the loops."
            )

        if not possible_resolutions:
            raise ValueError(
                "No valid bucket resolutions generated! Check "
                "constraints (min_size, max_size, dim_limit)."
            )

        # Sort for consistency - using NovelAI's sort key for closer matching
        # Sorts primarily by width, then inversely by height for tie-breaking
        sorted_res = sorted(
            list(possible_resolutions), key=lambda r: (r[0] * dim_limit - r[1])
        )

        # Pre-calculate log aspect ratios for efficient comparison
        log_aspect_ratios = [math.log(w / h) if h > 0 else 0 for w, h in sorted_res]
        # print(f"Generated {len(sorted_res)} unique aspect ratio buckets ")

        return sorted_res, log_aspect_ratios

    def assign_buckets(
        self, paths, resolutions, log_aspect_ratios, max_log_ar_diff=0.3
    ):
        """Assigns each image ID to its closest bucket by log aspect ratio."""
        # if not resolutions:
        if resolutions.size == 0:
            raise RuntimeError("Buckets must be generated before assignment.")

        initial_image_count = len(paths)
        assigned_count = 0
        pruned_image_ids = []
        # each bucket is a list of idx with the idx being the idx of the path list
        buckets = [[] for _ in range(len(log_aspect_ratios))]
        ar_array = np.array(log_aspect_ratios)

        for idx, (img_path, res) in enumerate(zip(paths, resolutions)):
            width, height = res
            if width <= 0 or height <= 0:
                pruned_image_ids.append(idx)
                continue

            # Use log aspect ratio for comparison (more robust than raw ratio)
            log_img_ar = math.log(width / height)

            min_diff = float("inf")

            diffs = np.abs(ar_array - log_img_ar)
            best_idx = int(diffs.argmin())
            min_diff = diffs[best_idx]

            # Assign if the difference is within the threshold
            # if best_bucket_index != -1 and min_diff <= max_log_ar_diff:
            if min_diff <= max_log_ar_diff:
                buckets[best_idx].append(idx)
                assigned_count += 1
            else:
                pruned_image_ids.append(idx)

        pruned_count = initial_image_count - assigned_count
        print(f"Assigned {assigned_count}/{initial_image_count} images.")
        if pruned_count > 0:
            print(
                f"Pruned {pruned_count} images due to extreme aspect "
                f"ratio (min log_ar_diff > {max_log_ar_diff:.3f}) "
                f"or invalid resolution."
            )

        # Convert the lists contained in each bucket to a numpy array
        buckets = [np.array(buckets[i]) for i in range(len(buckets))]

        if not buckets:
            raise ValueError(
                "No images were assigned to any buckets. Check "
                "dataset resolutions and bucketing parameters."
            )
        return buckets

    def fit_img2bucket(self, img, bucket_res):
        """
        Reshapes the image to fit the target bucket resolution while preserving
        aspect ratio, then applies random cropping if needed.

        Args:
            img: Input image as PIL Image object
            bucket_res: Target bucket resolution as (width, height) tuple

        Returns:
            Resized and potentially cropped image as PIL Image object
        """
        img_w, img_h = img.size
        target_w, target_h = bucket_res

        # Calculate the scaling factors for width and height
        w_scale = target_w / img_w
        h_scale = target_h / img_h

        # Choose the larger scale to ensure the image covers the target dimensions
        scale = max(w_scale, h_scale)

        # Calculate new dimensions after scaling
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        # Resize the image using the calculated scale
        resized_img = img.resize((new_w, new_h), Image.LANCZOS)

        # If the resized image matches the target exactly, return it
        if new_w == target_w and new_h == target_h:
            return resized_img

        # Otherwise, perform a random crop
        # Calculate the crop boundaries
        max_x = new_w - target_w
        max_y = new_h - target_h

        # Choose random starting positions for the crop
        start_x = random.randint(0, max(0, max_x))
        start_y = random.randint(0, max(0, max_y))

        # Perform the crop
        # torchvision.transforms.v2.RandomCrop for posible optim
        cropped_img = resized_img.crop(
            (start_x, start_y, start_x + target_w, start_y + target_h)
        )

        return cropped_img

    def _create_index_mapping(self) -> List[Tuple[int, int]]:
        """
        Create a flat index mapping for assigned images.
        Maps dataset index (0 to N-1) to (bucket_idx, original_img_idx).

        Returns:
            List of (bucket_idx, original_img_idx) tuples.
        """
        mapping = []
        for bucket_idx, bucket_content in enumerate(self.buckets):
            # empty buckets are ignored
            for original_img_idx in bucket_content:
                mapping.append((bucket_idx, original_img_idx))
        # Sort mapping by original_img_idx to potentially improve locality
        # if the dataloader doesn't shuffle.
        mapping.sort(key=lambda x: x[1])
        return mapping

    def __len__(self) -> int:
        """Return the total number of ASSIGNED samples."""
        return self.index_mapping.shape[0]  # len(self.index_mapping)

    def __getitem__(self, idx: int) -> Tuple[int, int, torch.Tensor, str]:
        """
        Get a sample prepared for encoding.

        Args:
            idx: Index of the sample (from 0 to len(self)-1).

        Returns:
            Tuple of (original_img_idx, bucket_idx, image_tensor, prompt_str, start_token, end_token)
        """
        # Map dataset index to bucket and original image index
        bucket_idx, original_img_idx = self.index_mapping[idx]
        # Convert numpy types back to standard Python int if needed downstream
        bucket_idx = int(bucket_idx)
        original_img_idx = int(original_img_idx)

        img_path = self.paths[original_img_idx]
        bucket_resolution = tuple(
            map(int, self.bucket_resolutions[bucket_idx])
        )  # self.bucket_resolutions[bucket_idx]

        # Load image and prompt
        try:
            img, prompt = self.load_entry(img_path, self.label_ext)
        except Exception as e:
            print(f"Error loading {img_path}: {e}. Returning dummy data.")
            # Return dummy data of expected types to avoid crashing DataLoader
            # Find a valid bucket resolution to create dummy tensor
            dummy_res = tuple(map(int, self.bucket_resolutions[0]))
            dummy_tensor = torch.zeros(
                (3, dummy_res[1], dummy_res[0]), dtype=torch.float32
            )
            # Use -1 to signal an error downstream if needed
            return dummy_tensor, "", -1, -1, -1, -1

        # Fit image to bucket and transform
        img_w, img_h = img.size
        print(
            f"Bucket ratio {self.log_aspect_ratios[bucket_idx]} and img ratio {math.log(img_w / img_h)}"
        )
        img = self.fit_img2bucket(img, bucket_resolution)
        # Converts a PIL Image (H x W x C) to a Tensor of shape (C x H x W).
        img_tensor = self.transform(img)  # Apply the transform

        # CHANGED: Retrieve token delimiter information
        start_token, end_token = -1, -1  # Default values
        if self.df_tokens is not None and not self.already_tokenized:
            try:
                base_filename = os.path.basename(img_path)
                # Split the filename from its extension to get the clean ID.
                img_id = os.path.splitext(base_filename)[0]

                # Look up token info using the pre-indexed DataFrame
                token_info = self.df_tokens.loc[img_id]
                start_token = int(token_info["general_start_token"])
                end_token = int(token_info["general_end_token"])
            except KeyError:
                # This occurs if the image ID is not in the CSV
                print(f"Warning: ID {img_id} not found in token file.")
            except Exception as e:
                print(f"Warning: Error getting tokens for {img_id}: {e}")

        # if already_tokenized then prompt is a dict with 3 keys, prefix, suffix and general
        return (
            img_tensor,
            prompt,
            original_img_idx,
            bucket_idx,
            start_token,
            end_token,
        )

    def get_prompts_for_assigned_indices(
        self, original_indices: List[int]
    ) -> Tuple[List[int], List[str]]:
        """
        Efficiently retrieves prompts for a given list of original image
        indices that are part of the assigned dataset samples using parallel processing.

        Args:
            original_indices: A list of integer indices corresponding to the
                            original `self.paths` list.

        Returns:
            A tuple containing:
            - A list of the original indices for which prompts were
            successfully loaded.
            - A list of the corresponding prompt strings.
        """

        # Define a worker function to load a prompt given an index
        def _load_prompt_worker(idx):
            try:
                img_path = self.paths[idx]
                prompt = self._load_prompt(img_path, self.label_ext)
                return idx, prompt, None  # Return index, prompt, and no error
            except IndexError:
                return idx, "", f"Original index {idx} out of bounds for paths."
            except Exception as e:
                return idx, "", f"Error loading prompt for index {idx}: {e}"

        loaded_indices = []
        prompts = []
        print(f"Loading prompts for {len(original_indices)} indices...")

        # Use ThreadPoolExecutor for parallel I/O operations
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            futures = [
                executor.submit(_load_prompt_worker, idx) for idx in original_indices
            ]

            # Process results as they complete
            for future in tqdm(
                futures,
                desc="Loading prompts",
                total=len(original_indices),
                leave=False,
            ):
                idx, prompt, error = future.result()
                if error is None:
                    loaded_indices.append(idx)
                    prompts.append(prompt)
                else:
                    print(f"Warning: {error}")

        if len(loaded_indices) != len(original_indices):
            print(
                f"Warning: Successfully loaded prompts for "
                f"{len(loaded_indices)} out of {len(original_indices)} "
                f"requested indices."
            )

        return loaded_indices, prompts

    def get_prepared_batch_by_indices(
        self, batch_original_indices: List[int], bucket_idx: int
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Loads, prepares (fits to bucket, transforms), and batches images and
        prompts for a given list of original indices belonging to the same
        specified bucket using parallel processing.

        Args:
            batch_original_indices: List of original image indices for the batch.
            bucket_idx: The index of the bucket these images belong to.

        Returns:
            A tuple containing:
            - A batch of image tensors (torch.Tensor, shape [B, C, H, W]).
            - A list of corresponding prompt strings (List[str]).
        """
        # Get the target resolution for this bucket (convert numpy ints to py ints)
        bucket_resolution = tuple(map(int, self.bucket_resolutions[bucket_idx]))

        # Define a worker function to process a single image
        def _process_image_worker(original_idx):
            img_path = self.paths[original_idx]
            try:
                # Load the image and prompt using the existing method
                img, prompt = self.load_entry(img_path, self.label_ext)

                # Fit image to the specified bucket resolution
                img = self.fit_img2bucket(img, bucket_resolution)

                # Apply transformations (e.g., ToTensor, Normalize)
                img_tensor = self.transform(img)

                # CHANGED: Retrieve token info within the worker
                start_token, end_token = -1, -1
                if self.df_tokens is not None and not self.already_tokenized:
                    try:
                        base_filename = os.path.basename(img_path)
                        # Split the filename from its extension to get the clean ID.
                        img_id = os.path.splitext(base_filename)[0]

                        # Look up token info using the pre-indexed DataFrame
                        token_info = self.df_tokens.loc[img_id]
                        start_token = int(token_info["general_start_token"])
                        end_token = int(token_info["general_end_token"])
                    except KeyError:
                        # This occurs if the image ID is not in the CSV
                        print(f"Warning: ID {img_id} not found in token file.")
                    except Exception as e:
                        print(f"Warning: Error getting tokens for {img_id}: {e}")

                return (img_tensor, prompt, start_token, end_token, original_idx, None)
            except Exception as e:
                return None, None, None, None, original_idx, str(e)

        img_tensors, prompts, start_tokens, end_tokens = [], [], [], []

        # Use ThreadPoolExecutor for parallel I/O and image processing
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            futures = [
                executor.submit(_process_image_worker, idx)
                for idx in batch_original_indices
            ]

            # Process results as they complete
            for future in tqdm(
                futures,
                desc=f"Processing images for bucket {bucket_idx}",
                total=len(batch_original_indices),
                leave=False,
            ):
                (img_tensor, prompt, start, end, o_idx, err) = future.result()
                if err is None:
                    img_tensors.append(img_tensor)
                    prompts.append(prompt)
                    start_tokens.append(start)
                    end_tokens.append(end)
                else:
                    print(f"Error processing index {o_idx}: {err}")

        # If no images were successfully processed, return empty batch
        if not img_tensors:
            # Determine expected shape for an empty tensor
            c, h, w = 3, bucket_resolution[1], bucket_resolution[0]  # Assume 3 channels
            # Use dtype from the class instance
            empty_tensor = torch.empty((0, c, h, w), dtype=self.dtype)
            print(
                f"Warning: No samples could be processed for batch in "
                f"bucket {bucket_idx}. Returning empty batch."
            )
            return empty_tensor, [], [], []

        # Stack the list of tensors into a single batch tensor
        try:
            return torch.stack(img_tensors), prompts, start_tokens, end_tokens

        except RuntimeError as e:
            print(
                f"Error stacking tensors for bucket {bucket_idx}. "
                f"Inconsistent shapes? {e}"
            )
            # Return empty tensor
            c, h, w = 3, bucket_resolution[1], bucket_resolution[0]
            empty_tensor = torch.empty((0, c, h, w), dtype=self.dtype)
            return empty_tensor, [], [], []

    def get_single_item_prepared(self, original_idx: int, bucket_idx: int):
        """Helper for fallback batch loading - prepares one item."""
        img_path = self.dataset.paths[original_idx]
        bucket_resolution = self.dataset.bucket_resolutions[bucket_idx]
        img, prompt = self.dataset.load_entry(img_path, self.dataset.label_ext)
        img = self.dataset.fit_img2bucket(img, bucket_resolution)
        img_tensor = self.dataset.transform(img)
        # Return structure similar to __getitem__ but without indices
        # CHANGED: Retrieve token delimiter information
        start_token, end_token = -1, -1
        if self.df_tokens is not None and not self.already_tokenized:
            try:
                base_filename = os.path.basename(img_path)
                # Split the filename from its extension to get the clean ID.
                img_id = os.path.splitext(base_filename)[0]
                token_info = self.df_tokens.loc[img_id]
                start_token = int(token_info["general_start_token"])
                end_token = int(token_info["general_end_token"])
            except KeyError:
                print(f"Warning: ID {img_id} not found in token file.")
            except Exception as e:
                print(f"Warning: Error getting tokens for {img_id}: {e}")

        return (img_tensor, prompt, original_idx, bucket_idx, start_token, end_token)

    def max_diff_ratio(self, buckets, paths, bucket_res, log_aspect_ratios):
        max_diff = 0.0
        idx = 0
        bucket_idx = 0
        for i, bucket in enumerate(buckets):
            for j in range(len(bucket)):
                sample = self.load_entry(paths[bucket[j]])
                # resize img
                img = self.fit_img2bucket(sample[0], bucket_res[i])
                img_w, img_h = sample[0].size
                diff = log_aspect_ratios[bucket_idx] - math.log(img_w / img_h)
                if abs(diff) > max_diff:
                    max_diff = diff
                    idx = bucket[j]
                    bucket_idx = i
                    print(f"New max idx:{idx}")

        sample = self.load_entry(paths[idx])
        print(paths[idx])
        print(f"Max diff {max_diff}")
        # resize img
        img = self.fit_img2bucket(sample[0], bucket_res[bucket_idx])
        img_w, img_h = sample[0].size
        print(
            f"Bucket ratio {log_aspect_ratios[bucket_idx]} and img ratio {math.log(img_w / img_h)}"
        )

        fig, ax1 = plt.subplots(1, 2)
        ax1[0].imshow(img)
        ax1[0].title.set_text("Cropped")
        ax1[1].imshow(sample[0])
        ax1[1].title.set_text("Original")
        plt.show()

    def get_image_id(self, idx):
        # the self.paths contains Path objects
        booru_id = self.paths[idx].stem
        return booru_id

    def get_prompt_from_id(self, booru_id: int):
        """Returns prompt from booru id.

        Args:
            booru_id (int): id inside danbooru dataset

        Returns:
            prompt: str or dictionary containing the prompt of the given img
        """
        idx = self.booru_id_to_idx[booru_id]
        image_path = self.paths[idx]
        prompt = self._load_prompt(image_path, self.label_ext)
        return prompt


# (Handles encoding loop, H5 sharding, and JSON metadata)


class LatentEncoder:
    """
    Handles encoding a dataset using VAE and CLIP, saving to sharded H5
    files (sharded by size) with bucket-specific groups, and creating a
    JSON metadata index.
    """

    def __init__(
        self,
        dataset: "LatentEncodingDataset",  # Use quotes for forward reference
        vae,
        text_encoder,
        tokenizer,
        output_dir: str,
        # 2K samples approx 1 GB of memory
        samples_per_shard: int = 4000,  # Approx samples per shard file
        batch_size: int = 8,
        num_workers: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
        # Max length should be the highest tier value.
        length_tiers: List[int] = [77, 152, 227],
        h5_compression: Optional[str] = "gzip",
        # Add batch size for tokenization precomputation
        tokenization_batch_size: int = 1024,
        cache_text_embeds: bool = False,
        store_tokenized_captions: bool = True,
        already_tokenized: bool = True,
        min_sample_count: int = 8,
        aesthetic_csv_path: Optional[str] = None,
    ):
        self.dataset = dataset
        self.vae = vae  # .to(device).eval()
        self.text_encoder = text_encoder  # .to(device).eval()
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.samples_per_shard = samples_per_shard  # Using sample count
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.dtype = dtype
        # Ensure tiers are sorted and store the overall max length
        self.length_tiers = sorted(length_tiers)
        self.max_length = self.length_tiers[-1]  # Overall max for truncation
        self.h5_compression = h5_compression
        self.tokenization_batch_size = tokenization_batch_size
        self.cache_text_embeds = cache_text_embeds
        self.store_tokenized_captions = store_tokenized_captions
        self.already_tokenized = already_tokenized
        self.min_sample_count = min_sample_count

        # Store bucket resolutions for easy access during encoding
        self.bucket_resolutions = dataset.bucket_resolutions
        # Precompute latent shapes for each bucket
        self.bucket_latent_shapes = self._precompute_latent_shapes()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.aesthetic_df = None
        if aesthetic_csv_path:
            print(f"Loading aesthetic tier data from {aesthetic_csv_path}...")
            try:
                # Read the CSV and set the 'id' column as the index for
                # fast lookups. The ID is read as a string to match the
                # booru_id type from the dataset.
                df = pd.read_csv(aesthetic_csv_path, dtype={"id": str})
                df = df.drop_duplicates(subset=["id"], keep="last")
                self.aesthetic_df = df.set_index("id")

                def get_tier_num(tier):
                    if pd.isna(tier):
                        return -1
                    return AESTHETIC_LABEL.get(tier, -1)

                self.aesthetic_df["final_tier_num"] = self.aesthetic_df[
                    "final_tier"
                ].apply(get_tier_num)
                print(
                    f"Successfully loaded {len(self.aesthetic_df)} "
                    "aesthetic tier entries."
                )
            except FileNotFoundError:
                print(
                    f"Warning: Aesthetic tier file not found at "
                    f"{aesthetic_csv_path}. Proceeding without it."
                )
            except Exception as e:
                print(
                    f"Warning: Failed to load aesthetic tier file: {e}. "
                    "Proceeding without it."
                )

    def _precompute_latent_shapes(self) -> List[Tuple[int, int, int]]:
        """
        Calculates the expected latent shape (C, H, W) for each bucket.
        Assumes VAE downsampling factor of 8.
        """
        # Get VAE latent channel count (usually 4 for SD 1.5)
        # Run a dummy tensor through the encoder to find C if needed
        dummy_input = torch.randn(1, 3, 512, 512, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            dummy_latent = torch.randn(
                1, 4, 64, 64
            )  # self.vae.encode(dummy_input) # Or direct output
        latent_channels = dummy_latent.shape[1]
        del dummy_input, dummy_latent
        if self.device == "cuda":
            torch.cuda.empty_cache()

        shapes = []
        vae_scale_factor = 8  # Common for SD VAEs
        for width, height in self.bucket_resolutions:
            width = int(width)
            height = int(height)

            latent_height = height // vae_scale_factor
            latent_width = width // vae_scale_factor
            shapes.append((latent_channels, latent_height, latent_width))
        print(f"Precomputed latent shapes for {len(shapes)} buckets.")
        return shapes

    def _determine_tier_from_length(self, length: int) -> int:
        """Assigns a length tier based on token count."""
        for tier_limit in self.length_tiers:
            if length <= tier_limit:
                return tier_limit
        # If length exceeds the max tier, assign to the highest tier
        return self.length_tiers[-1]

    def precompute_prompt_lengths_and_tiers(
        self, exclude_indices: Optional[Set[int]] = None
    ):
        """
        Calculates token length and assigns tiers for all valid prompts
        in the dataset, optionally excluding already processed indices.
        """
        if exclude_indices is None:
            exclude_indices = set()

        print("Pre-calculating prompt lengths and tiers...")
        # Maps original_idx -> {'tier': Y}
        prompt_metadata = {}
        # We need original indices and prompts for samples that were assigned
        # Use the dataset's index_mapping
        indices_to_process = [
            item[1]
            for item in self.dataset.index_mapping
            if item[1] not in exclude_indices
        ]
        num_valid_samples = len(indices_to_process)

        if num_valid_samples == 0:
            return {}  # Nothing to process

        print(f"Fetching prompts for {num_valid_samples} assigned samples...")
        try:
            # Ideal: A method that returns prompts only for assigned indices
            original_indices, prompts = self.dataset.get_prompts_for_assigned_indices(
                indices_to_process
            )
        except AttributeError:
            # Fallback: Load individually (less efficient)
            print(
                "Warning: Dataset lacks efficient prompt fetching. "
                "Loading prompts individually."
            )
            prompts = []
            original_indices = []
            for bucket_idx, original_idx in tqdm(
                self.dataset.index_mapping, desc="Loading prompts"
            ):
                original_idx = int(original_idx)
                try:
                    _, prompt = self.dataset.load_entry(
                        self.dataset.paths[original_idx], self.dataset.label_ext
                    )
                    prompts.append(prompt)
                    original_indices.append(original_idx)
                except Exception as e:
                    print(
                        f"Skipping prompt for index {original_idx} "
                        f"due to load error: {e}"
                    )
            if not prompts:
                raise RuntimeError("Failed to load any prompts.")

        print(f"Tokenizing {len(prompts)} prompts...")
        # Batch tokenization for efficiency
        for i in tqdm(
            range(0, len(prompts), self.tokenization_batch_size),
            desc="Tokenizing prompts",
        ):
            batch_prompts = prompts[i : i + self.tokenization_batch_size]
            batch_idxs = original_indices[i : i + self.tokenization_batch_size]

            if not self.already_tokenized:
                # Tokenize to get lengths after potential truncation
                # Use max_length (overall max) for truncation consistency
                inputs = self.tokenizer(
                    batch_prompts,
                    padding=False,  # Don't pad here, just count tokens
                    truncation=True,
                    max_length=self.max_length,  # Truncate to overall max
                    return_length=True,
                )
                # 'length' includes special tokens if tokenizer adds them
                lengths = inputs["length"]
            else:
                lengths = []
                for i, prompt in enumerate(batch_prompts):
                    length = len(prompt["prefix_tokens"])
                    length += len(prompt["general_tokens"])
                    length += len(prompt["suffix_tokens"])
                    lengths.append(length)

            for idx, length in zip(batch_idxs, lengths):
                # Determine tier based on the token count
                tier = self._determine_tier_from_length(length)
                prompt_metadata[idx] = {"tier": tier}

        print(f"Finished pre-calculating tiers for {len(prompt_metadata)} prompts.")
        # Add check: ensure all assigned indices got metadata
        if len(prompt_metadata) != num_valid_samples:
            print(
                f"Warning: Metadata count ({len(prompt_metadata)}) "
                f"differs from assigned samples ({num_valid_samples}). "
                f"Check for loading/tokenization errors."
            )
        return prompt_metadata

    def _load_and_prepare_cache_state(self):
        """
        Checks for an existing cache, loads its state, and prepares all
        necessary variables to resume the encoding process by appending.

        Returns:
            A tuple containing the state needed to append new data:
            - all_sample_info_existing (List[Dict]): The full list of sample
              metadata from the existing cache.
            - last_original_idx (int): The highest original_idx from the
              existing cache, used as an offset for new samples.
            - shard_idx_counter (int): The index of the shard file to
              write to (either continuing an existing one or starting a new one).
            - samples_processed_in_shard (int): The number of samples
              already present in the current shard file.
            - bucket_tier_counters (defaultdict): A dictionary mapping
              [bucket_idx][tier] to the next available index_in_tier.
        """
        metadata_path = self.output_dir / "metadata.json"
        if not metadata_path.exists():
            print("No existing cache found. Starting a new one.")
            return [], 0, 0, defaultdict(lambda: defaultdict(int))

        print(f"Found existing cache at {metadata_path}. Resuming...")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        all_sample_info = metadata.get("sample_mapping", [])
        if not all_sample_info:
            print("Cache metadata is empty. Starting fresh.")
            return [], 0, 0, defaultdict(lambda: defaultdict(int))

        # Find the highest original_idx to create an offset for new data
        # last_original_idx = max(
        #     s["original_idx"] for s in all_sample_info
        # )

        # Determine sharding state from the last entry
        last_sample = all_sample_info[-1]
        last_shard_file = last_sample["shard_file"]
        shard_idx_counter = int(last_shard_file.split("_")[-1].split(".")[0])

        samples_in_last_shard = sum(
            1 for s in all_sample_info if s["shard_file"] == last_shard_file
        )

        # If the last shard is full, we start a new one.
        if samples_in_last_shard >= self.samples_per_shard:
            shard_idx_counter += 1
            samples_processed_in_shard = 0
        else:
            samples_processed_in_shard = samples_in_last_shard

        # Determine the next index to write to for each bucket/tier group
        bucket_tier_counters = defaultdict(lambda: defaultdict(int))
        for sample in all_sample_info:
            b_idx, tier = sample["bucket_idx"], sample["tier"]
            # The next index is one greater than the current max index.
            current_max = bucket_tier_counters[b_idx][tier]
            next_idx = sample["idx_in_tier"] + 1
            if next_idx > current_max:
                bucket_tier_counters[b_idx][tier] = next_idx

        print(
            f"Resuming from shard {shard_idx_counter}. "
            f"{samples_processed_in_shard} samples in current shard."
        )
        # print(f"Last original_idx was {last_original_idx}. "
        #       "New indices will be offset accordingly.")

        return (
            all_sample_info,
            shard_idx_counter,
            samples_processed_in_shard,
            bucket_tier_counters,
        )

    def encode_dataset(self):
        """
        Encode the entire dataset. If a cache exists, appends the new
        data; otherwise, creates a new cache.
        """
        # 1. Load existing cache state if available.
        (
            all_sample_info,
            shard_idx_counter,
            samples_processed_in_shard,
            bucket_tier_sample_counters,
        ) = self._load_and_prepare_cache_state()
        # print(
        #     f"Last idx: {last_original_idx}"
        #     f"All sample info {all_sample_info}"
        #     f"Samples processed {samples_processed_in_shard}"
        #     f"Bucket tier sample counters {bucket_tier_sample_counters}"
        # )

        # CHANGED: Extract already processed indices to prevent re-encoding
        # the previous implementation assumed that original samples were deleted
        # from the root dir to reduce disk space
        exclude_indices = set()
        if all_sample_info:
            processed_booru_ids = {str(s["booru_id"]) for s in all_sample_info}
            for booru_id in processed_booru_ids:
                if booru_id in self.dataset.booru_id_to_idx:
                    exclude_indices.add(self.dataset.booru_id_to_idx[booru_id])
            print(
                f"Excluding {len(exclude_indices)} already processed samples from encoding."
            )

        # CHANGED: Pass exclude_indices to properly skip existing data
        prompt_metadata = self.precompute_prompt_lengths_and_tiers(
            exclude_indices=exclude_indices
        )

        # Group samples from the NEW dataset.
        grouped_samples = self._group_samples_by_bucket_and_tier(
            prompt_metadata, exclude_indices=exclude_indices
        )
        print("Sample grouping complete.")

        # --- H5 Sharding Setup ---
        total_samples_to_encode = sum(
            len(indices)
            for bucket_groups in grouped_samples.values()
            for indices in bucket_groups.values()
        )
        if total_samples_to_encode == 0:
            print("Error: No samples available for encoding after grouping.")
            return

        num_shards = math.ceil(total_samples_to_encode / self.samples_per_shard)
        print(
            f"Encoding {total_samples_to_encode} samples into "
            f"approximately {num_shards} shards (target "
            f"{self.samples_per_shard} samples/shard)..."
        )
        print(f"Cache text embeddings: {self.cache_text_embeds}")
        print(f"Store tokenized captions: {self.store_tokenized_captions}")

        h5_file = None
        # --- Updated H5 Handles: Nested dict per shard ---
        # h5_datasets[bucket_idx][tier]['latents'/'embeds'] = h5py.Dataset
        h5_datasets = defaultdict(lambda: defaultdict(dict))

        # This list will store metadata for ONLY the new samples.
        new_sample_info = []

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        global_sample_idx = 0

        try:
            # 3. Iterate through groups and encode tier-aware batches
            pbar = tqdm(total=total_samples_to_encode, desc="Encoding Batches")
            for bucket_idx in sorted(grouped_samples.keys()):
                latent_shape = self.bucket_latent_shapes[bucket_idx]

                for tier in sorted(grouped_samples[bucket_idx].keys()):
                    sample_indices_in_group = grouped_samples[bucket_idx][tier]
                    # print(f"Bucket id {bucket_idx} Tier {tier} : {len(sample_indices_in_group)} elements")
                    # Embedding shape depends on the tier
                    embed_dim = self.text_encoder.config.n_embd
                    embed_shape = (tier, embed_dim)
                    token_shape = (tier,)

                    # Create batches *from this specific group*
                    for i in range(0, len(sample_indices_in_group), self.batch_size):
                        # 1. --- Shard Management ---
                        if h5_file is None:
                            fname = f"latents_shard_{shard_idx_counter:05d}.h5"
                            shard_path = self.output_dir / fname
                            mode = "a" if samples_processed_in_shard > 0 else "w"

                            if mode == "a" and not shard_path.exists():
                                print(
                                    f"\nWarning: Shard file {shard_path} "
                                    "not found for appending. Creating a "
                                    "new file. This may indicate a "
                                    "corrupted cache."
                                )
                                mode = "w"

                            h5_file = h5py.File(shard_path, mode)
                            print(f"\nOpening shard: {shard_path} in mode '{mode}'")
                            h5_datasets.clear()
                            # bucket_tier_sample_counters.clear()
                            # shard_idx_counter += 1

                        # 2. --- Data Fetching & Encoding ---
                        batch_original_indices = sample_indices_in_group[
                            i : i + self.batch_size
                        ]
                        current_batch_size = len(batch_original_indices)

                        # --- Fetch data for this batch ---
                        # Assumes dataset has an efficient way to get data by index
                        # This needs implementation in LatentEncodingDataset
                        try:
                            # Ideal: Fetch transformed images and prompts
                            img_tensors, prompts, start_tokens, end_tokens = (
                                self.dataset.get_prepared_batch_by_indices(
                                    batch_original_indices, bucket_idx
                                )
                            )
                        except AttributeError:
                            # Fallback: Load and transform individually (SLOW)
                            print(
                                "Warning: Dataset lacks efficient batch "
                                "fetching. Loading/transforming samples "
                                "individually for batch."
                            )
                            img_tensors_list = []
                            prompts = []
                            valid_indices_in_batch = []
                            for orig_idx in batch_original_indices:
                                try:
                                    # Reuse __getitem__ logic carefully
                                    img_tensor, prompt, _, _ = (
                                        self.dataset.get_single_item_prepared(
                                            orig_idx, bucket_idx
                                        )
                                    )
                                    img_tensors_list.append(img_tensor)
                                    prompts.append(prompt)
                                    valid_indices_in_batch.append(orig_idx)
                                except Exception as e:
                                    print(
                                        f"Error preparing sample "
                                        f"{orig_idx} for batch: {e}"
                                    )
                            if not img_tensors_list:
                                continue  # Skip empty batch
                            img_tensors = torch.stack(img_tensors_list)
                            batch_original_indices = valid_indices_in_batch
                            current_batch_size = len(batch_original_indices)
                            # End Fallback

                        aesthetic_tiers_list = []
                        if self.aesthetic_df is not None:
                            for idx in batch_original_indices:
                                booru_id = self.dataset.get_image_id(idx)
                                try:
                                    # Use .loc for fast lookup
                                    tier_val = self.aesthetic_df.loc[
                                        booru_id, "final_tier_num"
                                    ]
                                    aesthetic_tiers_list.append(int(tier_val))
                                except KeyError:
                                    # Handle cases where the ID is not in the CSV
                                    print(
                                        f"Warning: booru_id {booru_id} not "
                                        "found in aesthetic CSV. "
                                        "Defaulting to -1."
                                    )
                                    aesthetic_tiers_list.append(-1)

                        # Move image tensors to device
                        img_tensors = img_tensors.to(self.device, dtype=self.dtype)

                        # --- Encode Batch (remains similar, ensure dtype is handled) ---
                        with torch.no_grad():
                            # VAE Encoding - Use .sample() or direct output as appropriate
                            image_latents = self.vae.encode(img_tensors)
                            # Convert latents to float32 numpy for H5 storage
                            image_latents_np = (
                                image_latents.detach().cpu().float().numpy()
                            )

                            # Text processing
                            text_embeddings_np = None
                            # tokenized_captions_np = None
                            if self.cache_text_embeds:
                                # Text Encoding
                                text_embeddings = encode_text_batch(
                                    prompts,
                                    self.text_encoder,
                                    self.tokenizer,
                                    max_length=tier,
                                    device=self.device,
                                )
                                # print(f"Tier {tier} Text embeddings shape {text_embeddings.shape} and Latents shape {image_latents.shape}")
                                # Convert embeddings to float32 numpy for H5 storage
                                text_embeddings_np = (
                                    text_embeddings.detach().cpu().float().numpy()
                                )

                        # 3. --- H5 Dataset Initialization (On-Demand) ---
                        if tier not in h5_datasets[bucket_idx]:
                            self._initialize_h5_datasets_for_tier(
                                h5_file,
                                h5_datasets,
                                bucket_idx,
                                tier,
                                latent_shape,
                                embed_shape,
                                token_shape,
                            )

                        # 4. --- Batch Data Writing to H5 ---
                        # The start_idx is the current count for this group,
                        # loaded from the previous state.
                        start_idx = bucket_tier_sample_counters[bucket_idx][tier]
                        self._write_batch_to_h5(
                            h5_datasets,
                            bucket_idx,
                            tier,
                            start_idx,
                            current_batch_size,
                            image_latents_np,
                            text_embeddings_np,
                            prompts,
                            start_tokens,
                            end_tokens,
                            token_shape,
                        )
                        # 5. --- Metadata Collection ---
                        for k in range(current_batch_size):
                            # This is the index from the *new* dataset instance
                            # new_dataset_original_idx = batch_original_indices[k]

                            new_sample_info.append(
                                {
                                    # Create a globally unique original_idx
                                    # "original_idx": (
                                    #     last_original_idx + 1 +
                                    #     new_dataset_original_idx
                                    # ),
                                    "shard_file": h5_file.filename.split("/")[-1],
                                    "bucket_idx": bucket_idx,
                                    "tier": tier,
                                    # Index within the specific dataset in the shard
                                    "idx_in_tier": start_idx + k,
                                    "booru_id": self.dataset.get_image_id(
                                        batch_original_indices[k]
                                    ),
                                }
                            )
                            if self.aesthetic_df is not None:
                                # take always the last element
                                new_sample_info[-1]["aesthetic_tier"] = (
                                    aesthetic_tiers_list[k]
                                )

                        # 6. --- Counter Updates ---
                        bucket_tier_sample_counters[bucket_idx][tier] += (
                            current_batch_size
                        )
                        samples_processed_in_shard += current_batch_size
                        global_sample_idx += current_batch_size
                        pbar.update(current_batch_size)

                        # Check if shard is full
                        if samples_processed_in_shard >= self.samples_per_shard:
                            h5_file.close()
                            h5_file = None
                            samples_processed_in_shard = 0  # Reset for next shard
                            shard_idx_counter += 1

                        # --- Memory Management ---
                        try:
                            del image_latents, img_tensors
                            del image_latents_np, prompts
                            if self.cache_text_embeds:
                                del text_embeddings, text_embeddings_np
                            if self.device == "cuda":
                                torch.cuda.empty_cache()
                            gc.collect()
                        except:
                            print("Error when free memory: {e}")

        finally:
            # Close the last H5 file
            if h5_file is not None and h5_file.__bool__():
                print(f"Closing final shard: {h5_file.filename}")
                h5_file.close()
            # Explicitly clear large objects
            del h5_datasets, bucket_tier_sample_counters, grouped_samples
            gc.collect()

        # --- Save JSON Metadata (Updated Structure) ---
        # Combine the old and new metadata before saving the final file.
        combined_sample_info = all_sample_info + new_sample_info
        self._save_json_metadata(combined_sample_info)

        print(f"Encoding complete! {global_sample_idx} samples saved.")

    def encode_tag_weights(self, update_all_samples: bool = False):
        """
        Adds or updates the tag weight value to the existing H5 dataset.

        This method efficiently updates H5 shards with a 'tag_weight'
        dataset. It can either update all samples from scratch or append
        weights for newly added samples.

        Args:
            update_all_samples (bool): If True, all existing 'tag_weight'
                datasets across all shards will be deleted and recreated.
                This is useful to fix datasets that were created without
                a resizable maxshape, resolving the RuntimeError.
                If False, the method runs in its standard resumable mode.
        """
        # 1. Load existing cache state.
        metadata_path = self.output_dir / "metadata.json"
        if not metadata_path.exists():
            print("Error: metadata.json not found. Cannot add tag weights.")
            return

        with open(metadata_path, "r") as f:
            all_sample_info = json.load(f).get("sample_mapping", [])

        if not all_sample_info:
            print("No samples found in metadata. Nothing to do.")
            return

        # 2. Group all samples by their shard file. This is the key
        #    optimization to avoid opening/closing H5 files repeatedly.
        samples_by_shard = defaultdict(list)
        for sample in all_sample_info:
            samples_by_shard[sample["shard_file"]].append(sample)

        print(
            f"Found {len(all_sample_info)} samples across "
            f"{len(samples_by_shard)} shards."
        )

        # 3. If update_all_samples is True, delete all existing tag_weight
        #    datasets first. This allows them to be recreated correctly.
        if update_all_samples:
            print("Update mode: Deleting all existing 'tag_weight' datasets...")
            for shard_filename in tqdm(
                samples_by_shard.keys(), desc="Resetting Tag Weights"
            ):
                shard_path = self.output_dir / shard_filename
                if not shard_path.exists():
                    continue

                try:
                    with h5py.File(shard_path, "a") as h5_file:
                        # Iterate through all groups to find and delete
                        # the target dataset without altering others.
                        for b_group_name in h5_file:
                            if not b_group_name.startswith("bucket_"):
                                continue
                            bucket_group = h5_file[b_group_name]
                            for t_group_name in bucket_group:
                                if not t_group_name.startswith("tier_"):
                                    continue
                                tier_group = bucket_group[t_group_name]

                                # If the dataset exists, delete it.
                                if "tag_weight" in tier_group:
                                    del tier_group["tag_weight"]
                except Exception as e:
                    print(
                        f"\nWarning: Could not process shard "
                        f"{shard_filename} for deletion: {e}"
                    )
            print("--- Deletion of existing tag weights complete. ---")

        # 3. Define the worker function for asynchronous prompt fetching.
        def fetch_weight_worker(sample: Dict) -> Tuple[int, float]:
            """
            Fetches the tag weight for a single sample.
            Returns the index within the tier and the weight.
            """
            booru_id = sample["booru_id"]
            idx_in_tier = sample["idx_in_tier"]
            try:
                # The prompt is expected to be a dictionary here.
                prompt = self.dataset.get_prompt_from_id(booru_id)
                # Use .get() for safety, default to 1.0 if key is missing.
                tag_weight = float(prompt.get("tag_weight", 1.0))
                return idx_in_tier, tag_weight
            except KeyError as e:
                # Handle cases where prompt is not a dict or weight invalid
                # shouldn't happen as the weights must be computed for all data
                print(
                    f"Warning: Could not parse tag_weight for "
                    f"booru_id {booru_id}. Defaulting to 1.0. Error: {e}"
                )
                return None, None

        # 4. Iterate over each shard file to perform the updates.
        for shard_filename, samples_in_shard in tqdm(
            samples_by_shard.items(), desc="Updating Shards"
        ):
            shard_path = self.output_dir / shard_filename
            if not shard_path.exists():
                print(f"Warning: Shard file not found, skipping: {shard_path}")
                continue

            # Group samples within this shard by their destination group
            # (bucket and tier) for targeted H5 dataset updates.
            samples_by_group = defaultdict(list)
            for sample in samples_in_shard:
                group_key = (sample["bucket_idx"], sample["tier"])
                samples_by_group[group_key].append(sample)

            # Use 'a' mode to read existing structure and add new data.
            with h5py.File(shard_path, "a") as h5_file:
                # Process each bucket/tier group within the shard.
                for (b_idx, tier), group_samples in samples_by_group.items():
                    b_group_name = f"bucket_{b_idx}"
                    t_group_name = f"tier_{tier}"

                    try:
                        tier_group = h5_file[b_group_name][t_group_name]
                    except KeyError:
                        print(
                            f"Warning: Group {b_group_name}/{t_group_name} "
                            f"not found in {shard_filename}. Skipping."
                        )
                        continue

                    # Skip if already processed to make the process resumable.
                    # if 'tag_weight' in tier_group:
                    #     continue

                    # Fetch all tag weights for this group in parallel.
                    indices, weights = [], []
                    with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                        # Map the worker to all samples in the current group.
                        future_to_sample = {
                            executor.submit(fetch_weight_worker, s): s
                            for s in group_samples
                        }
                        # Collect results as they complete.
                        for future in future_to_sample:
                            idx, weight = future.result()
                            if idx is not None and weight is not None:
                                indices.append(idx)
                                weights.append(weight)

                    if not indices:
                        print(
                            f"Warning: No weights fetched for group "
                            f"{b_idx}/{tier} in {shard_filename}."
                        )
                        continue

                    # Determine the required size from a reference dataset.
                    # This ensures the tag_weight dataset matches the number
                    # of latents already cached for this group.
                    required_size = len(tier_group["latents"])
                    dataset_name = "tag_weight"

                    # Check if the dataset already exists and handle
                    # creation or resizing accordingly.
                    if dataset_name in tier_group:
                        # Dataset exists, so we get a handle to it.
                        tag_weight_dset = tier_group[dataset_name]
                        # Ensure it's large enough. This is crucial if
                        # more latents were added after the tag weights
                        # were first created.
                        if tag_weight_dset.shape[0] < required_size:
                            tag_weight_dset.resize((required_size, 1))
                    else:
                        # Dataset does not exist. Create it with the correct
                        # size and, most importantly, make it resizable by
                        # setting maxshape.
                        tag_weight_dset = tier_group.create_dataset(
                            dataset_name,
                            shape=(required_size, 1),
                            maxshape=(None, 1),  # Allows future resizing
                            dtype=np.float32,
                            compression=self.h5_compression,
                        )

                    # Prepare data for efficient batch writing.
                    indices_np = np.array(indices, dtype=int)
                    weights_np = np.array(weights, dtype=np.float32).reshape(-1, 1)

                    # Write all fetched data for this group in one operation
                    # using advanced (fancy) indexing.
                    tag_weight_dset[indices_np] = weights_np

        print("--- Tag weight encoding finished ---")

    # --- _save_json_metadata updated to include tier info ---
    def _save_json_metadata(self, all_sample_info: List[Dict]):
        """Saves the collected metadata into a JSON file."""
        metadata_path = self.output_dir / "metadata.json"
        print(f"Saving metadata to {metadata_path}...")

        # Gather bucket and tier information from the actual encoded samples
        bucket_tier_counts = defaultdict(lambda: defaultdict(int))
        present_buckets = set()
        present_tiers = set()
        for sample in all_sample_info:
            b_idx = sample["bucket_idx"]
            tier = sample["tier"]
            bucket_tier_counts[b_idx][tier] += 1
            present_buckets.add(b_idx)
            present_tiers.add(tier)

        bucket_info = []
        for i in sorted(list(present_buckets)):
            # Check if bucket index is valid (should be)
            if i >= len(self.dataset.bucket_resolutions):
                print(f"Warning: Found invalid bucket index {i} in metadata.")
                continue

            tier_counts = {
                t: bucket_tier_counts[i][t]
                for t in sorted(bucket_tier_counts[i].keys())
            }
            total_count = sum(tier_counts.values())

            bucket_info.append(
                {
                    "bucket_idx": i,
                    "resolution": tuple(map(int, self.dataset.bucket_resolutions[i])),
                    "latents_resolution": self.bucket_latent_shapes[i][1:],  # H, W
                    "latent_channels": self.bucket_latent_shapes[i][0],  # C
                    "log_aspect_ratio": self.dataset.log_aspect_ratios[i],
                    "total_count": total_count,
                    "tier_counts": tier_counts,  # Counts per tier in this bucket
                }
            )

        metadata = {
            "dataset_info": {
                "total_encoded_samples": len(all_sample_info),
                "original_assigned_image_count": self.dataset.index_mapping.shape[0],
                "samples_per_shard_target": self.samples_per_shard,
                "length_tiers_used": sorted(list(present_tiers)),
                "max_token_length_limit": self.max_length,
                "cache_text_embeds": self.cache_text_embeds,
                "store_tokenized_captions": self.store_tokenized_captions,
            },
            "bucket_info": bucket_info,
            # Sample mapping now points to bucket/tier specific index
            "sample_mapping": all_sample_info,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print("Metadata saved successfully.")

    def _group_samples_by_bucket_and_tier(
        self, prompt_metadata: Dict, exclude_indices: Optional[Set[int]] = None
    ):
        """
        Helper to group sample indices by bucket and tier, optionally
        excluding already processed indices.
        """
        if exclude_indices is None:
            exclude_indices = set()

        print("Grouping samples by bucket and tier...")
        # grouped_samples[bucket_idx][tier] = [original_idx1, ...]
        grouped_samples = defaultdict(lambda: defaultdict(list))
        unassigned_count = 0
        for bucket_idx, original_idx in self.dataset.index_mapping:
            if original_idx in exclude_indices:
                continue

            bucket_idx = int(bucket_idx)
            original_idx = int(original_idx)

            if original_idx in prompt_metadata:
                tier = prompt_metadata[original_idx]["tier"]
                grouped_samples[bucket_idx][tier].append(original_idx)
            else:
                # This sample's prompt failed loading/tokenization earlier
                unassigned_count += 1
        if unassigned_count > 0:
            print(
                f"Warning: {unassigned_count} samples could not be "
                f"assigned a tier due to earlier errors."
            )

        if self.min_sample_count > 0:
            # Step 1: Use a nested dictionary comprehension to filter out
            # tiers that do not meet the minimum sample count. This creates
            # a new inner dictionary for each bucket.
            filtered_tiers = {
                bucket_idx: {
                    tier: samples
                    for tier, samples in tiers.items()
                    if len(samples) >= self.min_sample_count
                }
                for bucket_idx, tiers in grouped_samples.items()
            }

            # Step 2: After filtering, some buckets may become empty. This
            # second comprehension removes any bucket that no longer
            # contains any valid tiers. An empty dictionary evaluates to
            # False in a boolean context.
            grouped_samples = {
                bucket_idx: tiers
                for bucket_idx, tiers in filtered_tiers.items()
                if tiers
            }

        return grouped_samples

    def _initialize_h5_datasets_for_tier(
        self,
        h5_file,
        h5_datasets,
        bucket_idx,
        tier,
        latent_shape,
        embed_shape,
        token_shape,
    ):
        """Initializes all necessary HDF5 datasets for a new tier."""
        b_group_name = f"bucket_{bucket_idx}"
        t_group_name = f"tier_{tier}"

        bucket_group = h5_file.require_group(b_group_name)
        if "resolution_width" not in bucket_group.attrs:
            res = self.bucket_resolutions[bucket_idx]
            bucket_group.attrs["resolution_width"] = res[0]
            bucket_group.attrs["resolution_height"] = res[1]

        tier_group = bucket_group.require_group(t_group_name)

        # Common datasets
        h5_datasets[bucket_idx][tier]["latents"] = tier_group.require_dataset(
            "latents",
            shape=(0, *latent_shape),
            maxshape=(None, *latent_shape),
            dtype=np.float32,
            chunks=(1, *latent_shape),
            compression=self.h5_compression,
        )

        if self.cache_text_embeds:
            h5_datasets[bucket_idx][tier]["embeds"] = tier_group.require_dataset(
                "text_embeddings",
                shape=(0, *embed_shape),
                maxshape=(None, *embed_shape),
                dtype=np.float32,
                chunks=(1, *embed_shape),
                compression=self.h5_compression,
            )

        # Conditional token datasets
        if self.store_tokenized_captions:
            if self.already_tokenized:
                vlen_dtype = h5py.vlen_dtype(np.int64)
                for part in ["prefix", "general", "suffix"]:
                    key = f"{part}_tokens"
                    h5_datasets[bucket_idx][tier][key] = tier_group.require_dataset(
                        key, shape=(0,), maxshape=(None,), dtype=vlen_dtype, chunks=True
                    )
            else:
                h5_datasets[bucket_idx][tier]["tokens"] = tier_group.require_dataset(
                    "text_tokens",
                    shape=(0, *token_shape),
                    maxshape=(None, *token_shape),
                    dtype=np.int64,
                    chunks=(1, *token_shape),
                    compression=self.h5_compression,
                )
                h5_datasets[bucket_idx][tier]["start_end_tokens"] = (
                    tier_group.require_dataset(
                        "start_end_tokens",
                        shape=(0, 2),
                        maxshape=(None, 2),
                        dtype=np.int64,
                        chunks=(1, 2),
                        compression=self.h5_compression,
                    )
                )

    def _write_batch_to_h5(
        self,
        h5_datasets,
        bucket_idx,
        tier,
        start_idx,
        batch_size,
        latents_np,
        embeds_np,
        prompts,
        start_tokens,
        end_tokens,
        token_shape,
    ):
        """Writes a full batch of data to the appropriate H5 datasets."""
        datasets = h5_datasets[bucket_idx][tier]
        end_idx = start_idx + batch_size

        # Write fixed-size arrays in a single, efficient batch operation
        ds_latents = datasets["latents"]
        ds_latents.resize((end_idx, *ds_latents.shape[1:]))
        ds_latents[start_idx:end_idx] = latents_np

        if self.cache_text_embeds and embeds_np is not None:
            ds_embeds = datasets["embeds"]
            ds_embeds.resize((end_idx, *ds_embeds.shape[1:]))
            ds_embeds[start_idx:end_idx] = embeds_np

        # Handle token writing
        if self.store_tokenized_captions:
            if self.already_tokenized:
                # For vlen, we resize once then write sample-by-sample
                # This is still far more efficient than resizing per sample
                ds_prefix = datasets["prefix_tokens"]
                ds_general = datasets["general_tokens"]
                ds_suffix = datasets["suffix_tokens"]

                ds_prefix.resize((end_idx,))
                ds_general.resize((end_idx,))
                ds_suffix.resize((end_idx,))

                for k, prompt_dict in enumerate(prompts):
                    idx = start_idx + k
                    ds_prefix[idx] = np.array(
                        prompt_dict["prefix_tokens"], dtype=np.int64
                    )
                    ds_general[idx] = np.array(
                        prompt_dict["general_tokens"], dtype=np.int64
                    )
                    ds_suffix[idx] = np.array(
                        prompt_dict["suffix_tokens"], dtype=np.int64
                    )
            else:
                # Tokenize on the fly and write as a batch
                tokens_np = tokenize_text_batch(
                    prompts, self.tokenizer, max_length=tier
                )
                start_end_np = np.array(
                    list(zip(start_tokens, end_tokens)), dtype=np.int64
                )

                ds_tokens = datasets["tokens"]
                ds_tokens.resize((end_idx, *token_shape))
                ds_tokens[start_idx:end_idx] = tokens_np

                ds_start_end = datasets["start_end_tokens"]
                ds_start_end.resize((end_idx, 2))
                ds_start_end[start_idx:end_idx] = start_end_np
