import os
import torch
from src.prompts.tagger import Tagger
from src.core.prompt_upsampler import (
    TaggerDataset,
    combine_prompts_intelligently,
    custom_collate_with_paths,
)
from aesthetic.aesthetic import AestheticClassifier
from ..prompts.prompt_utils import validate_upsampled_batch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import pandas as pd
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def unified_aes_upsampler(
    root_dir: str,
    aesthetic_labels_csv: str,
    upsampled_tags_path: str,
    tagger_model_dir: str,
    classifier_path: str,
    device: str = "cuda",
    batch_size: int = 32,
    num_workers: int = 14,
    prefetch_factor: int = 6,
    tag_threshold: float = 0.35,
):
    """
    Performs aesthetic classification and prompt upsampling in a single,
    unified pass to avoid redundant feature computation.
    """
    print("--- Starting Step 3 & 4: Unified Classification & Upsampling ---")
    print(
        f"Searching data in {root_dir}, with aesthetic_labels_csv {aesthetic_labels_csv} and upsampled_tags_path {upsampled_tags_path}"
    )
    # 1. Load Models
    print("Loading models for unified processing...")
    # Load the SwinV2 model which serves as both tagger and feature extractor
    tagger = Tagger(model_dir=tagger_model_dir, device=device, threshold=tag_threshold)
    # The Tagger class holds the swinv2 model
    feature_extractor = tagger.model
    feature_extractor.eval().to(device)

    # Load the separate aesthetic classifier head
    ckpt = torch.load(classifier_path, map_location=device)
    aesthetic_classifier = AestheticClassifier(
        feature_dim=ckpt["feature_dim"],
        num_classes=ckpt["num_classes"],
        hidden_dims=ckpt.get("hidden_dims", 512),
        dropout_rate=ckpt.get("dropout_rate", 0.1),
    )
    # Adjust state dict keys if they have the '_orig_mod.' prefix
    adjusted_state_dict = {
        k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()
    }
    aesthetic_classifier.load_state_dict(adjusted_state_dict)
    aesthetic_classifier.eval().to(device)
    print("All models loaded successfully.")

    # 2. Setup DataLoader
    print("Setting up dataset and DataLoader...")
    dataset = TaggerDataset(
        root=root_dir,
        num_workers=num_workers,
        tagger_input_size=tagger.input_size,
        tagger_means=tagger.expected_processor_config["image_mean"],
        tagger_stds=tagger.expected_processor_config["image_std"],
    )

    # CHANGED: Read already processed samples to allow resuming
    processed_ids = set()
    if os.path.exists(aesthetic_labels_csv) and os.path.exists(upsampled_tags_path):
        try:
            up_df = pd.read_csv(upsampled_tags_path)
            if "id" in up_df.columns:
                processed_ids.update(up_df["id"].astype(str).tolist())

            print(
                f"Found {len(processed_ids)} already processed samples. Filtering dataset..."
            )
            dataset.filter_processed_images(processed_ids)
        except Exception as e:
            print(f"Warning: Could not read existing CSVs for resuming: {e}")

    if len(dataset) == 0:
        print("All samples have already been processed. Exiting unified upsampler.")
        return

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": custom_collate_with_paths,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": prefetch_factor,
    }
    data_loader = DataLoader(dataset, **loader_kwargs)

    # 3. Process data in a single loop
    print(f"Starting unified processing for {len(dataset)} images...")
    classification_results = []
    upsampling_results = []
    idx2label = {0: "worst", 1: "worse", 2: "better", 3: "best"}
    output_suffix = "_upsampled"
    processed_count = 0
    error_count = 0
    skipped_batches = 0

    for batch in tqdm(data_loader, desc="Unified Processing"):
        if batch is None:
            skipped_batches += 1
            continue

        # Unpack batch - custom_collate_with_paths yields (tensor, [prompt], [path], [metadata])
        image_tensors, prompts, img_paths, metadata_batch = batch
        image_tensors = image_tensors.to(device)

        with torch.no_grad():
            # --- SINGLE FORWARD PASS ---
            # 1. Compute features once (most expensive step)
            features = feature_extractor.forward_features(image_tensors)

            # 2. Get pooled features for both tasks
            pooled_features = feature_extractor.head.global_pool(features)

            # --- Path 1: Aesthetic Classification ---
            aesthetic_logits = aesthetic_classifier(pooled_features)
            predictions = torch.argmax(aesthetic_logits, dim=1)

            # --- Path 2: Prompt Upsampling ---
            # Continue the forward pass for the tagger using pooled features
            tagger_logits = feature_extractor.head.fc(
                feature_extractor.head.drop(pooled_features)
            )
            # Tagger expects scores as numpy array on CPU
            batch_scores_np = torch.sigmoid(tagger_logits).cpu().numpy()

        # 4. Process and store results for the batch
        for i in range(len(prompts)):
            img_path = img_paths[i]
            row_metadata = metadata_batch[i]

            # Store classification result
            label = idx2label[predictions[i].item()]
            # just with the ID is enough
            classification_results.append(
                {
                    "relative_path": img_path.parent / img_path.name,
                    # "id": img_path.stem,
                    "aesthetic_label": label,
                }
            )

            # Store upsampling result
            original_prompt = prompts[i]
            image_scores = batch_scores_np[i]
            new_tags = tagger.get_tags_from_scores(image_scores, skip_rating=True)
            _, upsampled_tags = combine_prompts_intelligently(original_prompt, new_tags)

            # Validate and clean the upsampled tags using metadata and heuristics
            upsampled_tags = validate_upsampled_batch(upsampled_tags, row_metadata)

            upsampling_results.append(
                {"id": img_path.stem, "upsampled_tags": upsampled_tags}
            )
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

    # 5. Save all results to CSV files
    print("Saving results...")

    # CHANGED: Append to existing files or create new ones to prevent overwriting
    aes_mode = "a" if os.path.exists(aesthetic_labels_csv) else "w"
    up_mode = "a" if os.path.exists(upsampled_tags_path) else "w"
    pd.DataFrame(classification_results).to_csv(
        aesthetic_labels_csv, mode=aes_mode, header=(aes_mode == "w"), index=False
    )
    print(f"Classification results saved to {aesthetic_labels_csv}")

    pd.DataFrame(upsampling_results).to_csv(
        upsampled_tags_path, mode=up_mode, header=(up_mode == "w"), index=False
    )
    print(f"Upsampling results saved to {upsampled_tags_path}")

    print("Unified processing complete.")
