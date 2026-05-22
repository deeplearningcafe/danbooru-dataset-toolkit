import yaml
import pandas as pd
import datetime
from .utils import loader
from .core.sampling import filter_and_sample_by_quality, sample_face_dataset
from .core.download import Downloader
from .core.classification import ImageClassifier
from .core.prompt_generator import PromptGenerator
from .core.prompt_upsampler import upsample_prompts_batch
from .core.encode_latents import LatentEncodingDataset, LatentEncoder
from .core.unified_aes_upsampler import unified_aes_upsampler
from .core.deduplication import (
    deduplicate_images,
    automate_deduplication,
    create_exclusion_list,
)
from .core.database_generation import create_prior_knowledge_dataset
from .core.dataset_analysis import analyze_prior_knowledge_dataset
from .core.face_cropping import FaceCropper

import os
import random

# Get the absolute path of the current file (module_B.py)
current_file_path = os.path.abspath(__file__)

# Get the directory of the current file (folder_B)
current_dir = os.path.dirname(current_file_path)

# Get the parent directory (common_parent)
common_parent_dir = os.path.dirname(current_dir)

# Construct the path to folder_A
folder_A_path = os.path.join(common_parent_dir, "models")
print(folder_A_path)
from vae import Vae, VaeConfig
from clip import Clip, ClipConfig
from transformers import CLIPTokenizer
import torch
import numpy as np


class DataPipeline:
    """
    Orchestrates the entire data preparation pipeline from loading
    parquets to generating final prompts.
    """

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        print("Configuration loaded.")

        self.experiment_name = self.config.get("experiment_name", "experiment")
        date_str = datetime.datetime.now().strftime("%Y_%m_%d")
        self.reports_dir = os.path.join("reports", f"{self.experiment_name}_{date_str}")
        os.makedirs(self.reports_dir, exist_ok=True)
        print(f"All reports and CSVs will be saved to: {self.reports_dir}")

        # Override CSV paths to organize them into the reports directory
        s_config = self.config["sampling"]
        s_config["sampled_ids_csv"] = os.path.join(
            self.reports_dir,
            os.path.basename(s_config.get("sampled_ids_csv", "sampled_ids.csv")),
        )

        d_config = self.config["download"]
        d_config["exclusion_list_csv"] = os.path.join(
            self.reports_dir,
            os.path.basename(d_config.get("exclusion_list_csv", "exclude_ids.csv")),
        )

        c_config = self.config["classification"]
        c_config["aesthetic_labels_csv"] = os.path.join(
            self.reports_dir,
            os.path.basename(c_config.get("aesthetic_labels_csv", "labels.csv")),
        )

        pu_config = self.config["prompt_upsampling"]
        pu_config["upsampled_tags_path"] = os.path.join(
            self.reports_dir,
            os.path.basename(pu_config.get("upsampled_tags_path", "upsampled.csv")),
        )

        p_config = self.config["prompts"]
        p_config["final_tiers_csv"] = os.path.join(
            self.reports_dir,
            os.path.basename(p_config.get("final_tiers_csv", "final_tiers.csv")),
        )
        # Pass the reports_dir down to prompt generator config
        p_config["reports_dir"] = self.reports_dir
        encode_config = self.config["latent_encoding"]
        encode_config["latents_output_dir"] = os.path.join(
            self.reports_dir,
            encode_config.get("latents_output_dir", "latents"),
        )
        # set seeds
        torch.manual_seed(self.config["sampling"]["random_seed"])
        random.seed(self.config["sampling"]["random_seed"])
        np.random.seed(self.config["sampling"]["random_seed"])
        print(f"Seed used {self.config['sampling']['random_seed']}")

    def run_generate_dataset(self):
        """Creates a parquet file with all the samples from the characters and artists lists."""
        print("--- Starting Step 0: Prior Dataset Creation ---")
        create_prior_knowledge_dataset(
            knowledge_bases_paths=self.config["prior_data"]["knowledge_bases_paths"],
            output_csv_path=self.config["prior_data"]["output_csv_path"],
            max_workers=self.config["prior_data"]["max_workers"],
            batch_size=self.config["prior_data"]["batch_size"],
            character_list=self.config["sampling"].get("character_list", []),
            artist_list=self.config["sampling"].get("artist_list", []),
        )
        print(f"Dataset creation complete.")

    def run_analyze_dataset(self, sampled: bool = False):
        print("--- Starting Step 0: Analyze prior dataset ---")
        analyze_prior_knowledge_dataset(
            prior_df_path=self.config["prior_data"]["output_csv_path"],
            sampled_ids_path=self.config["sampling"]["sampled_ids_csv"]
            if sampled
            else None,
            aes_scores_csv_path=f"{self.config['parquet_path']}/aes_2024.csv",
            knowledge_bases_paths=self.config["prior_data"]["knowledge_bases_paths"],
            tokenizer_path=self.config["prompts"]["tokenizer_path"],
        )
        print(f"Dataset creation complete.")

    def run_sampling(self):
        """Loads data, samples it, and saves the sampled IDs."""
        print("--- Starting Step 1: Dataset Sampling ---")
        df = loader.load_all_parquets(
            self.config["parquet_path"],
            num_parquets=self.config["num_parquets"],
        )

        sampled_df, _ = filter_and_sample_by_quality(
            df,
            total_samples=self.config["sampling"]["total_samples"],
            quality_percentages=self.config["sampling"]["quality_distribution"],
            ratings_percentage=self.config["sampling"]["rating_distribution"],
            prior_knowledge_path=self.config["prior_data"]["output_csv_path"],
            artists_txt=self.config["prior_data"]["knowledge_bases_paths"][0],
            aes_scores_csv_path=f"{self.config['parquet_path']}/aes_2024.csv",
            random_seed=self.config["sampling"]["random_seed"],
            output_csv=self.config["sampling"]["sampled_ids_csv"],
            skip_tags=self.config["sampling"]["skip_tags"],
            include_tags=self.config["sampling"]["include_tags"],
            exclude_path=self.config["download"]["exclusion_list_csv"],
            character_list=self.config["sampling"].get("character_list", []),
            artist_list=self.config["sampling"].get("artist_list", []),
            is_lora=self.config["sampling"].get("is_lora", False),
            reports_dir=self.reports_dir,
            verbose=True,
        )
        print(f"Sampling complete. {len(sampled_df)} IDs saved.")

    def run_download(self):
        """Downloads images based on the sampled IDs CSV."""
        print("--- Starting Step 2: Image Downloading ---")
        csv_path = self.config["sampling"]["sampled_ids_csv"]
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print("Error: Sampled IDs file not found. Run sampling first.")
            return

        # Get start_index from config to allow resuming. Defaults to 0.
        start_index = self.config["download"].get("start_index", 0)

        if start_index >= len(df):
            print(
                f"Start index ({start_index}) is beyond the dataframe "
                f"length ({len(df)}). Nothing to download."
            )
            return

        is_lora = self.config["sampling"].get("is_lora", False)
        print(
            f"Starting download from index {start_index}. And with lora as: {is_lora}"
        )

        downloader = Downloader(
            max_workers=self.config["download"]["max_workers"],
            timeout=self.config["download"]["timeout"],
            max_downloads=self.config["download"].get("max_downloads", None),
        )
        downloader.download_images(
            df,
            self.config["download_dir"],
            csv_path,
            output_csv_path=self.config["download"].get("exclusion_list_csv", None),
            start_index=start_index,
            character_list=self.config["sampling"]["character_list"] if is_lora else [],
        )
        print("Download step finished.")

    def run_deduplication(self, move_back: bool = False, automatic_check: bool = False):
        """
        Finds and moves similar images to a separate directory for review.
        This step helps improve dataset quality by removing near-duplicates.
        """
        print("--- Starting Step: Image Deduplication ---")
        # Get deduplication config, allowing it to be disabled
        dedup_config = self.config.get("deduplication", {})
        download_config = self.config.get("download", {})

        # Ensure required config keys are present
        if "output_dir" not in dedup_config:
            raise ValueError("Missing 'output_dir' in deduplication config.")

        deduplicate_images(
            root_dir=self.config["download_dir"],
            output_dir=dedup_config["output_dir"],
            threshold=dedup_config.get("threshold", 5),
            batch_size=dedup_config.get("batch_size", 32),
            move_back=move_back,
        )
        if automatic_check:
            automate_deduplication(
                deduplication_dir=dedup_config["output_dir"],
                root_dir=self.config["download_dir"],
                sampled_ids_csv=self.config["sampling"]["sampled_ids_csv"],
                prior_df_path=self.config["prior_data"]["output_csv_path"],
            )
        # If files were moved back, curation is done. We now generate an
        # exclusion list of images that were permanently deleted so they
        # are not sampled in future runs.
        if move_back:
            print("--- Creating exclusion list for deleted images ---")
            if "exclusion_list_csv" not in download_config:
                print(
                    "Warning: 'exclusion_list_csv' not in deduplication "
                    "config. Skipping exclusion list creation."
                )
            else:
                create_exclusion_list(
                    sampled_ids_csv=self.config["sampling"]["sampled_ids_csv"],
                    download_dir=self.config["download_dir"],
                    start_index=download_config.get("start_index", 0),
                    max_downloads=download_config.get("max_downloads", None),
                    output_csv_path=download_config["exclusion_list_csv"],
                    num_workers=download_config.get("max_workers", 4),
                )

        print("--- Image Deduplication Complete ---")

    def run_classification(self):
        """Classifies all downloaded images."""
        print("--- Starting Step 3: Image Classification ---")
        classifier = ImageClassifier(
            batch_size=self.config["classification"]["batch_size"]
        )
        classifier.load_models(
            self.config["classification"]["feature_extractor_path"],
            self.config["classification"]["classifier_path"],
        )
        classifier.label_images_in_directory(
            self.config["download_dir"],
            self.config["classification"]["aesthetic_labels_csv"],
        )
        print("Classification complete.")

    def run_prompt_generation(
        self, create_json: bool = True, use_aesthetic: bool = True
    ):
        """Generates final prompt files for training."""
        print("--- Starting Step 4: Prompt Generation ---")
        print(
            f"Using create_json {create_json} and use_aesthetic {use_aesthetic} parameters"
        )
        prompt_gen = PromptGenerator(self.config["prompts"])
        random.seed(
            self.config["sampling"]["random_seed"],
        )

        if use_aesthetic:
            final_df = prompt_gen.load_booru_df(
                parquet_path=self.config["parquet_path"],
                classifier_labels=[
                    self.config["classification"]["aesthetic_labels_csv"]
                ],
                sampled_ids_csv=self.config["sampling"]["sampled_ids_csv"],
                # create_json=False -> prompts have not been upsampled yet
                upsampled_prompts_csv=self.config["prompt_upsampling"][
                    "upsampled_tags_path"
                ]
                if create_json
                else None,
                prior_knowledge_path=self.config["prior_data"]["output_csv_path"],
                num_parquets=self.config["num_parquets"],
            )
        else:
            final_df = prompt_gen.load_booru_df_sampled(
                parquet_path=self.config["parquet_path"],
                sampled_ids_csv=self.config["sampling"]["sampled_ids_csv"],
                prior_knowledge_path=self.config["prior_data"]["output_csv_path"],
                num_parquets=self.config["num_parquets"],
            )

        if not final_df.empty:
            if use_aesthetic:
                final_df = prompt_gen.refine_tiers_and_assign_final_class(final_df)
            prompt_gen.create_prompt_files(
                final_df,
                self.config["download_dir"],
                model_path=self.config["models_dir"],
                create_json_files=create_json,
            )
            print("Prompt generation complete.")
        else:
            print("Final DataFrame is empty. No prompts generated.")

    def run_prompt_upsampling(self, anime_faces: bool = False):
        """
        Upsamples prompts for all downloaded images using a tagger.
        """
        print("--- Starting Step 4: Prompt Upsampling ---")
        dataset_root = (
            self.config["download_dir"]
            if not anime_faces
            else self.config["face_cropping"]["output_dir"]
        )
        output_csv_path = self.config["prompt_upsampling"]["upsampled_tags_path"]
        output_csv_path = (
            output_csv_path
            if not anime_faces
            else output_csv_path.replace(".csv", "_faces.csv")
        )
        upsample_prompts_batch(
            dataset_root=dataset_root,
            tagger_model_path=self.config["prompt_upsampling"]["tagger_model_dir"],
            output_csv_path=output_csv_path,
            skip_rating=True if not anime_faces else False,
        )
        print("Prompt upsampling complete.")

    def run_unified_processing(self):
        """
        Performs aesthetic classification and prompt upsampling in a single,
        unified pass to avoid redundant feature computation.
        """
        print("--- Starting Step 3 & 4: Unified Classification & Upsampling ---")
        unified_aes_upsampler(
            root_dir=self.config["download_dir"],
            aesthetic_labels_csv=self.config["classification"]["aesthetic_labels_csv"],
            upsampled_tags_path=self.config["prompt_upsampling"]["upsampled_tags_path"],
            tagger_model_dir=self.config["prompt_upsampling"]["tagger_model_dir"],
            classifier_path=self.config["classification"]["classifier_path"],
            device=self.config["device"],
            batch_size=self.config["classification"]["batch_size"],
            tag_threshold=self.config["prompt_upsampling"]["tag_threshold"],
        )

    def run_latent_encoding(self):
        """
        Encodes the prepared dataset into a sharded H5 latent cache
        using the configuration specified in the YAML file.
        """
        print("--- Starting Step 5: Latent Encoding ---")
        le_config = self.config["latent_encoding"]

        # 1. Initialize the dataset for encoding.
        #    The input directory is the output of the prompt generation.
        dataset = LatentEncodingDataset(
            root=self.config["download_dir"],
            label_ext=le_config["label_ext"],
            already_tokenized=le_config["already_tokenized"],
            df_tokens_path=None,  # Assuming not used in this flow
        )
        print(f"Loaded {len(dataset)} samples for latent encoding.")

        # 2. Load the required models (VAE, CLIP, Tokenizer).
        print("Loading models for encoding...")
        tokenizer = CLIPTokenizer.from_pretrained(
            self.config["prompts"]["tokenizer_path"], local_files_only=True
        )

        text_encoder = (
            Clip.from_pretrained(ClipConfig, le_config["clip_path"])
            .to(torch.float32)
            .eval()
        )
        text_encoder.requires_grad_(False)

        vae = (
            Vae.from_pretrained(VaeConfig, le_config["vae_path"])
            .to(self.config["device"])
            .eval()
        )
        vae.requires_grad_(False)

        print("Models loaded successfully.")

        # 3. Initialize and run the LatentEncoder.
        #    All parameters are pulled from the config for flexibility.
        encoder = LatentEncoder(
            dataset=dataset,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            output_dir=le_config["latents_output_dir"],
            samples_per_shard=le_config["samples_per_shard"],
            batch_size=le_config["batch_size"],
            num_workers=le_config["num_workers"],
            device=self.config["device"],
            length_tiers=le_config["length_tiers"],
            h5_compression=le_config["h5_compression"],
            cache_text_embeds=le_config["cache_text_embeds"],
            store_tokenized_captions=le_config["store_tokenized_captions"],
            min_sample_count=le_config["min_sample_count"],
            aesthetic_csv_path=self.config["prompts"]["final_tiers_csv"],
        )

        encoder.encode_dataset()
        print("Latent encoding complete.")

    def run_tag_weight_encoding(self):
        """
        Encodes the prepared dataset into a sharded H5 latent cache
        using the configuration specified in the YAML file.
        """
        print("--- Starting Step 5: Tag weight Encoding ---")
        le_config = self.config["latent_encoding"]
        prompt_gen = PromptGenerator(self.config["prompts"])

        final_df = prompt_gen.load_booru_df(
            parquet_path=self.config["parquet_path"],
            classifier_labels=[self.config["classification"]["aesthetic_labels_csv"]],
            sampled_ids_csv=self.config["sampling"]["sampled_ids_csv"],
            # create_json=False -> prompts have not been upsampled yet
            upsampled_prompts_csv=self.config["prompt_upsampling"][
                "upsampled_tags_path"
            ],
            prior_knowledge_path=self.config["prior_data"]["output_csv_path"],
            num_parquets=self.config["num_parquets"],
        )

        if not final_df.empty:
            final_df = prompt_gen.refine_tiers_and_assign_final_class(final_df)
            prompt_gen.count_and_weight_tags(
                final_df,
                root_dir=self.config["download_dir"],
                metadata_path=f"{le_config['latents_output_dir']}/metadata.json",
                num_workers=le_config["num_workers"],
            )
            print("Prompt generation complete.")
        else:
            print("Final DataFrame is empty. No prompts generated.")

        # 1. Initialize the dataset for encoding.
        #    The input directory is the output of the prompt generation.
        dataset = LatentEncodingDataset(
            root=self.config["download_dir"],
            label_ext=le_config["label_ext"],
            already_tokenized=le_config["already_tokenized"],
            df_tokens_path=None,  # Assuming not used in this flow
        )
        print(f"Loaded {len(dataset)} samples for latent encoding.")

        # 3. Initialize and run the LatentEncoder.
        #    All parameters are pulled from the config for flexibility.
        encoder = LatentEncoder(
            dataset=dataset,
            vae=None,
            text_encoder=None,
            tokenizer=None,
            output_dir=le_config["latents_output_dir"],
            samples_per_shard=le_config["samples_per_shard"],
            batch_size=le_config["batch_size"],
            num_workers=le_config["num_workers"],
            device=self.config["device"],
            length_tiers=le_config["length_tiers"],
            h5_compression=le_config["h5_compression"],
            cache_text_embeds=le_config["cache_text_embeds"],
            store_tokenized_captions=le_config["store_tokenized_captions"],
            min_sample_count=le_config["min_sample_count"],
        )

        encoder.encode_tag_weights()
        print("Latent encoding complete.")

    def run_sample_faces(self):
        """Samples images suitable for anime face extraction."""
        print("--- Starting Step: Face Dataset Sampling ---")
        df = loader.load_all_parquets(
            self.config["parquet_path"],
            num_parquets=self.config["num_parquets"],
        )
        sampled_df = sample_face_dataset(
            df, output_csv=self.config["sampling"]["sampled_ids_csv"], verbose=True
        )
        print(f"Face sampling complete. {len(sampled_df)} IDs saved.")

    def run_crop_faces(self, from_downloaded: bool = True):
        """
        Detects and crops faces. If from_downloaded is True, it filters
        the already downloaded dataset CSV. If False, it samples and downloads from scratch.
        """
        print("--- Starting Step: Face Cropping ---")
        crop_config = self.config.get("face_cropping", {})
        model_path = crop_config.get("model_path", "models/yolov8n-face.pt")
        output_dir = crop_config.get(
            "output_dir", self.config["download_dir"] + "_faces"
        )
        min_res = crop_config.get("min_resolution", 256)
        batch_size = crop_config.get("batch_size", 16)

        csv_path = self.config["sampling"]["sampled_ids_csv"]

        if from_downloaded:
            print("Option 1 selected: Using already downloaded images...")
            try:
                downloaded_df = pd.read_csv(csv_path)
                downloaded_df = downloaded_df.dropna(subset=["relative_path"])
            except FileNotFoundError:
                print(f"Error: {csv_path} not found. Cannot process from downloaded.")
                return

            # print("Loading parquets to get full metadata for filtering...")
            # full_df = loader.load_all_parquets(
            #     self.config["parquet_path"],
            # )
            print(
                f"\nStep 0: Loading prior knowledge from '{self.config['prior_data']['output_csv_path']}'..."
            )
            prior_knowledge_samples = pd.read_csv(
                self.config["prior_data"]["output_csv_path"], header=0, low_memory=False
            )
            prior_knowledge_samples = (
                prior_knowledge_samples.drop_duplicates().reset_index(drop=True)
            )

            # Merge to get metadata (tags, created_at, etc.) for the downloaded images
            merged_df = pd.merge(
                prior_knowledge_samples,
                downloaded_df[["id", "relative_path"]],
                on="id",
                how="inner",
            )

            # Save the filtered face dataset to a new CSV to avoid overwriting the main one
            faces_csv = csv_path.replace(".csv", "_faces.csv")
            face_df = sample_face_dataset(merged_df, output_csv=faces_csv, verbose=True)
        else:
            print("Option 2 selected: Downloading images from start...")
            self.run_sample_faces()
            self.run_download()

            try:
                face_df = pd.read_csv(csv_path)
                face_df = face_df.dropna(subset=["relative_path"])
            except FileNotFoundError:
                print(f"Error: {csv_path} not found after download.")
                return

        if face_df.empty:
            print("No valid images found for face cropping after filtering.")
            return

        artist_list = self.config.get("sampling", {}).get("artist_list", [])
        cropper = FaceCropper(
            model_path=model_path, min_res=min_res, allowed_artists=artist_list
        )
        cropper.process_dataframe(
            df=face_df,
            input_dir=self.config["download_dir"],
            output_dir=output_dir,
            batch_size=batch_size,
        )
        print(f"Face cropping complete. Saved to {output_dir}")
        print(
            "NOTE: You can now run the 'upsample-prompts' command pointing "
            f"to '{output_dir}' to generate fresh tags for the faces."
        )
