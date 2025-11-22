# Danbooru Dataset Generation Toolkit

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Workflow and Usage](#workflow-and-usage)
- [Configuration](#configuration)
- [References](#references)

## Overview

This repository provides a comprehensive, configurable pipeline for creating high-quality, specialized datasets for training text-to-image models. It is designed to process large-scale image collections of Danbooru to generate datasets tailored to specific characters, artists, and styles.

The primary goal is to facilitate model training, by preparing the data in an optimal format. This includes prompt upsampling, aesthetic classification, and pre-computing VAE and text encoder latents into a sharded HDF5 cache for maximum training efficiency.

## Features

*   **Targeted Dataset Creation**: Generate datasets focused on specific lists of artists and characters to improve model knowledge.
*   **Advanced Sampling**: Filter and sample images based on quality metrics (score, favorite count), ratings (general, sensitive, etc.), and aesthetic predictions.
*   **Prompt Upsampling**: Automatically enhances brief Danbooru tags using a powerful SwinV2 tagger, leading to richer prompts and better model convergence.
*   **Image Deduplication**: Identifies and isolates near-duplicate images to improve dataset quality.
*   **Latent Pre-computation**: Drastically speeds up model training by pre-computing VAE image latents and CLIP text embeddings.
*   **Efficient Storage (HDF5)**: Saves the final dataset into a sharded H5 file, optimized for fast I/O and handling massive datasets that don't fit in RAM. Supports bucketing for variable image aspect ratios.
*   **Tag Weighting**: Calculates and stores tag weights to address data imbalance, ensuring the model learns rare concepts effectively.
*   **End-to-End Pipeline**: A fully automated pipeline managed by a single configuration file and executable via a simple CLI.

## Repository Structure

*   `main.py`: The main CLI entry point, built with Typer.
*   `configs/`: Contains YAML configuration files for the pipeline.
*   `src/pipeline.py`: The core `DataPipeline` class that orchestrates all steps.
*   `src/core/`: Directory containing the logic for individual pipeline steps:
    *   `database_generation.py`: Generates data from an excel of characters and a text file of artists.
    *   `deduplication.py`: Deduplicates data using image hash algorithm, with automatic selection based on prompt lenght and score.
    *   `sampling.py`: Data filtering and sampling.
    *   `download.py`: Image downloader.
    *   `unified_aes_upsampler.py`: Unified aesthetic classification and prompt upsampling.
    *   `prompt_generator.py`: Final prompt file creation.
    *   `encode_latents.py`: VAE/CLIP encoding and H5 dataset creation.
*   `models/`: Directory to store required pre-trained models (VAE, CLIP, Tagger).

## Prerequisites

*   Python 3.12+
*   A GPU with CUDA for model inference (classification, encoding).
*   Key Python libraries: `typer`, `torch`, `transformers`, `h5py`, `pandas`, `Pillow`, `pyarrow`, `faiss`, `opencv-python`. You can install them via `pip` using the `requirements.txt` file.
*   The [aesthetic](https://github.com/deeplearningcafe/aesthetic) module is required for the quality classification, however any model predicting 4 classes (worst, worse, better, best) would work as well.
* The artists list is expected to be WebUI formatted (e.g. `yd \(orange maru\)`) while the characters .xlsx file must contain in the `character_tag` column the original danbooru formatting (e.g. `souryuu_asuka_langley`,`reze_(chainsaw_man)`).
*   **Required Models**:
    1.  **Stable Diffusion VAE & CLIP**: The VAE and CLIP text encoder (e.g., from SD 1.5).
    2.  **Tokenizer**: The CLIP tokenizer corresponding to the text encoder.
    3.  **SwinV2 Tagger**: A Danbooru tagger model (e.g., `SmilingWolf/wd-swinv2-tagger-v3`).
    4.  **Aesthetic Classifier**: A trained classifier model to predict aesthetic scores.
    Place these models in the `models/` directory as specified in your config file.
*   **Danbooru Parquet Files**: The pipeline reads data from the Danbooru dataset provided in Parquet format (e.g., `p1atdev/danbooru-2024`), but the generated data is always prioritized.

## Workflow and Usage

The entire pipeline is controlled via `main.py`. It is highly recommended to execute each step separatedly to check the logs and results for better tuning.

### Step 1: Configuration

Before running, create a configuration file (e.g., `configs/my_dataset_config.yaml`) or modify the default. This file defines all paths, sampling parameters, and model settings.

The `default_config.yaml` has some characters and tags as reference. For the `character_list` variable inside `sampling`, the tags should be in danbooru format (e.g. `souryuu_asuka_langley`, `reze_(chainsaw_man)`, `sena_ajisai`).

### Step 2: Running the Pipeline

All commands are run from your terminal.

1.  **Generate Prior Knowledge Base**
    This step processes your artist and character lists to create a unified database for sampling.
    ```bash
    python main.py generate-data --config-path configs/my_dataset_config.yaml
    ```

2.  **Sample Image IDs**
    Filters the main Danbooru parquet files and samples image IDs based on your criteria (quality, ratings, tags, and prior knowledge base).
    ```bash
    python main.py sample --config-path configs/my_dataset_config.yaml
    ```

3.  **Download Images**
    Downloads all images corresponding to the sampled IDs.
    ```bash
    python main.py download --config-path configs/my_dataset_config.yaml
    ```

4.  **Deduplicate Images**
    Finds and moves visually similar images to a separate folder for review.
    ```bash
    python main.py deduplicate --config-path configs/my_dataset_config.yaml
    ```

5.  **Classify & Upsample Prompts**
    Runs a unified process to get aesthetic scores and upsample tags for all downloaded images, saving results to CSV files. **First we need to create the prompt files from the dataset**, both arguments are False by default so you can omitt them.
    ```bash
    python main.py generate-prompts --config-path configs/my_dataset_config.yaml --create-json False --use-aesthetic False
    python main.py unified-aes-prompts --config-path configs/my_dataset_config.yaml
    ```
    As this step uses the gpu, the processing can take quite a lot of time for old gpus. For reference on a gtx 1660 super the maximum batch size is 32, and it takes around 6 hours for 200K samples.

6.  **Generate Final Prompt Files**
    Combines all metadata (original tags, upsampled tags, aesthetic scores) and creates the final `.json` caption files for each image.
    ```bash
    python main.py generate-prompts --config-path configs/my_dataset_config.yaml --create-json True --use-aesthetic True
    ```

7.  **Encode Latents into H5**
    This is the final step. It processes all images and their JSON captions, encodes them into VAE and CLIP latents, and saves them into a sharded H5 file ready for training.
    ```bash
    python main.py encode-latents --config-path configs/my_dataset_config.yaml
    ```
    This step is extremely slow on old gpus. For reference on a gtx 1660 super the maximum batch size is 32, and it takes around 17 hours for 200K samples.

8.  **Calculate and Store Tag Weights (Optional)**
    If you need to handle tag imbalance, run this step to calculate tag weights and add them to the H5 metadata. For disk space constrained environments, the idea is to delete the images after encoding them and leaving only the text prompts which are needed to compute the tag weights.
    ```bash
    python main.py encode-tag-weights --config-path configs/my_dataset_config.yaml
    ```

### All-in-One Command

To run the entire pipeline from sampling to latent encoding sequentially, use the `full-pipeline` command.

```bash
python main.py full-pipeline --config-path configs/my_dataset_config.yaml
```

## Configuration

The `config.yaml` file is central to the pipeline. Key sections include:
*   `parquet_path`: Path to your Danbooru parquet files.
*   `download_dir`: Where to save downloaded images and intermediate files.
*   `prior_data`: Paths to your artist/character lists.
*   `sampling`: Parameters controlling the number of samples, quality distribution, and tag-based filtering.
*   `prompt_upsampling`: Settings for the tagger model.
*   `latent_encoding`: **Crucial section** defining the output path for the H5 file, model paths (VAE, CLIP), and parameters for sharding, batching, and bucketing (`length_tiers`).

## References
*   **Danbooru2024 Dataset**: [p1atdev/danbooru-2024](https://huggingface.co/datasets/p1atdev/danbooru-2024)
*   **SwinV2 Tagger Model**: [SmilingWolf/wd-swinv2-tagger-v3](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3)
*   The aesthetic classification and prompt generation concepts are inspired by the work done by the [Waifu Diffusion](https://github.com/waifu-diffusion/aesthetic) team.
*   The custom aesthetic classification model's data and training is done with the following repository [Aesthetic](https://github.com/deeplearningcafe/aesthetic).

## Author
[aipracticecafe](https://github.com/deeplearningcafe)

## License
This project is licensed under the `Apache license 2.0`. Details are in the [LICENSE](LICENSE.txt) file. I don't own the data from the `danbooru` website.