import typer
from src.pipeline import DataPipeline

app = typer.Typer(
    help="A CLI for the Danbooru dataset preparation pipeline."
)

@app.command()
def generate_data(
    config_path: str = typer.Option(
        "configs/default_config.yaml",
        help="Path to the configuration file."
    )
):
    """Runs the prior data generation step."""
    pipeline = DataPipeline(config_path)
    pipeline.run_generate_dataset()

@app.command()
def analyze_data(
    config_path: str = typer.Option(
        "configs/default_config.yaml",
        help="Path to the configuration file."
    ),
    sampled: bool = typer.Option(False),
):
    """Runs the data analysis step."""
    pipeline = DataPipeline(config_path)
    pipeline.run_analyze_dataset(sampled)

@app.command()
def sample(
    config_path: str = typer.Option(
        "configs/default_config.yaml",
        help="Path to the configuration file."
    )
):
    """Runs the data sampling and tiering step."""
    pipeline = DataPipeline(config_path)
    pipeline.run_sampling()

@app.command()
def download(
    config_path: str = typer.Option(
        "configs/default_config.yaml",
        help="Path to the configuration file."
    )
):
    """Downloads images from the sampled dataset CSV."""
    pipeline = DataPipeline(config_path)
    pipeline.run_download()

@app.command()
def deduplicate(
    config_path: str = typer.Option(
        "configs/default_config.yaml",
        help="Path to the configuration file."
    ),
    move_back: bool = typer.Option(False),
    automatic_check: bool = typer.Option(False),
):
    """Deduplicates images from the downloaded dataset"""
    pipeline = DataPipeline(config_path)
    pipeline.run_deduplication(move_back=move_back, automatic_check=automatic_check)

@app.command()
def classify(
    config_path: str = typer.Option(
        "configs/default_config.yaml",
        help="Path to the configuration file."
    )
):
    """Runs aesthetic classification on the downloaded images."""
    pipeline = DataPipeline(config_path)
    pipeline.run_classification()

@app.command()
def generate_prompts(
    config_path: str = typer.Option(
        "configs/default_config.yaml",
        help="Path to the configuration file."
    ),
    create_json: bool = typer.Option(False),
    use_aesthetic: bool = typer.Option(False)
):
    """Generates the final .txt prompt files for training."""
    pipeline = DataPipeline(config_path)
    pipeline.run_prompt_generation(create_json=create_json, use_aesthetic=use_aesthetic)

@app.command()
def unified_aes_prompts(
    config_path: str = typer.Option(
        "configs/default_config.yaml",
        help="Path to the configuration file."
    ),
):
    """Performs classification and prompt upsampling in a single step"""
    pipeline = DataPipeline(config_path)
    pipeline.run_unified_processing()

@app.command()
def upsample_prompts(
    config_path: str = typer.Option(
        "configs/default_config.yaml",
        help="Path to the configuration file."
    )
):
    """Upsamples prompts for downloaded images using a tagger."""
    pipeline = DataPipeline(config_path)
    pipeline.run_prompt_upsampling()

@app.command()
def encode_latents(
    config_path: str = typer.Option(
        "configs/default_config.yaml",
        help="Path to the configuration file."
    )
):
    """Encodes image/text pairs into a sharded H5 latent cache."""
    pipeline = DataPipeline(config_path)
    pipeline.run_latent_encoding()

@app.command()
def encode_tag_weights(
    config_path: str = typer.Option(
        "configs/default_config.yaml",
        help="Path to the configuration file."
    )
):
    """Encodes tags weights to the metadata json file."""
    pipeline = DataPipeline(config_path)
    pipeline.run_tag_weight_encoding()

@app.command()
def full_pipeline(
    config_path: str = typer.Option(
        "configs/default_config.yaml",
        help="Path to the configuration file."
    )
):
    """Runs all steps of the pipeline in sequence."""
    pipeline = DataPipeline(config_path)
    pipeline.run_generate_dataset()
    pipeline.run_sampling()
    pipeline.run_download()
    pipeline.run_prompt_generation(create_json=False, use_aesthetic=False)
    pipeline.run_unified_processing()
    pipeline.run_prompt_generation(create_json=True, use_aesthetic=True)
    pipeline.run_latent_encoding()
    pipeline.run_tag_weight_encoding()
    print("--- Full Pipeline Finished Successfully ---")

if __name__ == "__main__":
    app()