import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer


def compute_lens(
    model_id: str, path: str, save_plot: str = None, no_show: bool = False
):
    target_path = Path(path)

    if not target_path.exists():
        print(f"Error: Path '{path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Find all .txt files (recursively or directly in folder)
    txt_files = list(target_path.glob("*.txt"))
    if not txt_files:
        # Check subdirectories if no files found in top-level
        txt_files = list(target_path.rglob("*.txt"))

    if not txt_files:
        print(f"Error: No .txt files found in '{path}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(txt_files)} text files. Loading tokenizer '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    lens = []
    print("Processing files and computing token lengths...")
    for file in txt_files:
        try:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                prompt = f.read()

            inputs = tokenizer(
                prompt, padding=False, truncation=False, return_length=True
            )

            raw_len = (
                inputs["length"][0]
                if isinstance(inputs["length"], list)
                else inputs["length"]
            )
            length_excl_bos = max(0, raw_len - 1)
            lens.append(length_excl_bos)
        except Exception as e:
            print(f"Warning: Failed to process {file}: {e}")

    if not lens:
        print("No valid prompt lengths were computed.", file=sys.stderr)
        sys.exit(1)

    total_prompts = len(lens)
    mean_len = np.mean(lens)
    median_len = np.median(lens)
    std_len = np.std(lens)
    min_len = np.min(lens)
    max_len = np.max(lens)
    quantiles = np.quantile(lens, [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

    print("\n--- Prompt Length Statistics ---")
    print(f"Total prompts analyzed: {total_prompts}")
    print(f"Mean length: {mean_len:.2f}")
    print(f"Median length: {median_len}")
    print(f"Standard Deviation: {std_len:.2f}")
    print(f"Min length: {min_len}")
    print(f"Max length: {max_len}")
    print(f"Quantiles (25%, 50%, 75%, 90%, 95%, 99%): {quantiles.round(2).tolist()}")

    plt.figure(figsize=(12, 6))
    plt.hist(lens, bins=50, color="skyblue", edgecolor="black")
    plt.title(f"Distribution of Prompt Token Lengths ({model_id})")
    plt.xlabel("Token Length (excluding BOS)")
    plt.ylabel("Number of Prompts")
    plt.grid(axis="y", alpha=0.75)

    if save_plot:
        plt.savefig(save_plot, bbox_inches="tight")
        print(f"Plot saved to '{save_plot}'")

    if not no_show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compute and plot token length distribution for text prompts using a Hugging Face tokenizer."
    )
    parser.add_argument(
        "model_id",
        type=str,
        help="Hugging Face model ID or local path (e.g., 'meta-llama/Llama-2-7b-hf', 'gpt2')",
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to directory containing .txt prompt files",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Optional file path to save histogram plot (e.g., 'histogram.png')",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the plot window (useful for headless/server environments)",
    )

    args = parser.parse_args()
    compute_lens(
        args.model_id, args.path, save_plot=args.save_plot, no_show=args.no_show
    )


if __name__ == "__main__":
    main()
