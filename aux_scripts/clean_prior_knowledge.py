import pandas as pd
import re
from src.prompts.prompt_utils import get_tags_from_knowledge_bases
import yaml

def clean_dataset_with_prior_knowledge(
    input_csv_path: str,
    knowledge_bases_paths: list[str],
    output_csv_path: str
):
    """
    Filters a large dataset CSV to keep only entries containing specific
    character or artist tags from a new, focused knowledge base.

    This function uses a fast, vectorized regex approach for high
    performance on large datasets.

    Args:
        input_csv_path (str): Path to the large, unfiltered CSV file.
        knowledge_bases_paths (list[str]): A list of paths to the new
                                           knowledge base files (e.g.,
                                           updated character/artist lists).
        output_csv_path (str): The path to save the cleaned CSV file.
    """
    # 1. Load the desired tags from the new knowledge base files. These are
    #    the tags we want to keep in the final dataset.
    print("--- Loading Prior Knowledge Tags ---")
    prior_knowledge_tags = get_tags_from_knowledge_bases(
        knowledge_bases_paths
    )
    if not prior_knowledge_tags:
        print("Error: No prior knowledge tags were found. Exiting.")
        return
    print(f"Loaded {len(prior_knowledge_tags)} unique tags to filter by.")

    # 2. Create a single regex pattern for efficient searching.
    # The pattern `\b(tag1|tag2|...)\b` matches any of the whole tags.
    # `re.escape` handles tags with special characters.
    regex_pattern = r'\b(' + '|'.join(
        re.escape(tag) for tag in prior_knowledge_tags
    ) + r')\b'

    # 3. Load the large dataset from the CSV file into a pandas DataFrame.
    print(f"\n--- Loading Dataset from '{input_csv_path}' ---")
    df = pd.read_csv(input_csv_path)
    print(f"Original dataset contains {len(df)} entries.")

    # 4. Filter the DataFrame using the vectorized regex search.
    print("\n--- Filtering Dataset ---")
    # Combine character and artist tags into a single column for searching.
    # Fill any potential NaN values with empty strings to prevent errors.
    search_series = (
        df['tag_string_character'].fillna('') + ' ' +
        df['tag_string_artist'].fillna('')
    )

    # `str.contains` with a regex pattern is highly optimized for this task
    # and much faster than iterating row-by-row.
    mask = search_series.str.contains(regex_pattern, regex=True, na=False)

    cleaned_df = df[mask].copy()
    print(f"Filtered dataset now contains {len(cleaned_df)} entries.")

    # 5. Save the newly cleaned and filtered DataFrame to a new CSV file.
    print(f"\n--- Saving Cleaned Dataset to '{output_csv_path}' ---")
    cleaned_df.to_csv(output_csv_path, index=False)
    print("Dataset cleaning complete.")


if __name__ == '__main__':
    config_path = "configs/default_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    INPUT_CSV_PATH = config['prior_data']['output_csv_path']

    # Paths to your UPDATED knowledge bases with the desired tags.
    # For example, a smaller, more focused list of characters and artists.
    KNOWLEDGE_BASE_PATHS = config['prior_data']['knowledge_bases_paths']

    # Path where the new, smaller, and cleaned CSV file will be saved.
    OUTPUT_CSV_PATH = "cleaned_prior_knowledge.csv"

    # --- Execution ---
    clean_dataset_with_prior_knowledge(
        input_csv_path=INPUT_CSV_PATH,
        knowledge_bases_paths=KNOWLEDGE_BASE_PATHS,
        output_csv_path=OUTPUT_CSV_PATH
    )