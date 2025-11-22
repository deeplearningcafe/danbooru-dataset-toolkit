import pandas as pd
import numpy as np
import re
from typing import List, Tuple
import yaml

from src.core.sampling import create_quality_tiers
from src.utils.loader import load_prior_knowledge_df

def filter_df_for_debug_tags(
    df: pd.DataFrame,
    debug_tags: List[str]
) -> pd.DataFrame:
    """
    Filters a DataFrame to keep only rows containing specific character or
    artist tags.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        debug_tags (List[str]): A list of tags to search for.

    Returns:
        pd.DataFrame: The filtered DataFrame containing only rows that match
                      at least one of the debug tags.
    """
    if not debug_tags or df.empty:
        return df.copy()

    # Initialize a boolean Series with all False values. This will accumulate
    # the results of our search for each tag.
    combined_mask = pd.Series(False, index=df.index)

    print(f"\n--- Filtering DataFrame for {len(debug_tags)} debug tags ---")
    for tag in debug_tags:
        # Escape the tag to handle special regex characters like '(', ')'.
        escaped_tag = re.escape(tag)
        # Use word boundaries (\b) to ensure we match the whole tag.
        regex_pattern = f'\\b{escaped_tag}\\b'

        # Create masks to find the tag in character and artist columns.
        # na=False treats NaN values as not containing the tag.
        mask_char = df['tag_string_character'].str.contains(
            regex_pattern, case=False, na=False, regex=True
        )
        mask_artist = df['tag_string_artist'].str.contains(
            regex_pattern, case=False, na=False, regex=True
        )

        # Combine the masks using a logical OR. A row is kept if the tag
        # is found in either the character OR the artist string.
        # The result is then OR'd with the combined_mask from previous tags.
        combined_mask |= (mask_char | mask_artist)

    filtered_df = df[combined_mask].copy()
    print(f"--- Filtering complete. Found {len(filtered_df)} rows matching "
          f"the debug tags. ---\n")

    return filtered_df


def main():
    """
    Main execution function to run the debugging process.
    """
    # --- Configuration ---
    # Define the paths to your data files.
    config_path = "configs/default_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    DEBUG_CSV_PATH = config["debug_tags.csv"]

    # --- Step 1: Load Data ---
    # Load the debug CSV to get the list of tags we need to investigate.
    try:
        debug_df = pd.read_csv(DEBUG_CSV_PATH)
        debug_tags = debug_df['tag'].tolist()
        print(f"Successfully loaded {len(debug_tags)} tags from "
              f"'{DEBUG_CSV_PATH}'.")
    except FileNotFoundError:
        print(f"Error: Debug CSV file not found at '{DEBUG_CSV_PATH}'.")
        return

    # Load the main dataset using your custom function.
    # Note: This uses the placeholder function defined above.
    prior_df = load_prior_knowledge_df(
        config['prior_data']['output_csv_path'],
        f"{config['parquet_path']}/aes_2024.csv"
    )

    # --- Step 2: Filter for Debugging ---
    # Isolate the subset of the main DataFrame that contains the problematic
    # tags. This makes the debugging process much faster.
    debug_subset_df = filter_df_for_debug_tags(prior_df, debug_tags)

    if debug_subset_df.empty:
        print("No rows found matching the debug tags. Exiting.")
        return

    # --- Step 3: Run the Quality Tiering Function ---
    # Now, run the create_quality_tiers function on this small, targeted
    # subset. The verbose output will show exactly where the samples are
    # being filtered out.
    print("--- Running create_quality_tiers on the filtered subset ---")
    create_quality_tiers(debug_subset_df, verbose=True)
    print("\n--- Debugging process finished ---")


if __name__ == "__main__":
    main()
