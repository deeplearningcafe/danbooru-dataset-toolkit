import pandas as pd
import yaml

def keep_best_duplicate(group):
    """
    Auxiliary function to decide which duplicated row to keep.

    This function takes a pandas DataFrame group (representing duplicated rows
    for a single 'id') and determines which row to keep based on the
    following criteria:
    1. The row with the longest 'tag_string'.
    2. If 'tag_string' lengths are equal, the row with the highest 'score'.
    3. If scores are also equal, the row with the highest 'fav_count'.

    Args:
        group (pd.DataFrame): A DataFrame containing duplicated rows for a
        single 'id'.

    Returns:
        pd.Series: The single row from the group that should be kept.
    """
    # Calculate the length of the 'tag_string' for each row in the group
    tag_string_lengths = group['tag_string'].str.len()

    # Find the maximum length of the 'tag_string' in the group
    max_len = tag_string_lengths.max()

    # Filter the group to only include rows with the maximum 'tag_string'
    # length
    candidates = group[tag_string_lengths == max_len]

    # If there's only one candidate, return it
    if len(candidates) == 1:
        return candidates.iloc[0]

    # If there are multiple candidates with the same max length, check the
    # 'score'
    max_score = candidates['score'].max()
    candidates = candidates[candidates['score'] == max_score]

    # If there's only one candidate after checking the score, return it
    if len(candidates) == 1:
        return candidates.iloc[0]

    # If there are still multiple candidates, check the 'fav_count' and
    # return the one with the highest value
    return candidates.loc[candidates['fav_count'].idxmax()]


def remove_duplicates(df_path):
    """
    Loads a DataFrame, removes duplicates based on the 'id' column,
    and keeps the best entry based on specific criteria.

    Args:
        df_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame with duplicates removed.
    """
    # Load the dataset
    df = pd.read_csv(df_path, header=0, low_memory=False)
    df = df.drop_duplicates().reset_index(drop=True)
    print("Original DataFrame:")
    print(df.shape)

    # Find all duplicated IDs
    duplicated_ids = df[df.duplicated(subset=['id'], keep=False)]['id'].unique()

    # Separate the DataFrame into duplicated and non-duplicated parts
    df_duplicates = df[df['id'].isin(duplicated_ids)]
    df_non_duplicates = df[~df['id'].isin(duplicated_ids)]

    # Group the duplicated rows by 'id' and apply the 'keep_best_duplicate'
    # function to each group.
    # The 'groupby' and 'apply' approach is efficient for this operation as
    # it processes each group of duplicates independently.
    df_deduplicated = df_duplicates.groupby('id', group_keys=False).apply(
        keep_best_duplicate
    )

    # Concatenate the non-duplicated rows with the processed duplicated rows
    final_df = pd.concat([df_non_duplicates, df_deduplicated]).reset_index(
        drop=True
    )
    print("\nCleaned DataFrame:")
    print(final_df.shape)
    final_df.to_csv(df_path, index=False)


if __name__ == '__main__':

    config_path = "configs/default_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Using the new function to remove duplicates with the specified logic
    remove_duplicates(config["prior_data"]["output_csv_path"])

