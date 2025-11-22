import numpy as np
import pandas as pd
import os
from typing import Tuple, Dict, List, Set, Union
from ..prompts.prompt_utils import get_tags_from_knowledge_bases
from ..utils.loader import load_prior_knowledge_df
import matplotlib.pyplot as plt

import seaborn as sns
from collections import Counter
import math
from tqdm import tqdm
from transformers import CLIPTokenizer


def plot_histogram(data, column, title, xlabel, ylabel="Frequency", bins=50):
    """Plots a histogram for a given column."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], bins=bins, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_bar_chart(counts, title, xlabel, ylabel="Count", top_n=20):
    """Plots a bar chart for the top N items in a Counter object."""
    if not counts:
        print(f"No data to plot for {title}")
        return
    common_items = counts.most_common(top_n)
    if not common_items:
        print(f"No common items to plot for {title}")
        return

    labels, values = zip(*common_items)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=list(values), y=list(labels), hue=list(labels), palette="viridis", legend=False)
    plt.title(title)
    plt.xlabel(ylabel) # Swapped for horizontal bar chart
    plt.ylabel(xlabel) # Swapped for horizontal bar chart
    plt.gca().invert_yaxis() # Display top item at the top
    plt.tight_layout()
    plt.show()

def parse_tags(tag_series, min_tag_len=1):
    """Parses a series of space-separated tag strings into a Counter."""
    tag_list = []
    for tags_str in tag_series.dropna():
        tags = str(tags_str).split()
        # Filter out very short or potentially problematic tags if needed
        tag_list.extend([tag for tag in tags if len(tag) >= min_tag_len])
    return Counter(tag_list)

def score_analysis(df):
    print("\n--- 1. Score Analysis ---")
    if 'score' in df.columns:
        # plot_histogram(df, 'score', 'Distribution of Image Scores', 'Score')
        print(f"Score statistics:\n{df['score'].describe()}")
    else:
        print("Column 'score' not found.")

    if 'aes_score' in df.columns:
        # aes_score might have NaNs or be out of a typical range, clip for viz
        df_aes = df.dropna(subset=['aes_score'])

        df_aes['aes_score_clipped'] = df_aes['aes_score'].clip(0, 10)
        # plot_histogram(df_aes, 'aes_score_clipped',
        #             'Distribution of Aesthetic Scores (Clipped 0-10)',
        #             'Aesthetic Score')
        print(f"\nAesthetic score statistics:\n{df['aes_score'].describe()}")
    else:
        print("Column 'aes_score' not found.")

    # if 'score' in df.columns and 'aes_score' in df.columns:
    #     plt.figure(figsize=(10, 6))
    #     # Use a sample for scatter plot if dataset is very large to avoid overplotting
    #     sample_df = df.sample(n=min(5000, len(df)), random_state=42)
    #     sns.scatterplot(data=sample_df, x='score', y='aes_score', alpha=0.5)
    #     plt.title('Score vs. Aesthetic Score')
    #     plt.xlabel('Community Score')
    #     plt.ylabel('Aesthetic Score')
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.show()
    #     # Correlation might be useful
    #     if pd.api.types.is_numeric_dtype(df['score']) and \
    #     pd.api.types.is_numeric_dtype(df['aes_score']):
    #         correlation = df[['score', 'aes_score']].corr()
    #         print(f"\nCorrelation between score and aes_score:\n{correlation}")

    if 'fav_count' in df.columns:
        # plot_histogram(df, 'fav_count', 'Distribution of Image Fav count', 'Fav count')
        print(f"Fav Count statistics:\n{df['fav_count'].describe()}")
    else:
        print("Column 'score' not found.")

def rating_imgs_analysis(df):
    # Section 2: Rating Analysis (rating)
    # Ratings (e.g., 's' for safe, 'q' for questionable, 'e' for explicit)
    print("\n--- 2. Rating Analysis ---")
    if 'rating' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x='rating', order=df['rating'].value_counts().index)
        plt.title('Distribution of Image Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.show()
        print(f"Rating distribution:\n{df['rating'].value_counts(normalize=True)}")
    else:
        print("Column 'rating' not found.")

    if 'quality_tier' in df.columns:
        print("\n--- 2. Quality Tiers Analysis ---")
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x='quality_tier', order=df['quality_tier'].value_counts().index)
        plt.title('Distribution of Image Quality Tiers')
        plt.xlabel('Quality Tiers')
        plt.ylabel('Count')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.show()
        print(f"Quality Tiers distribution:\n{df['quality_tier'].value_counts(normalize=True)}")
    else:
        print("Column 'quality_tier' not found.")


    # Section 3: Image Dimensions Analysis (image_width, image_height)
    print("\n--- 3. Image Dimensions Analysis ---")
    if 'image_width' in df.columns and 'image_height' in df.columns:
        image_pixels = (
                df["image_width"] * df["image_height"]
        )
        # Define the minimum resolution threshold.
        # we must use a lower than 500x500 to include low_res samples
        low_resolution = 512 * 512
        # Filter out images with a pixel count below the threshold.
        low_res_df = df[image_pixels < low_resolution]
        print(
            f"There are {len(low_res_df)} samples of low resolution."
            f" Which is {(len(low_res_df)/len(df))*100}% of the total."
        )

        # Scatter plot of width vs height
        plt.figure(figsize=(10,6))
        sns.scatterplot(data=low_res_df, x='image_width', y='image_height',
                        alpha=0.3, s=10) # s for marker size
        plt.title('Image Width vs. Height')
        plt.xlabel('Image Width (pixels)')
        plt.ylabel('Image Height (pixels)')
        plt.axvline(512, color='r', linestyle='--', label='512px Width')
        plt.axhline(512, color='g', linestyle='--', label='512px Height')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

        # Calculate aspect ratio
        # Add a small epsilon to avoid division by zero if height can be 0
        df['aspect_ratio'] = df['image_width'] / (df['image_height'] + 1e-6)
        # Cap aspect ratio for better visualization if there are extreme outliers
        df['aspect_ratio_clipped'] = df['aspect_ratio'].clip(0.2, 4.0)
        plot_histogram(df, 'aspect_ratio_clipped',
                    'Distribution of Aspect Ratios (Clipped 0.2-4.0)',
                    'Aspect Ratio (Width/Height)')
        print(f"\nAspect ratio statistics:\n{df['aspect_ratio'].describe()}")

    else:
        print("Columns 'image_width' or 'image_height' not found.")

# Type alias for clarity, consistent with potential use in LatentEncodingDataset
Resolution = Tuple[int, int]


def _generate_bucket_definitions(
    max_size: Resolution,
    min_size: int,
    divisible: int,
    base_res: Resolution,
    vae_factor: int,
    dim_limit: int
) -> Tuple[List[Resolution], List[float]]:
    """
    Generates bucket resolutions and their log aspect ratios.
    This is an adapted version of the logic from
    LatentEncodingDataset.generate_buckets for standalone analysis.

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
    max_tokens = (max_size[0] / vae_factor) * (max_size[1] / vae_factor)
    possible_resolutions: Set[Resolution] = set()

    # Generate Landscape-dominant buckets
    w = min_size
    while (w / vae_factor) * (min_size / vae_factor) <= max_tokens \
            and w <= dim_limit:
        h = min_size
        while (w / vae_factor) * ((h + divisible) / vae_factor) <= max_tokens \
                and (h + divisible) <= dim_limit:
            h += divisible
        if h >= min_size: # Ensure height is valid
            possible_resolutions.add((w, h))
        w += divisible

    # Generate Portrait-dominant buckets
    h = min_size
    while (min_size / vae_factor) * (h / vae_factor) <= max_tokens \
            and h <= dim_limit:
        w = min_size
        while ((w + divisible) / vae_factor) * (h / vae_factor) <= max_tokens \
                and (w + divisible) <= dim_limit:
            w += divisible
        if w >= min_size: # Ensure width is valid
            possible_resolutions.add((w, h))
        h += divisible

    # Ensure base resolution is included if it meets constraints
    if (base_res[0] / vae_factor) * (base_res[1] / vae_factor) <= max_tokens \
            and base_res[0] >= min_size and base_res[1] >= min_size \
            and base_res[0] <= dim_limit and base_res[1] <= dim_limit \
            and base_res[0] % divisible == 0 \
            and base_res[1] % divisible == 0:
        possible_resolutions.add(base_res)

    if not possible_resolutions:
        print("Warning: No valid bucket resolutions generated during analysis. "
              "Check generation parameters.")
        return [], []

    # Sort for consistency
    sorted_res_list = sorted(list(possible_resolutions),
                             key=lambda r: (r[0] * dim_limit - r[1]))

    log_aspect_ratios_list = [
        math.log(res_w / res_h) if res_h > 0 else 0
        for res_w, res_h in sorted_res_list
    ]

    print(f"Generated {len(sorted_res_list)} unique aspect ratio bucket "
          f"definitions for analysis.")

    return sorted_res_list, log_aspect_ratios_list


def analyze_dataframe_bucketing(
    df: pd.DataFrame,
    max_res_area: Resolution = (768, 512),
    min_size: int = 256,
    divisible: int = 64,
    base_res: Resolution = (512, 512),
    vae_factor: int = 8, # Standard VAE downscaling
    dim_limit: int = 1024,
    max_log_ar_diff: float = 0.3,
) -> Dict[Resolution, int]:
    """
    Analyzes a DataFrame to determine aspect ratio bucket distribution
    based on image dimensions.

    Args:
        df: Pandas DataFrame with 'image_width' and 'image_height' columns.
        max_res_area: Max resolution area for bucket generation.
        min_size: Min dimension for bucket generation.
        divisible: Dimension divisibility for bucket generation.
        base_res: Base resolution to include in buckets.
        vae_factor: VAE factor for bucket generation.
        dim_limit: Max dimension limit for bucket generation and image filtering.
        max_log_ar_diff: Max log aspect ratio difference for assignment.

    Returns:
        A dictionary mapping bucket resolution (tuple) to sample count.
    """
    if not ({'image_width', 'image_height'}.issubset(df.columns)):
        raise ValueError("DataFrame must contain 'image_width' and "
                         "'image_height' columns.")

    bucket_resolutions_list, log_aspect_ratios_list = \
        _generate_bucket_definitions(
            max_size=max_res_area,
            min_size=min_size,
            divisible=divisible,
            base_res=base_res,
            vae_factor=vae_factor,
            dim_limit=dim_limit
        )

    if not bucket_resolutions_list:
        print("Error: No bucket definitions were generated. "
              "Cannot perform analysis.")
        return {}

    # Initialize counts for each bucket definition
    # Using tuple(res) as dict keys
    bucket_counts: Dict[Resolution, int] = {
        tuple(res): 0 for res in bucket_resolutions_list
    }

    # For efficient lookup of bucket resolution by its index
    bucket_res_tuples_ordered = [tuple(res) for res in bucket_resolutions_list]
    log_ar_array = np.array(log_aspect_ratios_list, dtype=np.float32)

    assigned_count = 0
    pruned_invalid_dim_count = 0
    pruned_ar_diff_count = 0

    print(f"Analyzing {len(df)} images from DataFrame for bucketing...")
    # Iterate over DataFrame rows, more efficient than to_numpy for large DF
    # if only two columns are needed per row.
    for _, row in tqdm(df[['image_width', 'image_height']].iterrows(),
                       total=df.shape[0],
                       desc="Assigning images to buckets"):
        width = int(row['image_width'])
        height = int(row['image_height'])

        if not (min_size <= width and \
                min_size <= height and \
                width > 0 and height > 0):
            pruned_invalid_dim_count += 1
            continue

        # Use log aspect ratio for comparison
        log_img_ar = math.log(width / height) # height > 0 checked by min_size

        # Find the bucket with the minimum log aspect ratio difference
        diffs = np.abs(log_ar_array - log_img_ar)
        best_bucket_original_idx = int(diffs.argmin())
        min_diff = diffs[best_bucket_original_idx]

        if min_diff <= max_log_ar_diff:
            target_bucket_res = bucket_res_tuples_ordered[best_bucket_original_idx]
            bucket_counts[target_bucket_res] += 1
            assigned_count += 1
        else:
            pruned_ar_diff_count += 1

    total_pruned = pruned_invalid_dim_count + pruned_ar_diff_count
    print(f"\n--- Bucketing Analysis Summary ---")
    print(f"Total images processed: {len(df)}")
    print(f"Assigned to buckets: {assigned_count}")
    print(f"Pruned (invalid dimensions/too small): {pruned_invalid_dim_count}")
    print(f"Pruned (aspect ratio too different): {pruned_ar_diff_count}")
    print(f"Total pruned: {total_pruned}")

    print("\n--- Aspect Ratio Bucket Distribution ---")
    # Sort buckets by resolution for consistent printing
    sorted_bucket_items = sorted(bucket_counts.items(), key=lambda item: item[0])

    for i, (res, count) in enumerate(sorted_bucket_items):
        # Find the log_ar for this res; requires matching res back to
        # bucket_resolutions_list
        try:
            idx_in_original_list = bucket_resolutions_list.index(res)
            log_ar = log_aspect_ratios_list[idx_in_original_list]
            print(f"Bucket {res} (Log AR: {log_ar:.3f}): {count} samples")
        except ValueError:
            # Should not happen if logic is correct
            print(f"Bucket {res} (Log AR: N/A): {count} samples")

    print("--------------------------------------")

    return bucket_counts

def analyze_percentage_counter(counter: Counter, total_samples:int, name:str, min_samples:int=200,):
    """Shows the percentage of samples with respect to a given threshold and the representation of
    the total dataset.
    """
    print("*"*20)
    less_min= sum(1 for count in counter.values() if count < min_samples)
    less_min_counts = sum(count for count in counter.values() if count < min_samples)
    print(f"Num of unique {name} {len(counter)}")
    print(f"\n{name} with less than {min_samples} images: "
        f"{less_min}\n"
        f"Percentage of {name} with less than {min_samples} images {(less_min/len(counter)):.2%}\n"
        f"The {name} with more than {min_samples} images that are the {(1-(less_min/len(counter))):.2%} "
        f"of the data and they have {(total_samples-less_min_counts)/total_samples:.2%} of the total samples.\n"
    )

def gender_and_tag_cluster_analysis(
    df: pd.DataFrame,
    general_tags_counter: Counter,
    total_samples: int
):
    """
    Analyzes gender distribution, character counts, and performs simple
    keyword-based clustering on general tags.

    Args:
        df (pd.DataFrame): The full dataframe, needed for co-occurrence.
        general_tags_counter (Counter): A Counter object with general tags
                                        and their frequencies.
        total_samples (int): The total number of samples in the dataset.
    """
    print("\n--- 6. Character Count and Gender Analysis ---")

    # This section remains useful for understanding multi-character images.
    character_count_tags = [
        '1girl', '2girls', '3girls', '4girls', '5girls', '6+girls',
        '1boy', '2boys', '3boys', '4boys', '5boys', '6+boys',
        'multiple_girls', 'multiple_boys', 'group', 'solo'
    ]


    print("\n--- Character Count Distribution ---")
    count_dist = {
        tag: general_tags_counter.get(tag, 0)
        for tag in character_count_tags
        if general_tags_counter.get(tag, 0) > 0
    }
    if count_dist:
        df_counts = pd.DataFrame.from_dict(
            count_dist, orient='index', columns=['count']
        )
        df_counts['percentage'] = (df_counts['count']/total_samples * 100)
        print(df_counts.sort_values(by='count', ascending=False))
    else:
        print("No character count tags found.")

    # --- Gender Representation Analysis ---
    print("\n--- Gender Representation Analysis ---")
    count_1girl = general_tags_counter.get('1girl', 0)
    count_1boy = general_tags_counter.get('1boy', 0)
    count_solo = general_tags_counter.get('solo', 0)

    print(f"Total images with '1girl' tag: {count_1girl} "
        f"({count_1girl / total_samples:.2%})")
    print(f"Total images with '1boy' tag: {count_1boy} "
        f"({count_1boy / total_samples:.2%})")

    if count_solo > 0 and 'tag_string_general' in df.columns:
        # Use regex with word boundaries (\b) to ensure exact tag matching
        is_solo = df['tag_string_general'].str.contains(
            r'\bsolo\b', regex=True, na=False
        )
        is_1girl = df['tag_string_general'].str.contains(
            r'\b1girl\b', regex=True, na=False
        )
        is_1boy = df['tag_string_general'].str.contains(
            r'\b1boy\b', regex=True, na=False
        )

        # Count images with both 'solo' and the respective gender tag
        solo_girl_count = (is_solo & is_1girl).sum()
        solo_boy_count = (is_solo & is_1boy).sum()

        print("\n--- Gender Distribution within 'solo' Tagged Images ---")
        print(f"Total images with 'solo' tag: {count_solo}")
        print(f"  - Images with '1girl' and 'solo': {solo_girl_count} "
            f"({solo_girl_count / count_solo:.2%})")
        print(f"  - Images with '1boy' and 'solo': {solo_boy_count} "
            f"({solo_boy_count / count_solo:.2%})")
    else:
        print("\nCould not perform solo gender analysis: 'solo' tag not "
            "found or 'tag_string_general' column is missing.")

    # Section 7: Simple Tag Clustering
    print("\n--- 7. Simple Tag Clustering ---")

    # Define categories and associated keywords
    tag_clusters = {
        'Body Features': [
            'breasts', 'thighs', 'hair', 'eyes', 'skin', 'ass', 'navel',
            'legs', 'feet', 'face'
        ],
        'Clothing': [
            'skirt', 'dress', 'uniform', 'swimsuit', 'bra', 'panties',
            'sailor_suit', 'school_uniform', 'bikini', 'socks', 'shirt'
        ],
        'Accessories': [
            'ribbon', 'hat', 'jewelry', 'glasses', 'hair_ornament', 'halo',
            'wings', 'gloves', 'necklace'
        ],
        'Actions/Poses': [
            'lying', 'sitting', 'standing', 'looking_at_viewer', 'on_back',
            'spread_legs', 'from_behind', 'posing', 'holding'
        ],
        'Environment': [
            'outdoors', 'indoors', 'sky', 'water', 'night', 'bed', 'room',
            'city', 'background', 'simple_background'
        ],
        'Explicit/Rating': ['nsfw', 'nipples', 'pussy', 'nude', 'sex']
    }

    cluster_counts = Counter()
    # Create a reverse mapping from keyword to cluster for efficiency
    keyword_to_cluster = {
        keyword: cluster for cluster, keywords in tag_clusters.items()
        for keyword in keywords
    }

    # Iterate through all tags and categorize them
    for tag, count in general_tags_counter.items():
        for keyword, cluster in keyword_to_cluster.items():
            # Check if a keyword is part of the tag (e.g., 'long_hair')
            if keyword in tag:
                cluster_counts[cluster] += count
                break  # Assign tag to the first matching cluster

    if cluster_counts:
        print("Distribution of tags across simple clusters:")
        total_tags_cluster = sum(count for count in cluster_counts.values())
        # plot_bar_chart(
        #     cluster_counts,
        #     'Distribution of General Tag Clusters',
        #     'Cluster'
        # )
        # Print for console view
        for cluster, count in cluster_counts.most_common():
            print(f"- {cluster}: {count} tags"
                   f"({count / total_tags_cluster:.2%})")
    else:
        print("Could not form any tag clusters from the general tags.")

def counts_comparison(
    prior_counts: Union[Counter, pd.DataFrame],
    filtered_counts_df: pd.DataFrame,
    category_name: str,
    debug_tags_path: str="debug_tags.csv"
):
    """Compares tag counts from a prior dataset with counts from a
    filtered dataset, providing detailed statistics on the differences.

    This function is optimized to use pandas DataFrames for efficient
    comparison. It takes a Counter for the prior counts and a DataFrame
    for the filtered counts, avoiding redundant data conversions.

    Args:
        prior_counts (Counter): A Counter object with tags as keys and
                                their counts as values from the original
                                dataset.
        filtered_counts_df (pd.DataFrame): A DataFrame containing the
                                           filtered counts, with 'tag'
                                           and 'count' columns.
        category_name (str): The category to analyze (e.g., 'character',
                             'artist').
    """
    print(
        f"\n--- Comparing Tag Counts for Category: {category_name.title()} ---"
    )

    # 1. Prepare DataFrames for efficient merging.
    # if not Counter then already a df
    if type(prior_counts) == Counter:
        prior_df = pd.DataFrame(
            prior_counts.items(), columns=['tag', 'prior_count']
        )
    else:
        prior_df = prior_counts.copy()
        if 'image_count' in prior_counts:
            prior_df = prior_df.rename(
                columns={'image_count': 'prior_count'}
            )
        if 'artist' in prior_counts:
            prior_df = prior_df.rename(
                columns={'artist': 'tag'}
            )
    # Use the filtered DataFrame directly, just rename the count column.
    filtered_df = filtered_counts_df.rename(
        columns={'count': 'filtered_count'}
    )

    if prior_df.empty:
        print("Prior counts are empty. Cannot perform comparison.")
        return

    # 2. Merge and calculate differences.
    # An outer merge is crucial to capture tags that were either completely
    # removed or newly added during the filtering process.
    comparison_df = pd.merge(
        prior_df,
        filtered_df,
        on='tag',
        how='outer'
    )

    # Fill NaN with 0 for tags not present in one of the datasets.
    comparison_df.fillna(0, inplace=True)

    # Ensure counts are integer types for clean calculations.
    for col in ['prior_count', 'filtered_count']:
        comparison_df[col] = comparison_df[col].astype(int)

    # Calculate the change in sample count for each tag.
    comparison_df['difference'] = (
        comparison_df['prior_count'] - comparison_df['filtered_count']
    )

    # A negative value indicates a percentage decrease after filtering.
    comparison_df['difference_percentage'] = 100 * (
        (comparison_df['filtered_count'] - comparison_df['prior_count']) /
         comparison_df['prior_count']
    )
    # Handle division by zero for new tags (where prior_count is 0).
    # Replace resulting 'inf' with NaN, then fill NaN with 0.0.
    comparison_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    comparison_df.fillna({'difference_percentage': 0.0}, inplace=True)

    # 3. Report statistics to summarize the filtering's impact.
    total_prior_tags = len(prior_df)
    total_filtered_tags = len(filtered_df)
    tags_removed = comparison_df[
        (comparison_df['prior_count'] > 0) &
        (comparison_df['filtered_count'] == 0)
    ].shape[0]

    print(f"Total unique tags (prior): {total_prior_tags}")
    print(f"Total unique tags (filtered): {total_filtered_tags}")
    print(f"Tags from prior set completely removed: {tags_removed}")

    # Sort by 'difference' to highlight tags most affected by filtering.
    top_dif_subset = comparison_df.sort_values(
            by='difference_percentage', ascending=True
        ).head(25)
    mode = 'a' if os.path.exists(debug_tags_path) else 'w'
    header = True if mode == 'w' else False
    top_dif_subset.to_csv(
        debug_tags_path,
        mode=mode, header=header,
        index=False,
        encoding='utf-8'
    )

    print("\nTop 25 tags with the largest decrease in samples:")
    print(top_dif_subset.to_string())

    print("\nTop 10 tags with the largest increase (or smallest decrease):")
    print(
        comparison_df.sort_values(
            by='difference_percentage', ascending=False
        ).head(10).to_string()
    )
    print(f"Relevant statistics:\n {comparison_df['difference_percentage'].describe()}")

def tag_analysis(df, prior_tags=None, filtered_counts_df=None, debug_tags_csv:str=None):
    # Section 4: Tag Count Analysis
    # The number of tags per image. More tags can provide richer context for
    # the text encoder, but too few might lead to poor conditioning.
    # Your plan to upsample tags is good; this shows the baseline.
    print("\n--- 4. Tag Count Analysis ---")
    tag_count_cols = {
        'tag_count': 'Total Tags',
        'tag_count_general': 'General Tags',
        'tag_count_character': 'Character Tags',
        'tag_count_copyright': 'Copyright Tags',
        'tag_count_artist': 'Artist Tags',
        # 'tag_count_meta': 'Meta Tags'
    }
    for col, name in tag_count_cols.items():
        if col not in df.columns:
            class_name = col.split('_')
            if len(class_name) == 2:
                df['tag_count'] = df[
                    'tag_string'
                ].str.split(' ').str.len()
            else:
                df[f'tag_count_{class_name[-1]}'] = df[
                    f'tag_string_{class_name[-1]}'
                ].str.split(' ').str.len()
        # plot_histogram(df, col, f'Distribution of {name} per Image',
        #             f'Number of {name}')
        print(f"\n{name} count statistics:\n{df[col].describe()}")


    # Section 5: Common Tags Analysis
    # Identifying the most frequent tags helps understand the dataset's content
    # focus and potential biases. This is crucial for ensuring diversity.
    print("\n--- 5. Common Tags Analysis ---")
    tag_string_cols = {
        'tag_string_general': 'General Tags',
        'tag_string_character': 'Character Tags',
        'tag_string_copyright': 'Copyright/Source Tags',
        'tag_string_artist': 'Artist Tags',
        'tag_string_meta': 'Meta Tags'
    }

    # Store all general tags for later use if needed (e.g. word cloud)
    character_tags_counter = Counter()
    general_tags_counter = Counter()
    for col, name in tag_string_cols.items():
        if col in df.columns:
            print(f"\n--- Top 25 Most Common {name} ---")
            # Ensure the column is treated as string, handle potential NaNs
            tag_series = df[col].astype(str).fillna('')
            tags_counter = parse_tags(tag_series)
            if tags_counter:
                plot_bar_chart(tags_counter, f'Top 25 Most Common {name}', name,
                            top_n=25)
                # Print top 5 for brevity in console
                print(tags_counter.most_common(5))
                if col == 'tag_string_character':
                    character_tags_counter = tags_counter
                elif col == 'tag_string_general':
                    general_tags_counter = tags_counter

            else:
                print(f"No tags found or parsed for {name}.")
        else:
            print(f"Column '{col}' not found for common tag analysis.")
    total_samples = len(df)
    # incorrect as it uses images in the df not the number of tags
    analyze_percentage_counter(general_tags_counter, total_samples,name="General", min_samples=500)
    analyze_percentage_counter(character_tags_counter, total_samples,name="Character", min_samples=200)

    if prior_tags:
        character_tags_counter_prior = Counter({key: character_tags_counter[key] for key in prior_tags if key in character_tags_counter})
        print(f"5 most popular characters: {character_tags_counter_prior.most_common(5)}")
        print(f"20 characters with lowest samples: {character_tags_counter_prior.most_common()[-20:]}")
        if filtered_counts_df is not None:
            # Select only the 'character' rows from the filtered counts df.
            character_filtered_df = filtered_counts_df[
                filtered_counts_df['category'] == 'character'
            ]
            # filter to use only prior characters
            character_filtered_df = character_filtered_df[character_filtered_df["tag"].isin(prior_tags)]

            if not character_filtered_df.empty:
                counts_comparison(
                    character_tags_counter_prior,
                    character_filtered_df,
                    'character',
                    debug_tags_path=debug_tags_csv
                )

    if general_tags_counter:
        gender_and_tag_cluster_analysis(
            df, general_tags_counter, total_samples
        )

def artist_types_analysis(df, prior_tags=None, filtered_counts_df=None, debug_tags_csv:str=None):
    # Section 6: Artist Diversity (from tag_string_artist)
    # Crucial for learning diverse styles. Over-representation of a few artists
    # can lead to style collapse.
    # This was partially covered in "Common Artist Tags", but a dedicated look.
    print("\n--- 6. Artist Analysis ---")
    if 'tag_string_artist' in df.columns:
        # Assuming one artist tag per image, or the primary one is listed.
        # If multiple artists can be tagged, parse_tags is appropriate.
        # For Danbooru, 'tag_string_artist' usually contains one or few artists.
        # Let's use parse_tags to handle multiple artist tags per image if present.
        artist_series = df['tag_string_artist'].astype(str).fillna('')
        artist_counter = parse_tags(artist_series)

        if artist_counter:
            print(f"Total unique artist tags: {len(artist_counter)}")
            print("*"*20)

            # Distribution of images per artist
            artist_counts_df = pd.DataFrame(artist_counter.most_common(),
                                            columns=['artist', 'image_count'])
            artist_counts_df.sort_values(by='image_count', ascending=False)
            if prior_tags:
                artist_counts_df_prior = artist_counts_df[artist_counts_df["artist"].isin(prior_tags)]
                print(f"5 most popular artist: {artist_counts_df_prior.head(5)}")
                print(f"20 artist with lowest samples: {artist_counts_df_prior.tail(20)}")
                if filtered_counts_df is not None:
                    # Select only the 'artist' rows from the filtered counts df.
                    artist_filtered_df = filtered_counts_df[
                        filtered_counts_df['category'] == 'artist'
                    ]
                    artist_filtered_df = artist_filtered_df[artist_filtered_df["tag"].isin(prior_tags)]
                    if not artist_filtered_df.empty:
                        counts_comparison(
                            artist_counts_df_prior,
                            artist_filtered_df,
                            'artist',
                            debug_tags_path=debug_tags_csv
                        )

            analyze_percentage_counter(artist_counter, len(df),name="Artists", min_samples=50)
        else:
            print("No artist tags found or parsed.")
    else:
        print("Column 'tag_string_artist' not found for artist analysis.")


    # Section 7: File Type Analysis (file_ext)
    # Good to know the distribution of image formats, though VAE processes pixels.
    print("\n--- 7. File Type Analysis ---")
    if 'file_ext' in df.columns:
        print(f"File extension distribution:\n"
            f"{df['file_ext'].value_counts(normalize=True)}")
    else:
        print("Column 'file_ext' not found.")

    # Section 8: Temporal Analysis (created_at)
    # Distribution of images over time. Can reveal if the dataset is skewed
    # towards recent trends or has consistent data over years.
    print("\n--- 8. Temporal Analysis ---")
    if 'created_at' in df.columns:
        # Ensure 'created_at' is in datetime format
        # The format from tail() looks like '2024-08-31T23:59:47.107+09:00'
        # Pandas can often infer this, but explicit parsing is safer.
        try:
            # Attempt to convert, handling potential errors for mixed formats
            df['created_at_dt'] = pd.to_datetime(df['created_at'],
                                                errors='coerce', utc=True)

            # Drop rows where conversion failed
            df_temporal = df.dropna(subset=['created_at_dt'])

            if not df_temporal.empty:
                df_temporal['year_month'] = df_temporal['created_at_dt'].dt.to_period('M')

                monthly_counts = df_temporal['year_month'].value_counts().sort_index()

                if not monthly_counts.empty:
                    plt.figure(figsize=(15, 7))
                    monthly_counts.plot(kind='line')
                    plt.title('Number of Images Uploaded Over Time (Monthly)')
                    plt.xlabel('Year-Month')
                    plt.ylabel('Number of Images')
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    plt.show()
                else:
                    print("No valid monthly data to plot for temporal analysis.")

                # Distribution by year
                df_temporal['year'] = df_temporal['created_at_dt'].dt.year
                yearly_counts = df_temporal['year'].value_counts().sort_index()
                if not yearly_counts.empty:
                    plt.figure(figsize=(10, 6))
                    yearly_counts.plot(kind='bar')
                    plt.title('Number of Images Uploaded by Year')
                    plt.xlabel('Year')
                    plt.ylabel('Number of Images')
                    plt.xticks(rotation=45)
                    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
                    plt.tight_layout()
                    plt.show()
                else:
                    print("No valid yearly data to plot for temporal analysis.")
            else:
                print("No valid datetime entries found in 'created_at' after conversion.")

        except Exception as e:
            print(f"Error processing 'created_at' column: {e}")
    else:
        print("Column 'created_at' not found for temporal analysis.")


    print("\n### End of EDA ###")

def analyze_prompt_lengths(
    df: pd.DataFrame,
    tokenizer_path: str = "model/tokenizer"
) -> list[int]:
    """
    Computes the token length for each generated prompt file.

    This function reads the final text prompts, tokenizes them using the
    CLIP tokenizer, and returns a list of their lengths. This is crucial
    for understanding the prompt distribution and potential truncation.

    Args:
        df (pd.DataFrame): DataFrame containing the 'relative_path' to
                           locate the prompt files.
        tokenizer_path (str): Path to the tokenizer.

    Returns:
        list[int]: A list of token lengths for each prompt.
    """
    print("Analyzing prompt token lengths...")
    try:
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
    except OSError:
        print(f"Error: Tokenizer not found at '{tokenizer_path}'.")
        print("Cannot analyze prompt lengths. Please check the path.")
        return []

    lengths = []

    # 1. Vectorized String Formatting:
    # This replaces the slow .apply() call. We chain .str accessors to
    # perform all replacements across the entire Series at once.
    # regex=False is a small optimization for literal string replacements.
    # .fillna('') handles any missing tags gracefully.
    prompts = (
        df['tag_string'].str.replace('(', r'\(', regex=False)
                        .str.replace(')', r'\)', regex=False)
                        .str.replace(' ', ', ', regex=False)
                        .str.replace('_', ' ', regex=False)
                        .fillna('')
                        .tolist()
    )

    # 2. Batch Tokenization:
    # The tokenizer is called only once on the entire list of prompts.
    # This is the most significant optimization, as it uses the fast,
    # Rust-based backend of the Hugging Face tokenizer library.
    print("Tokenizing all prompts in a single batch...")
    inputs = tokenizer(
        prompts,
        padding=False,
        truncation=False,
        return_length=True,
    )

    # 3. Efficient Length Calculation:
    # A list comprehension is a fast, idiomatic way to process the results.
    # The 'length' from the tokenizer includes BOS/EOS tokens, so we
    # subtract 2 to get the count of actual content tokens.
    lengths = [length - 2 for length in inputs['length']]

    return lengths

def plot_length_distribution(
    lengths: list[int],
    output_path: str = "prompt_length_distribution.png"
):
    """
    Plots the distribution of prompt lengths and prints statistics.

    Args:
        lengths (list[int]): A list of prompt token lengths.
        output_path (str): The file path to save the plot.
    """
    if not lengths:
        print("Cannot plot distribution: length list is empty.")
        return

    plt.figure(figsize=(12, 6))
    plt.hist(lengths, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Prompt Token Lengths')
    plt.xlabel('Token Length (excluding BOS/EOS)')
    plt.ylabel('Number of Prompts')

    # Add a vertical line at 75 tokens, a common chunk size for CLIP
    plt.axvline(
        x=225, color='r', linestyle='--',
        label='CLIP Max Length (225 tokens)'
    )
    plt.legend()
    plt.grid(axis='y', alpha=0.75)

    # Save the plot
    # plt.savefig(output_path)
    plt.show()
    print(f"Saved prompt length distribution plot to '{output_path}'.")
    plt.close()

    quantiles = np.percentile(
        lengths, [10, 25, 75, 90]
    )
    # Print descriptive statistics
    print("\n--- Prompt Length Statistics ---")
    print(f"Total prompts analyzed: {len(lengths)}")
    print(f"Mean length: {np.mean(lengths):.2f}")
    print(f"Median length: {np.median(lengths)}")
    print(f"Quantiles 10% : {quantiles[0]:.2f}")
    print(f"25% : {quantiles[1]:.2f}")
    print(f"75% : {quantiles[2]:.2f}")
    print(f"90% : {quantiles[3]:.2f}")
    print(f"Standard Deviation: {np.std(lengths):.2f}")
    print(f"Min length: {np.min(lengths)}")
    print(f"Max length: {np.max(lengths)}")

    # Calculate percentage of prompts exceeding the standard CLIP length
    over_225 = sum(1 for length in lengths if length > 225)
    percent_over_225 = (over_225 / len(lengths)) * 100
    print(f"Prompts > 225 tokens: {over_225} ({percent_over_225:.2f}%)")
    print("--------------------------------\n")


def analyze_prior_knowledge_dataset(
    prior_df_path: str,
    sampled_ids_path: str=None,
    aes_scores_csv_path: str=None,
    knowledge_bases_paths: str=None,
    tokenizer_path: str=None,
    debug_tags_csv: str=None,
):
    prior_knowledge_samples = load_prior_knowledge_df(prior_df_path, aes_scores_csv_path)
    print(f"Loaded prior knowledge df of {len(prior_knowledge_samples)} samples.")
    prior_tags = get_tags_from_knowledge_bases(knowledge_bases_paths)
    print(f"Loaded {len(prior_tags)} tags from prior knowledge")
    tag_counts_path = "tag_counts_sampled_ids.csv"
    try:
        tag_counts_df = pd.read_csv(tag_counts_path, header=0)
    except FileNotFoundError:
        print(f"Skipping tag counts as not found at {tag_counts_path}")

    if sampled_ids_path:
        try:
            sampled_df = pd.read_csv(sampled_ids_path, header=0, low_memory=False)
        except FileNotFoundError:
            print("Error: Sampled IDs file not found. Run sampling first.")
            return
        # merge to get the quality_tier
        prior_knowledge_samples = pd.merge(prior_knowledge_samples, sampled_df[['id', 'quality_tier']], how="left", on="id")
        # samples without quality_tier means not sampled
        prior_knowledge_samples.dropna(subset=['quality_tier'], inplace=True)
        # prior_knowledge_samples = prior_knowledge_samples[prior_knowledge_samples['id'].isin(include_ids)]
        # free memory
        del sampled_df
        print(f"Filtered prior knowledge to {len(prior_knowledge_samples)} samples.")

    score_analysis(prior_knowledge_samples)
    rating_imgs_analysis(prior_knowledge_samples)
    analyze_dataframe_bucketing(
        prior_knowledge_samples,
        max_res_area=(768, 512), # Max area for a 512x512 equivalent in tokens
        min_size=256,           # Smallest dimension allowed for an image/bucket
        divisible=64,           # Bucket dimensions must be divisible by this
        base_res=(512, 512),    # Ensure 512x512 is a possible bucket
        vae_factor=8,           # VAE downsampling factor
        dim_limit=1024,         # Max dimension for any side of an image/bucket
        max_log_ar_diff=0.3    # Max difference in log AR for assignment
    )

    tag_analysis(
        df=prior_knowledge_samples,
        prior_tags=prior_tags,
        filtered_counts_df=tag_counts_df,
        debug_tags_csv=debug_tags_csv,
    )
    artist_types_analysis(
        df=prior_knowledge_samples,
        prior_tags=prior_tags,
        filtered_counts_df=tag_counts_df,
        debug_tags_csv=debug_tags_csv,
    )
    if tokenizer_path is not None:
        lengths = analyze_prompt_lengths(prior_knowledge_samples, tokenizer_path)
        plot_length_distribution(lengths)