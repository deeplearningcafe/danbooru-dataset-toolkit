from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from .prompt_utils import format_danbooru_tag


class TagWeighter:
    def __init__(
        self,
        tag_counts_csv_path: str,
        artists_txt_path: Optional[str] = None,
        characters_txt_path: Optional[str] = None,
        default_weight: float = 1.0,
        min_weight: float = 0.25,
        max_weight: float = 2.0,
        smoothing_factor: float = 0.05,
        prior_multiplier_artists: float = 1.35,
        prior_multiplier_chars: float = 1.15,
    ):
        """
        Initialize the tag weighting system.

        Args:
            tag_counts_csv_path: Path to the CSV with tag counts.
            artists_txt_path: Optional path to a TXT file with one
                              prioritized artist per line.
            characters_txt_path: Optional path to a TXT file with one
                                 prioritized character per line.
            default_weight: Default weight for untracked tags.
            min_weight: The minimum possible weight for any tag.
            max_weight: The maximum possible weight for any tag.
            smoothing_factor: Factor to prevent division by zero for
                              rare tags.
            prior_multiplier: Multiplier for prioritized artists and
                              characters.
        """

        # Core weighting parameters from the configuration
        self.default_weight = default_weight
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.smoothing_factor = smoothing_factor
        self.prior_multiplier_artists = prior_multiplier_artists
        self.prior_multiplier_chars = prior_multiplier_chars

        # Define the categories to be used for weighting.
        # We ignore metadata and other categories as per the requirements.
        self.categories = ["general", "character", "artist"]

        # Dictionaries to store tag data
        self.tag_counts: Dict[str, Dict[str, int]] = {
            cat: {} for cat in self.categories
        }
        self.tag_weights: Dict[str, Dict[str, float]] = {
            cat: {} for cat in self.categories
        }
        # A mapping for quick lookup of a tag's category
        self.tag_to_category: Dict[str, str] = {}

        # Sets for fast lookup of prioritized tags
        self.prior_artists = set()
        self.prior_characters = set()


        # Load, process, and compute weights from the provided CSV
        try:
            self._load_and_process_tags(tag_counts_csv_path)
            self._compute_all_weights()
            # Load prioritized artists and characters
            self._load_priors(artists_txt_path, characters_txt_path)
            print("TagWeighter initialized successfully.")
        except FileNotFoundError:
            print(f"Tag counts CSV not found at: {tag_counts_csv_path}")
            raise
        except Exception as e:
            print(f"Failed to initialize TagWeighter: {e}")
            raise

    def _load_priors(
        self,
        artists_path: Optional[str],
        characters_path: Optional[str]
    ):
        """Load prioritized artists and characters from text files."""
        if artists_path:
            print(f"Loading prioritized artists from {artists_path}...")
            with open(artists_path, 'r', encoding='utf-8') as f:
                self.prior_artists = {
                    format_danbooru_tag(line.strip())
                    for line in f if line.strip()
                }
            print(f"Loaded {len(self.prior_artists)} prioritized artists.")

        if characters_path:
            print(f"Loading prioritized characters from {characters_path}...")
            if characters_path.endswith(".txt"):
                with open(characters_path, 'r', encoding='utf-8') as f:
                    self.prior_characters = {
                        format_danbooru_tag(line.strip())
                        for line in f if line.strip()
                    }
            # Process .xlsx files for characters
            elif characters_path.endswith(".xlsx"):
                char_df = pd.read_excel(characters_path)
                if 'character_tag' in char_df.columns:
                    for char_tag in char_df['character_tag'].dropna():
                        # Add only if tag is valid and not already present
                        if char_tag:
                            # already in booru format.
                            self.prior_characters.add(char_tag)

            print(f"Loaded {len(self.prior_characters)} prioritized characters.")

    def _load_and_process_tags(self, csv_path: str):
        """Load tag counts from CSV and populate internal data structures."""
        print(f"Loading tag counts from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Ensure required columns exist
        if not {'tag', 'category', 'count'}.issubset(df.columns):
            raise ValueError(
                "CSV must contain 'tag', 'category', and 'count' columns."
            )

        # Filter for the categories we care about
        df = df[df['category'].isin(self.categories)]

        for _, row in df.iterrows():
            tag, category, count = row['tag'], row['category'], int(row['count'])
            if category in self.categories:
                # Store the count for the tag within its category
                self.tag_counts[category][tag] = count
                # Map the tag to its category for fast lookups later
                self.tag_to_category[tag] = category

    def _compute_all_weights(self):
        """
        Compute weights for all tags based on their frequency within their
        category using vectorized operations for efficiency.
        """
        print("Computing weights for all tags...")
        min_max_diff = self.max_weight - self.min_weight
        
        for category in self.categories:
            category_counts = self.tag_counts[category]
            if not category_counts:
                print(f"No tags found for category: {category}")
                continue

            tags = list(category_counts.keys())
            counts = np.array(list(category_counts.values()), dtype=np.float32)

            total_count = np.sum(counts)
            if total_count == 0:
                continue

            # Calculate frequency of each tag in the category
            frequencies = counts / total_count

            # Vectorized weight calculation based on inverse frequency
            # This formula up-weights rare tags (low frequency) and
            # down-weights common tags (high frequency).
            weights = 1.0 / (frequencies + self.smoothing_factor)

            # Normalize weights to fit within the specified min/max range
            # We scale the weights to ensure they fall within a predictable
            # range, providing stability during training.
            min_w, max_w = np.min(weights), np.max(weights)
            if max_w > min_w:
                normalized_weights = self.min_weight + (
                    (weights - min_w) / (max_w - min_w)
                ) * min_max_diff
            else:
                # If all weights are the same, assign the default
                normalized_weights = np.full_like(
                    weights, self.default_weight
                )
            
            # Clip weights to enforce the absolute min/max boundaries
            final_weights = np.clip(
                normalized_weights, self.min_weight, self.max_weight
            )

            # Store the computed weights
            self.tag_weights[category] = dict(zip(tags, final_weights.tolist()))

    def get_caption_weight(self, caption: str) -> float:
        """
        Calculate the final weight for a given caption.

        The process is as follows:
        1. Parse tags from the comma-separated caption string.
        2. For each tag, look up its pre-computed weight.
        3. If a tag is a prioritized artist or character, its weight is
           multiplied by `prior_multiplier` and clamped.
        4. Group weights by category ('general', 'character', 'artist').
        5. Calculate the arithmetic mean of weights for each category.
        6. Compute the exponential average (geometric mean) of the
           category means to get the final caption weight.

        Args:
            caption: A string of comma-separated tags.

        Returns:
            A single float representing the caption's training weight.
        """
        if not isinstance(caption, str) or not caption:
            return self.default_weight

        # Use defaultdict to simplify appending weights
        category_wise_weights: Dict[str, List[float]] = defaultdict(list)

        # Normalize tags: comma-separated and spaces inside tags
        tags = [tag.strip() for tag in caption.split(',') if tag.strip()]

        for tag in tags:
            # Fast lookup for tag's category
            category = self.tag_to_category.get(tag)
            if category:
                # Look up the pre-computed weight
                weight = self.tag_weights[category].get(tag, self.default_weight)

                # Apply prior multiplier for specified artists/characters
                if category == 'artist' and tag in self.prior_artists:                
                    weight *= self.prior_multiplier_artists
                # use if instead of elif to give higher weight artists and chars
                if category == 'character' and tag in self.prior_characters:
                    weight *= self.prior_multiplier_chars
                # Clamp the weight to not exceed the max_weight
                weight = min(weight, self.max_weight)

                category_wise_weights[category].append(weight)

        # Calculate the arithmetic mean for each category
        category_means = []
        for category in self.categories:
            if category_wise_weights[category]:
                mean_val = np.mean(category_wise_weights[category])
                category_means.append(mean_val)

        if not category_means:
            return self.default_weight

        # Compute the final weight using the exponential average (geometric mean)
        # This ensures a balanced contribution from all present categories.
        final_weight = np.exp(np.mean(np.log(category_means)))

        return float(final_weight)