"""Module for encoding raw image/text data into Parquet streaming shards."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import random
from ..prompts.prompt_utils import sanitize_prompt
from ..utils.loader import AESTHETIC_LABEL


class ImageStreamEncoder:
    """Encodes raw image files and text prompts into Parquet dataset shards."""

    def __init__(
        self,
        dataset,
        tokenizer,
        output_dir: str,
        samples_per_shard: int = 10000,
        length_tiers: List[int] = [77, 152, 227],
        parquet_compression: str = "SNAPPY",
        min_sample_count: int = 8,
        seed: int = 42,
        aesthetic_csv_path: Optional[str] = None,
    ):
        """Initialize the stream encoder.

        Args:
            dataset: LatentEncodingDataset instance.
            tokenizer: CLIP tokenizer instance.
            output_dir: Path to write Parquet shards.
            samples_per_shard: Maximum samples per Parquet file.
            length_tiers: Sequence length tier thresholds.
            parquet_compression: Parquet compression codec.
            min_sample_count: Minimum samples required per bucket/tier.
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.samples_per_shard = samples_per_shard
        self.length_tiers = sorted(length_tiers)
        self.max_length = self.length_tiers[-1]
        self.parquet_compression = parquet_compression
        self.min_sample_count = min_sample_count
        self.seed = seed

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.aesthetic_df = None
        if aesthetic_csv_path:
            try:
                df = pd.read_csv(aesthetic_csv_path, dtype={"id": str})
                df = df.drop_duplicates(subset=["id"], keep="last")
                self.aesthetic_df = df.set_index("id")

                def get_tier_num(tier_val):
                    if pd.isna(tier_val):
                        return -1
                    return AESTHETIC_LABEL.get(tier_val, -1)

                self.aesthetic_df["final_tier_num"] = self.aesthetic_df[
                    "final_tier"
                ].apply(get_tier_num)
            except Exception as e:
                print(f"Warning: Failed to load aesthetic CSV: {e}")

        self.schema = pa.schema(
            [
                ("booru_id", pa.string()),
                ("image", pa.binary()),
                ("prompt", pa.string()),
                ("bucket_idx", pa.int32()),
                ("target_width", pa.int32()),
                ("target_height", pa.int32()),
                ("original_width", pa.int32()),
                ("original_height", pa.int32()),
                ("aspect_ratio", pa.float32()),
                ("tier", pa.int32()),
                ("aesthetic_tier", pa.int32()),
                ("tag_weight", pa.float32()),
            ]
        )

    def _determine_tier(self, length: int) -> int:
        """Finds the appropriate token tier for a prompt length."""
        for tier in self.length_tiers:
            if length <= tier:
                return tier
        return self.max_length

    def _get_prompt_str_and_len(self, prompt_obj) -> Tuple[str, int]:
        """Extracts text string and token count from prompt object."""
        if isinstance(prompt_obj, dict):
            parts = []
            length = 0
            tokens_keys = ["prefix_tokens", "general_tokens", "suffix_tokens"]
            for k in tokens_keys:
                if k in prompt_obj and self.tokenizer:
                    toks = prompt_obj[k]
                    length += len(toks)
                    text = self.tokenizer.decode(toks, skip_special_tokens=True)
                    if text.strip():
                        parts.append(text.strip())
            prompt_str = ", ".join(parts)
            length = sum(
                len(prompt_obj[k])
                for k in tokens_keys
                if k in prompt_obj and isinstance(prompt_obj[k], list)
            )
        else:
            prompt_str = str(prompt_obj)
            if self.tokenizer:
                toks = self.tokenizer(
                    prompt_str,
                    padding=False,
                    truncation=True,
                    max_length=self.max_length,
                    return_length=True,
                )
                length = toks["length"][0]
            else:
                length = len(prompt_str.split())

        return prompt_str, length

    def _read_prompt_and_weight(self, img_path: Path) -> Tuple[str, int, float]:
        """Reads raw text prompt and tag weight from .txt and .json files."""
        img_path = Path(img_path)
        txt_path = img_path.with_suffix(".txt")

        prompt_str = ""
        prompt_obj = self.dataset._load_prompt(img_path, self.dataset.label_ext)
        if txt_path.exists():
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    prompt_str = f.read().strip()
                    prompt_str = sanitize_prompt(prompt_str)
            except UnicodeDecodeError:
                with open(txt_path, "r", encoding="utf-16") as f:
                    prompt_str = f.read().strip()
        else:
            print(f"Prompt for {img_path} with path {txt_path} doesn't exist")
            prompt_str, _ = self._get_prompt_str_and_len(prompt_obj)

        tag_weight = 1.0
        if "tag_weight" in prompt_obj:
            tag_weight = float(prompt_obj.get("tag_weight", 1.0))

        length = sum(
            len(prompt_obj[k])
            for k in ["prefix_tokens", "general_tokens", "suffix_tokens"]
            if k in prompt_obj and isinstance(prompt_obj[k], list)
        )

        return prompt_str, length, tag_weight

    def encode_dataset(self) -> Dict:
        """Encodes samples into Parquet files with metadata index."""
        print("Pre-calculating prompt tiers and grouping samples...")
        grouped_samples = defaultdict(lambda: defaultdict(list))

        for idx, (bucket_idx, orig_idx) in enumerate(self.dataset.index_mapping):
            bucket_idx = int(bucket_idx)
            orig_idx = int(orig_idx)
            img_path = Path(self.dataset.paths[orig_idx])

            _, length, _ = self._read_prompt_and_weight(img_path)

            tier = self._determine_tier(length)
            grouped_samples[bucket_idx][tier].append(orig_idx)

        # Collect lightweight metadata descriptors without loading image bytes
        flat_descriptors = []
        for bucket_idx in sorted(grouped_samples.keys()):
            res = tuple(map(int, self.dataset.bucket_resolutions[bucket_idx]))
            target_w, target_h = res

            for tier in sorted(grouped_samples[bucket_idx].keys()):
                indices = grouped_samples[bucket_idx][tier]
                if len(indices) < self.min_sample_count:
                    continue

                for orig_idx in indices:
                    flat_descriptors.append(
                        (orig_idx, bucket_idx, tier, target_w, target_h)
                    )

        total_samples = len(flat_descriptors)
        print(
            f"Filtered {total_samples} valid samples. "
            "Shuffling globally across all shards..."
        )

        rng = random.Random(self.seed)
        rng.shuffle(flat_descriptors)

        print(f"Encoding {total_samples} samples into Parquet shards...")
        shard_counter = 0
        shard_metadata = []
        pbar = tqdm(total=total_samples, desc="Processing Parquet Shards")

        for start_idx in range(0, total_samples, self.samples_per_shard):
            chunk_descriptors = flat_descriptors[
                start_idx : start_idx + self.samples_per_shard
            ]
            sample_records = []

            for (
                orig_idx,
                bucket_idx,
                tier,
                target_w,
                target_h,
            ) in chunk_descriptors:
                img_path = Path(self.dataset.paths[orig_idx])
                booru_id = str(self.dataset.get_image_id(orig_idx))

                with open(img_path, "rb") as f:
                    img_bytes = f.read()

                prompt_str, _, tag_weight = self._read_prompt_and_weight(img_path)
                orig_w, orig_h = map(int, self.dataset.raw_res[orig_idx])
                ar = float(orig_w / orig_h) if orig_h > 0 else 1.0

                aes_tier = -1
                if self.aesthetic_df is not None:
                    try:
                        aes_tier = int(
                            self.aesthetic_df.loc[booru_id, "final_tier_num"]
                        )
                    except KeyError:
                        aes_tier = -1

                record = {
                    "booru_id": booru_id,
                    "image": img_bytes,
                    "prompt": prompt_str,
                    "bucket_idx": bucket_idx,
                    "target_width": target_w,
                    "target_height": target_h,
                    "original_width": orig_w,
                    "original_height": orig_h,
                    "aspect_ratio": ar,
                    "tier": tier,
                    "aesthetic_tier": aes_tier,
                    "tag_weight": tag_weight,
                }
                sample_records.append(record)
                pbar.update(1)

            shard_path = self._write_shard(sample_records, shard_counter)
            shard_metadata.append(
                {
                    "shard_file": str(shard_path.name),
                    "sample_count": len(sample_records),
                }
            )
            shard_counter += 1

        pbar.close()

        meta_dict = {
            "total_samples": total_samples,
            "num_shards": len(shard_metadata),
            "samples_per_shard": self.samples_per_shard,
            "length_tiers": self.length_tiers,
            "shards": shard_metadata,
            "bucket_info": [
                {
                    "bucket_idx": i,
                    "resolution": tuple(map(int, self.dataset.bucket_resolutions[i])),
                    "log_aspect_ratio": float(self.dataset.log_aspect_ratios[i]),
                }
                for i in range(len(self.dataset.bucket_resolutions))
            ],
        }

        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=4)

        print(f"Parquet encoding finished. Output saved to {self.output_dir}")
        return meta_dict

    def _write_shard(self, records: List[Dict], shard_idx: int) -> Path:
        """Writes a batch of records to a Parquet file."""
        fname = f"data_shard_{shard_idx:05d}.parquet"
        out_path = self.output_dir / fname

        pydict = {col: [r[col] for r in records] for col in self.schema.names}
        table = pa.Table.from_pydict(pydict, schema=self.schema)

        pq.write_table(
            table,
            out_path,
            compression=self.parquet_compression,
            use_dictionary=True,
        )
        return out_path
