"""Module for encoding raw image/text data into Parquet streaming shards."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


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

        self.output_dir.mkdir(parents=True, exist_ok=True)

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
            for k in ["prefix_tokens", "general_tokens", "suffix_tokens"]:
                if k in prompt_obj and self.tokenizer:
                    toks = prompt_obj[k]
                    text = self.tokenizer.decode(toks, skip_special_tokens=True)
                    if text.strip():
                        parts.append(text.strip())
            prompt_str = ", ".join(parts)
            length = sum(len(prompt_obj.get(k, [])) for k in prompt_obj)
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
                length = toks["length"]
            else:
                length = len(prompt_str.split())

        return prompt_str, length

    def encode_dataset(self) -> Dict:
        """Encodes samples into Parquet files with metadata index."""
        print("Pre-calculating prompt tiers and grouping samples...")
        grouped_samples = defaultdict(lambda: defaultdict(list))

        for idx, (bucket_idx, orig_idx) in enumerate(self.dataset.index_mapping):
            bucket_idx = int(bucket_idx)
            orig_idx = int(orig_idx)
            img_path = self.dataset.paths[orig_idx]

            prompt_obj = self.dataset._load_prompt(img_path, self.dataset.label_ext)
            prompt_str, length = self._get_prompt_str_and_len(prompt_obj)

            tier = self._determine_tier(length)
            grouped_samples[bucket_idx][tier].append(orig_idx)

        total_samples = sum(
            len(indices)
            for b_groups in grouped_samples.values()
            for indices in b_groups.values()
        )
        print(f"Encoding {total_samples} samples into Parquet shards...")

        shard_counter = 0
        sample_records = []
        shard_metadata = []

        pbar = tqdm(total=total_samples, desc="Processing Parquet Shards")

        for bucket_idx in sorted(grouped_samples.keys()):
            res = tuple(map(int, self.dataset.bucket_resolutions[bucket_idx]))
            target_w, target_h = res

            for tier in sorted(grouped_samples[bucket_idx].keys()):
                indices = grouped_samples[bucket_idx][tier]
                if len(indices) < self.min_sample_count:
                    pbar.update(len(indices))
                    continue

                for orig_idx in indices:
                    img_path = self.dataset.paths[orig_idx]
                    booru_id = self.dataset.get_image_id(orig_idx)

                    with open(img_path, "rb") as f:
                        img_bytes = f.read()

                    prompt_obj = self.dataset._load_prompt(
                        img_path, self.dataset.label_ext
                    )
                    prompt_str, _ = self._get_prompt_str_and_len(prompt_obj)

                    orig_w, orig_h = map(int, self.dataset.raw_res[orig_idx])
                    ar = float(orig_w / orig_h) if orig_h > 0 else 1.0

                    tag_weight = 1.0
                    if isinstance(prompt_obj, dict) and "tag_weight" in prompt_obj:
                        tag_weight = float(prompt_obj["tag_weight"])

                    record = {
                        "booru_id": str(booru_id),
                        "image": img_bytes,
                        "prompt": prompt_str,
                        "bucket_idx": int(bucket_idx),
                        "target_width": target_w,
                        "target_height": target_h,
                        "original_width": orig_w,
                        "original_height": orig_h,
                        "aspect_ratio": ar,
                        "tier": int(tier),
                        "tag_weight": tag_weight,
                    }
                    sample_records.append(record)
                    pbar.update(1)

                    if len(sample_records) >= self.samples_per_shard:
                        shard_path = self._write_shard(sample_records, shard_counter)
                        shard_metadata.append(
                            {
                                "shard_file": str(shard_path.name),
                                "sample_count": len(sample_records),
                            }
                        )
                        sample_records = []
                        shard_counter += 1

        if sample_records:
            shard_path = self._write_shard(sample_records, shard_counter)
            shard_metadata.append(
                {
                    "shard_file": str(shard_path.name),
                    "sample_count": len(sample_records),
                }
            )

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
