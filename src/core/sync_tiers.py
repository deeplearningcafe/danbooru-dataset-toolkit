"""Synchronizes dataset directory layout with updated quality tiers.

Moves image, prompt (.txt), and token (.json) files into their target
tier folders (e.g. 0, 1, 2, 3) according to final_tiers_csv without
re-downloading or re-extracting features.
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from typing import Optional, List
from ..utils.loader import build_id_path_map, AESTHETIC_LABEL


def sync_tiers_on_disk(
    download_dir: str,
    final_tiers_csv: str,
    sampled_ids_csv: Optional[str] = None,
    character_list: Optional[List[str]] = None,
    verbose: bool = True,
) -> int:
    """Moves files to match final_tier folders and updates CSV paths."""
    root_path = Path(download_dir)
    if not root_path.exists() or not os.path.exists(final_tiers_csv):
        print(f"Error: Path {download_dir} or {final_tiers_csv} not found.")
        return 0

    df_tiers = pd.read_csv(final_tiers_csv)
    tier_map = dict(zip(df_tiers["id"], df_tiers["final_tier"]))

    id_path_map = build_id_path_map(root_path)
    if verbose:
        print(f"Found {len(id_path_map)} total image files on disk.")

    moved_count = 0
    for img_id, src_img_path in id_path_map.items():
        if img_id not in tier_map:
            continue

        target_tier_name = tier_map[img_id]
        target_class_id = str(AESTHETIC_LABEL.get(target_tier_name, 1))

        # Determine target directory (preserve character folder if present)
        parent_dir = src_img_path.parent
        grandparent_dir = parent_dir.parent

        if (
            character_list
            and grandparent_dir != root_path
            and grandparent_dir.name in character_list
        ):
            target_dir = root_path / grandparent_dir.name / target_class_id
        else:
            target_dir = root_path / target_class_id

        if parent_dir.resolve() == target_dir.resolve():
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        dest_img_path = target_dir / src_img_path.name

        # Move image, .txt prompt, and .json token files
        shutil.move(str(src_img_path), str(dest_img_path))

        for ext in [".txt", ".json", ".tmp"]:
            src_aux = src_img_path.with_suffix(ext)
            if src_aux.exists():
                shutil.move(str(src_aux), str(target_dir / src_aux.name))

        moved_count += 1

    if verbose:
        print(f"Successfully migrated {moved_count} items to target tiers.")

    # Update relative_path in sampled_ids_csv if provided
    if sampled_ids_csv and os.path.exists(sampled_ids_csv):
        df_sampled = pd.read_csv(sampled_ids_csv)
        updated_map = build_id_path_map(root_path)

        def _get_new_rel_path(row_id):
            if row_id in updated_map:
                return str(updated_map[row_id].relative_to(root_path))
            return None

        df_sampled["relative_path"] = df_sampled["id"].apply(_get_new_rel_path)
        df_sampled.to_csv(sampled_ids_csv, index=False)
        if verbose:
            print(f"Updated relative paths in '{sampled_ids_csv}'.")

    return moved_count
