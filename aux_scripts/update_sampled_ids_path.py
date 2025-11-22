import os
import pandas as pd
import yaml
import concurrent.futures

# Common image extensions to process
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

def scan_subdirectory(sub_dir: str, root_dir: str) -> dict[int, str]:
    """
    Scans a single subdirectory for images and returns a dictionary
    mapping their IDs to relative paths. This function is designed to be
    run in a separate thread.

    Args:
        sub_dir (str): The subdirectory to scan.
        root_dir (str): The top-level root directory for calculating
                        relative paths.

    Returns:
        dict[int, str]: A dictionary mapping file IDs to paths.
    """
    id_to_path = {}
    for root, _, files in os.walk(sub_dir):
        for file in files:
            # Check if the file has a valid image extension
            if any(file.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, root_dir)
                # Extract the ID from the filename (without extension)
                file_id = os.path.splitext(file)[0]
                try:
                    id_to_path[int(file_id)] = relative_path
                except ValueError:
                    # Ignore files that do not have an integer-based name
                    pass
    return id_to_path

def add_relative_path_to_csv(root_dir: str, csv_path: str):
    """
    Scans a directory for images, matches them with entries in a CSV file
    by their ID, and adds a 'relative_path' column to the CSV.

    Args:
        root_dir (str): The root directory containing the image files.
        csv_path (str): The path to the CSV file to be updated.
    """
    # Load the existing metadata from the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)

    
    # Get a list of subdirectories to distribute among threads.
    # This works best if your dataset is organized into subfolders.
    try:
        sub_dirs = [
            os.path.join(root_dir, d) for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]
    except FileNotFoundError:
        print(f"Error: Root directory not found at {root_dir}")
        return

    # If the root directory has no subdirectories, scan it directly.
    if not sub_dirs:
        sub_dirs = [root_dir]

    print(f"Scanning {len(sub_dirs)} subdirectories in parallel...")
    
    id_to_path = {}
    # Use a ThreadPoolExecutor to manage a pool of worker threads.
    # The 'with' statement ensures threads are cleaned up properly.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit a scan task for each subdirectory.
        future_to_dir = {
            executor.submit(scan_subdirectory, sub_dir, root_dir): sub_dir
            for sub_dir in sub_dirs
        }
        
        # As each thread completes, merge its results into the main dict.
        for future in concurrent.futures.as_completed(future_to_dir):
            try:
                result_dict = future.result()
                id_to_path.update(result_dict)
            except Exception as exc:
                dir_name = future_to_dir[future]
                print(f'{dir_name} generated an exception: {exc}')
    
    print(f"Found {len(id_to_path)} images to process")

    if 'relative_path' not in df.columns:
        df['relative_path'] = None

    # Map the 'id' column of the DataFrame to the relative paths found.
    # The .map() function is an efficient way to apply this mapping.
    df['relative_path'] = df['id'].map(id_to_path)

    # Save the updated DataFrame back to the same CSV file, overwriting it.
    # The index=False argument prevents pandas from writing the DataFrame
    # index as a column in the CSV.
    df.to_csv(csv_path, index=False)
    print(f"Successfully updated {csv_path} with relative paths.")

if __name__ == "__main__":
    config_path = "configs/default_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    add_relative_path_to_csv(
        root_dir=config["download_dir"],
        csv_path=config["sampling"]["sampled_ids_csv"]
    )