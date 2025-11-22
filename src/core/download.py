import os
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.parse
import time
import threading
from pathlib import Path
from typing import List, Optional

class Downloader:
    """
    Handles the downloading of images from URLs specified in a DataFrame.
    """
    def __init__(
        self,
        max_workers: int = 8,
        timeout: int = 10,
        max_downloads: int= 50000,
        retry_attempts: int = 3,
        initial_retry_delay: int = 1
    ):
        """
        Initializes the downloader.

        Args:
            max_workers (int): Number of parallel download workers.
            timeout (int): Timeout in seconds for each download request.
            max_downloads (int): Maximum number of images to download in one
                                 run. If None, downloads all images from the
                                 start_index.
            retry_attempts (int): Number of times to retry a failed
                                  download.
            initial_retry_delay (int): Initial delay in seconds for retries.
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_downloads = max_downloads
        self.retry_attempts = retry_attempts
        self.initial_retry_delay = initial_retry_delay
        self.cf_clearance_cookie = "YOUR-COOKIE"

    def _download_image_with_retry(
        self,
        index,
        row,
        headers,
        output_dir,
        character_list: Optional[List[str]] = None
    ):
        """
        Downloads a single image with a retry mechanism for transient errors.
        Returns the relative path on success, None on failure.
        """
        url = row.get("large_file_url", "")
        img_width = row.get('image_width', 0)
        img_height =  row.get('image_height', 0)
        # if small file is too small, we use the original size
        if len(url) == 0 or img_width * img_height <= 768**2 or img_height <= 1200 :
            url = row.get("file_url", "")

        if not url or not isinstance(url, str):
            print(f"Skipping row {index} due to invalid URL.")
            return False

        valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        quality2id = {
            'masterpiece': 3, 'good_score': 2, 'bad_score': 1, 'worse_score': 0
        }

        # Determine class_id and create filename
        if "aesthetic_class" in row.index:
            class_id = row["aesthetic_class"]
        elif "quality_tier" in row.index:
            class_id = quality2id.get(row["quality_tier"], 0)
        else:
            class_id = 0 # Default class

        # Determine Sub-directory based on Character
        # Default folder is just the class_id
        relative_subdir = str(class_id)

        if len(character_list) > 0:
            # Parse the tag string into a set for exact matching.
            # Danbooru tags are space-separated.
            image_tags = set(row.get("tag_string", "").split())

            for char_tag in character_list:
                if char_tag in image_tags:
                    # If match found, nest class_id inside character folder
                    # Format: character_name/class_id/
                    relative_subdir = os.path.join(char_tag, str(class_id))
                    break

        id = row["id"]
        parsed_url = urllib.parse.urlparse(url)
        _, ext = os.path.splitext(os.path.basename(parsed_url.path).lower())

        if ext not in valid_extensions:
            print(f"Skipping invalid extension for row {index}: {ext}")
            return None

        filename = f"{id}{ext}"
        relative_path = os.path.join(relative_subdir, filename)
        save_path = os.path.join(output_dir, relative_path)
        temp_file = save_path + ".tmp"

        # Ensure the specific directory exists (dynamic creation)
        # not necessary when no chars as created in main
        if len(character_list) > 0:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if os.path.exists(save_path):
            # Consider existing files a success to avoid re-downloading
            return relative_path

        delay = self.initial_retry_delay
        for attempt in range(self.retry_attempts):
            start_time = time.time()
            try:
                cookies = {'cf_clearance': self.cf_clearance_cookie}
                response = requests.get(
                    url, stream=True, timeout=self.timeout, headers=headers,
                    cookies=cookies
                )
                response.raise_for_status()

                file_size = 0
                with open(temp_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if time.time() - start_time > self.timeout:
                            raise TimeoutError("Download exceeded timeout")
                        f.write(chunk)
                        file_size += len(chunk)

                os.rename(temp_file, save_path)
                elapsed = time.time() - start_time
                speed = (file_size / 1024) / elapsed if elapsed > 0 else 0
                print(
                    f"Downloaded: {save_path} in {elapsed:.2f}s "
                    f"({speed:.2f} KB/s)"
                )
                time.sleep(0.35)
                return relative_path  # Success

            except (requests.RequestException, TimeoutError) as e:
                print(
                    f"Attempt {attempt + 1}/{self.retry_attempts} failed for "
                    f"id {id}: {e}. Retrying in {delay}s..."
                )
                if os.path.exists(temp_file):
                    os.remove(temp_file)

                if attempt < self.retry_attempts - 1:
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    print(f"All retries failed for id {id}.")
                    return None

            except Exception as e:
                print(f"Unexpected error for id {id} ({url}): {e}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                return None # Do not retry on other errors
        return None

    def download_images(
        self,
        df: pd.DataFrame,
        output_dir: str,
        csv_path: str,
        output_csv_path: str,
        start_index: int = 0,
        character_list: Optional[List[str]] = None
    ) -> list:
        """
        Download images from the dataframe to the specified output directory.

        This function supports partial downloads and resuming, and it cleans
        the input CSV file by removing entries for images that failed to
        download.

        Args:
            df (pd.DataFrame): DataFrame with image URLs and metadata.
            output_dir (str): Directory to save the downloaded images.
            csv_path (str): Path to the source CSV. This file will be
                            updated to remove failed downloads.
            start_index (int): Index in the dataframe to start downloading.
            character_list (List[str]): Optional list of characters to organize
                            images into specific folders.
        """

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Add 'relative_path' column if it doesn't exist
        if 'relative_path' not in df.columns:
            df['relative_path'] = None

        # if there is characters, they will be created dynamically
        if len(character_list) == 0:
            # Create class subdirectories
            for class_id in range(4):
                class_dir = os.path.join(output_dir, str(class_id))
                os.makedirs(class_dir, exist_ok=True)
        else:
            print(f"Downloading images for {', '.join(character_list)} characters")

        # Correctly slice the DataFrame to respect max_downloads
        if self.max_downloads is not None:
            end_index = start_index + self.max_downloads
            df_subset = df.iloc[start_index:end_index]
        else:
            df_subset = df.iloc[start_index:]


        download_count = 0
        failed_indices = []
        lock = threading.Lock()
        headers = None#{'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36'} # Add User-Agent

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit download tasks using the new retry-enabled method
            futures = {
                executor.submit(
                    self._download_image_with_retry,
                    index,
                    row,
                    headers,
                    output_dir,
                    character_list
                ): index
                for index, row in df_subset.iterrows()
            }


            print(f"Submitted {len(futures)} download tasks.")
            for future in as_completed(futures):
                index = futures[future]
                relative_path = future.result()
                with lock:
                    if relative_path:
                        download_count += 1
                        # Update DataFrame with the relative path
                        df.loc[index, 'relative_path'] = relative_path
                    else:
                        failed_indices.append(index)

            print(
                "Finished. Successful downloads in this run: "
                f"{download_count}"
            )

        if failed_indices:
            print(f"\nDropping {len(failed_indices)} failed download entries.")
            # df.drop(failed_indices, inplace=True)
            missing_ids = df.loc[failed_indices, 'id']

            exclude_df = pd.DataFrame(list(missing_ids), columns=['id'])
            Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
            exclude_df.to_csv(output_csv_path, index=False)


        # Always save the DataFrame to persist new relative_path values
        # and remove rows for failed downloads.
        print(f"Saving updated dataframe to {csv_path}...")
        df.to_csv(csv_path, index=False)
        print("Save complete.")

        if not failed_indices:
            print("\nNo download failures detected in this run.")
