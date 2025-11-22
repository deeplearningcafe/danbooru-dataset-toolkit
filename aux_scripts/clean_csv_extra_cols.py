import os
import yaml

def clean_csv_file(file_path: str):
    """
    Cleans a CSV file by ensuring all rows have the same number of
    columns as the header. It truncates rows that are too long.

    Args:
        file_path (str): The path to the CSV file to clean.
    """
    if not os.path.exists(file_path):
        print(f"File not found at '{file_path}'. Nothing to clean.")
        return

    print(f"\n--- Starting CSV cleaning process for '{file_path}' ---")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            print("File is empty. No cleaning needed.")
            return

        # Determine the correct number of columns from the header
        header = lines[0].strip()
        num_columns = len(header.split(','))

        cleaned_lines = [header + '\n']
        cleaned_count = 0

        # Process each line, skipping the header
        for i, line in enumerate(lines[1:]):
            parts = line.strip().split(',')
            if len(parts) > num_columns:
                # Truncate the line to the correct number of columns
                cleaned_line = ",".join(parts[:num_columns]) + '\n'
                cleaned_lines.append(cleaned_line)
                cleaned_count += 1
            else:
                cleaned_lines.append(line)

        if cleaned_count > 0:
            print(f"Found and cleaned {cleaned_count} corrupted rows.")
            # Overwrite the original file with the cleaned data
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(cleaned_lines)
            print(f"Successfully cleaned and saved '{file_path}'.")
        else:
            print("No corrupted rows found. File is already clean.")

    except Exception as e:
        print(f"An error occurred during the cleaning process: {e}")

if __name__ == "__main__":
    config_path = "configs/default_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    csv_path = config['prior_data']['output_csv_path']
    clean_csv_file(csv_path)