import os
import random
import fire


def sync_random_delete(dir1: str, dir2: str, percentage: float):
    """
    Randomly deletes a percentage of files from dir1 and matching files from dir2.

    Args:
        dir1 (str): Path to first directory
        dir2 (str): Path to second directory
        percentage (float): Percentage of files to delete (0.0 to 1.0)

    Returns:
        tuple: (list of deleted files from dir1, list of deleted files from dir2)
    """
    # Verify directories exist
    if not os.path.isdir(dir1) or not os.path.isdir(dir2):
        raise ValueError("Both paths must be valid directories")

    # Get matching file lists (without full paths)
    files1 = set(os.listdir(dir1))
    files2 = set(os.listdir(dir2))

    # Find common files
    common_files = sorted(files1 & files2)

    if not common_files:
        print("Warning: No matching files found between directories")
        return [], []

    # Calculate number of files to delete
    num_to_delete = int(len(common_files) * percentage)
    if num_to_delete < 1:
        print("Percentage too small - no files deleted")
        return [], []

    # Randomly select files
    files_to_delete = random.sample(common_files, num_to_delete)

    # Delete files from both directories
    deleted_from_dir1 = []
    deleted_from_dir2 = []

    for filename in files_to_delete:
        path1 = os.path.join(dir1, filename)
        path2 = os.path.join(dir2, filename)

        try:
            os.remove(path1)
            deleted_from_dir1.append(filename)
        except OSError as e:
            print(f"Error deleting {path1}: {e}")

        try:
            os.remove(path2)
            deleted_from_dir2.append(filename)
        except OSError as e:
            print(f"Error deleting {path2}: {e}")

    # Verification
    remaining1 = set(os.listdir(dir1))
    remaining2 = set(os.listdir(dir2))

    if remaining1 != remaining2:
        print("Warning: Directory contents no longer match completely")
        print("Differences:", remaining1.symmetric_difference(remaining2))
    else:
        print(f"Successfully deleted {len(deleted_from_dir1)} file pairs")
        print(f"Remaining files: {len(remaining1)} in each directory")

    return deleted_from_dir1, deleted_from_dir2


if __name__ == "__main__":
    fire.Fire(sync_random_delete)
