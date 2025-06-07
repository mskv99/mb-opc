import os
import random
import shutil

# Define the paths to the directories
root_dir = "data/processed/gds_dataset"
origin_dir = os.path.join(root_dir, "origin")
correction_dir = os.path.join(root_dir, "correction")
train_origin_dir = os.path.join(origin_dir, "train_origin")
train_correction_dir = os.path.join(correction_dir, "train_correction")
valid_origin_dir = os.path.join(origin_dir, "test_origin")
valid_correction_dir = os.path.join(correction_dir, "test_correction")

# Create validation directories if they don't exist
os.makedirs(valid_origin_dir, exist_ok=True)
os.makedirs(valid_correction_dir, exist_ok=True)


# Function to move a percentage of files from source to destination
def move_matching_files(
    origin_dir, correction_dir, dest_origin_dir, dest_correction_dir, percentage=0.1
):
    # List all files in the train_origin directory
    origin_files = os.listdir(origin_dir)

    # Calculate the number of files to move
    num_files_to_move = int(len(origin_files) * percentage)

    # Randomly select files to move from the train_origin directory
    files_to_move = random.sample(origin_files, num_files_to_move)

    for file_name in files_to_move:
        # Construct full file paths
        src_origin_file_path = os.path.join(origin_dir, file_name)
        src_correction_file_path = os.path.join(correction_dir, file_name)

        # Move the files
        dest_origin_file_path = os.path.join(dest_origin_dir, file_name)
        dest_correction_file_path = os.path.join(dest_correction_dir, file_name)

        # Move the train_origin file
        shutil.move(src_origin_file_path, dest_origin_file_path)
        print(f"Moved: {src_origin_file_path} to {dest_origin_file_path}")

        # Move the corresponding train_correction file
        shutil.move(src_correction_file_path, dest_correction_file_path)
        print(f"Moved: {src_correction_file_path} to {dest_correction_file_path}")


# Move 10% of the files from train_origin to valid_origin and from train_correction to valid_correction
move_matching_files(
    train_origin_dir,
    train_correction_dir,
    valid_origin_dir,
    valid_correction_dir,
    percentage=0.01,
)

print("Test subset created successfully.")
