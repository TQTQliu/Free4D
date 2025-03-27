import os
import shutil
import argparse

def reorganize_folders(input_dir, output_dir):
    """
    Reorganizes folder structure.
    
    :param input_dir: Path to the original directory (contains N subfolders, each with M images).
    :param output_dir: Path to the output directory (will contain M subfolders, each with N images).
    """
    # Get all subfolders in the input directory
    subfolders = [os.path.join(input_dir, folder) for folder in os.listdir(input_dir)
                  if os.path.isdir(os.path.join(input_dir, folder))]
    subfolders.sort()
    # Get image paths from each subfolder
    all_images = []
    for folder in subfolders:
        images = [os.path.join(folder, img) for img in os.listdir(folder)
                  if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        images.sort()
        all_images.append(images)

    # Check if all subfolders have the same number of images
    num_images_per_folder = len(all_images[0])
    for images in all_images:
        if len(images) != num_images_per_folder:
            raise ValueError("Inconsistent number of images in subfolders. Please check the input directory structure!")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Reorganize images into the new folder structure
    for i in range(num_images_per_folder):  # Iterate over columns
        new_folder_name = f"cam{i + 1:02d}"
        new_folder_path = os.path.join(output_dir, new_folder_name)
        os.makedirs(new_folder_path)

        for j, images in enumerate(all_images):  # Iterate over rows
            img_path = images[i]
            new_img_name = f"frame_{j + 1:05d}{os.path.splitext(img_path)[1]}"
            new_img_path = os.path.join(new_folder_path, new_img_name)
            shutil.copy(img_path, new_img_path)

    print(f"Folder structure reorganized to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize folder structure of images.")
    parser.add_argument("-i", "--input_directory", type=str, required=True, help="Path to the input directory containing subfolders with images.")
    parser.add_argument("-o", "--output_directory", type=str, required=True, help="Path to the output directory where the reorganized structure will be saved.")
    args = parser.parse_args()

    reorganize_folders(args.input_directory, args.output_directory)