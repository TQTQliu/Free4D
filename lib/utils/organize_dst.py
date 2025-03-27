import os
import glob
import shutil
import argparse

def organize_files(vd_dir, mv_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Get sorted lists of image files
    imgs_mv_ls = glob.glob(os.path.join(mv_dir, '*'))
    imgs_mv_ls.sort()
    len_mv = len(imgs_mv_ls)

    imgs_vd_ls = glob.glob(os.path.join(vd_dir, '*'))
    imgs_vd_ls.sort()
    len_vd = len(imgs_vd_ls)

    # Organize files into output directory
    for i in range(len_mv):
        dir_cam = os.path.join(output_dir, f"cam{i + 1:02d}")
        os.makedirs(dir_cam, exist_ok=True)
        
        # Copy mv files
        shutil.copyfile(imgs_mv_ls[i], os.path.join(dir_cam, f"frame_{1:05d}.jpg"))

        # Copy vd files to the first camera directory
        if i == 0:
            for j in range(1, len_vd):
                shutil.copyfile(imgs_vd_ls[j], os.path.join(dir_cam, f"frame_{j + 1:05d}.jpg"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize files.")
    parser.add_argument("-vd", "--input_vd_dir", type=str, required=True, help="Path to the input directory containing video frames.")
    parser.add_argument("-mv", "--input_mv_dir", type=str, required=True, help="Path to the input directory containing multi-view images.")
    parser.add_argument("-o", "--output_directory", type=str, required=True, help="Path to the output directory where the reorganized structure will be saved.")
    args = parser.parse_args()

    organize_files(args.input_vd_dir, args.input_mv_dir, args.output_directory)