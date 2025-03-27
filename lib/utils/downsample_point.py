import open3d as o3d
import sys
def process_ply_file(input_file, output_file):
    pcd = o3d.io.read_point_cloud(input_file)
    print(f"Total points: {len(pcd.points)}")
    voxel_size=0.02
    while len(pcd.points) > 40000:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"Downsampled points: {len(pcd.points)}")
        voxel_size+=0.01

    o3d.io.write_point_cloud(output_file, pcd)

process_ply_file(sys.argv[1], sys.argv[2])