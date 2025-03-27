SCENE=$1
python lib/utils/extractimages.py data/gs/${SCENE}
DATASET_PATH=data/gs/${SCENE}/colmap
colmap feature_extractor \
   --database_path ${DATASET_PATH}/database.db \
   --image_path ${DATASET_PATH}/images
colmap exhaustive_matcher \
   --database_path ${DATASET_PATH}/database.db
mkdir ${DATASET_PATH}/sparse
colmap mapper \
    --database_path ${DATASET_PATH}/database.db \
    --image_path ${DATASET_PATH}/images \
    --output_path ${DATASET_PATH}/sparse
mkdir ${DATASET_PATH}/dense
colmap image_undistorter \
    --image_path ${DATASET_PATH}/images \
    --input_path ${DATASET_PATH}/sparse/0 \
    --output_path ${DATASET_PATH}/dense \
    --output_type COLMAP \
    --max_image_size 2000
colmap patch_match_stereo \
    --workspace_path ${DATASET_PATH}/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true
colmap stereo_fusion \
    --workspace_path ${DATASET_PATH}/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path ${DATASET_PATH}/dense/fused.ply

DATASET_PATH=data/gs/${SCENE}/colmap
mkdir ./data/gs/${SCENE}/sparse_
cp -r ${DATASET_PATH}/sparse/0/* ./data/gs/${SCENE}/sparse_
python lib/utils/downsample_point.py ${DATASET_PATH}/dense/fused.ply ./data/gs/${SCENE}/points3D_multipleview.ply
python lib/utils/llff_utils/imgs2poses.py ${DATASET_PATH}
cp ${DATASET_PATH}/poses_bounds.npy ./data/gs/${SCENE}/poses_bounds_multipleview.npy