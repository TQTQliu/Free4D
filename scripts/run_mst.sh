SCENE=$1
python lib/viewcrafter/inference.py \
--image_dir data/vc/${SCENE} \
--traj_txt configs/vc_configs/trajs/${SCENE}.txt \
--exp_name ${SCENE} \
--out_dir ./output/vc/ \
--mode 'single_view_txt' \
--recon monst3r \
--ddim_steps 50 \
--video_length 25 \
--height 576 --width 1024 \
--config configs/vc_configs/inference_pvd_1024.yaml \
--ckpt_path ckpt/vc/model.ckpt \
--model_path ckpt/vc/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth