mkdir -p ckpt/vc
# monst3r
gdown --fuzzy https://drive.google.com/file/d/1Z1jO_JmfZj0z3bgMvCwqfUhyZ1bIbc9E/view?usp=sharing -O ckpt/vc/
# sam2 ckpt
cd lib/viewcrafter/extern/monst3r/third_party/sam2
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -P checkpoints/
cd ../../../../../../
# sea-raft ckpt
mkdir -p ckpt/sea_raft
gdown --fuzzy https://drive.google.com/file/d/1a0C5FTdhjM4rKrfXiGhec7eq2YM141lu/view?usp=drive_link -O ckpt/sea_raft/

# dust3r
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P ckpt/vc/

# viewcrafter
wget https://huggingface.co/Drexubery/ViewCrafter_25/resolve/main/model.ckpt -P ckpt/vc/

mkdir -p ckpt/lcm
# stable diffusion - lcm
# original SD1.5 ckpt will be automatically downloaded from https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
wget https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors -P ckpt/lcm/