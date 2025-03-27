import os
from PIL import Image
import numpy as np
import torch
import torchvision
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
import importlib
from collections import OrderedDict
from utils_vc.diffusion_utils import image_guided_synthesis

def load_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            ## rename the keys for 256x256 model
            new_pl_sd = OrderedDict()
            for k,v in state_dict.items():
                new_pl_sd[k] = v
            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=True)
    else:
        # deepspeed
        new_pl_sd = OrderedDict()
        for key in state_dict['module'].keys():
            new_pl_sd[key[16:]]=state_dict['module'][key]
        model.load_state_dict(new_pl_sd)
    print('>>> model checkpoint loaded.')
    return model

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def save_video(data,images_path,folder=None):
    if isinstance(data, np.ndarray):
        tensor_data = (torch.from_numpy(data) * 255).to(torch.uint8)
    elif isinstance(data, torch.Tensor):
        tensor_data = (data.detach().cpu() * 255).to(torch.uint8)
    elif isinstance(data, list):
        folder = [folder]*len(data)
        images = [np.array(Image.open(os.path.join(folder_name,path))) for folder_name,path in zip(folder,data)]
        stacked_images = np.stack(images, axis=0)
        tensor_data = torch.from_numpy(stacked_images).to(torch.uint8)
    torchvision.io.write_video(images_path, tensor_data, fps=8, video_codec='h264', options={'crf': '10'})

def setup_diffusion(opts):
    seed_everything(opts.seed)
    config = OmegaConf.load(opts.config)
    model_config = config.pop("model", OmegaConf.create())
    ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
    model_config['params']['unet_config']['params']['use_checkpoint'] = False
    model = instantiate_from_config(model_config)
    model = model.to(opts.device)
    model.cond_stage_model.device = opts.device
    model.perframe_ae = opts.perframe_ae
    assert os.path.exists(opts.ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, opts.ckpt_path)
    model.eval()
    diffusion = model
    h, w = opts.height // 8, opts.width // 8
    channels = model.model.diffusion_model.out_channels
    n_frames = opts.video_length
    noise_shape = [opts.bs, channels, n_frames, h, w]
    return noise_shape, diffusion

def run_diffusion(opts, renderings, viewmask, noise_shape, diffusion, x0_ref=None, replace_mask=None):
    prompts = [opts.prompt]
    videos = (renderings * 2. - 1.).permute(3,0,1,2).unsqueeze(0).cuda()
    condition_index = [0]
    with torch.no_grad(), torch.cuda.amp.autocast():
        # [1,1,c,t,h,w]
        batch_samples = image_guided_synthesis(diffusion, prompts, videos, noise_shape, opts.n_samples, opts.ddim_steps, opts.ddim_eta, \
                            opts.unconditional_guidance_scale, opts.cfg_img, opts.frame_stride, opts.text_input, opts.multiple_cond_cfg, opts.timestep_spacing, opts.guidance_rescale, condition_index, viewmask=viewmask, x0_ref=x0_ref, replace_mask=replace_mask)
    return torch.clamp(batch_samples[0][0].permute(1,2,3,0), -1., 1.)