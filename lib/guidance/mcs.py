import torch
import numpy as np
import torchvision.transforms as tvtf
from guidance.Hack_SD_stepwise import Hack_SDPipe_Stepwise

def step_gaussian_optimization(step, RGB_LCM, rgb_prompt_latent, denoise_steps):
    fix_step = denoise_steps
    # denoise to x0 and d0
    with torch.no_grad():
        # we left the last 2 steps for stronger guidances
        rgb_t = RGB_LCM.timesteps[-fix_step+step]
        rgb_t = torch.tensor([rgb_t]).cuda()
        rgb_noise_pr,rgb_denoise = RGB_LCM._denoise_to_x0(rgb_t, rgb_prompt_latent)
        rgb_denoise = rgb_denoise.permute(0,2,3,1)
    return rgb_t, rgb_noise_pr, rgb_denoise

'''
Input: Multiview images with added noise
denoise to x0
denoise from step t1 to step t2
'''    

class HackSD_MCS():
    '''
        transform images to self.latents
        add noise to self.latents
        predict step noise --> x0
        mv RGB-D warp as target image
        target image encode to latent and get target noise
        noise rectification
        step denoise
    '''
    def __init__(self,device='cpu',use_lcm=True,denoise_steps=5,
                sd_ckpt=f'stable-diffusion-v1-5/stable-diffusion-v1-5',
                lcm_ckpt=f'latent-consistency/lcm-lora-sdv1-5') -> None:
        '''
        ref_rgb should be -1~1 tensor B*3*H*W
        '''
        self.device = device
        self.target_type = np.float32
        self.use_lcm = use_lcm
        self.sd_ckpt = sd_ckpt
        self.lcm_ckpt = lcm_ckpt
        self._load_model()
        # define step to add noise and steps to denoise
        self.denoise_steps = denoise_steps
        self.timesteps = self.model.timesteps

    def _load_model(self):
        self.model = Hack_SDPipe_Stepwise.from_pretrained(self.sd_ckpt)
        self.model._use_lcm(self.use_lcm,self.lcm_ckpt)
        self.model.re_init(num_inference_steps=50)
        try:
            self.model.enable_xformers_memory_efficient_attention()
        except:
            pass  # run without xformers
        self.model = self.model.to(self.device)

    def to(self, device):
        self.device = device
        self.model.to(device)

    @ torch.no_grad()
    def _add_noise_to_latent(self,latents):
        bsz = latents.shape[0]
        # in the Stable Diffusion, the iterations numbers is 1000 for adding the noise and denosing.
        timestep = self.timesteps[-self.denoise_steps]
        timestep = timestep.repeat(bsz).to(self.device)
        # target noise
        noise = torch.randn_like(latents)
        # add noise
        noisy_latent = self.model.scheduler.add_noise(latents, noise, timestep)
        # -------------------- noise for supervision -----------------
        if self.model.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.model.scheduler.config.prediction_type == "v_prediction":
            target = self.model.scheduler.get_velocity(latents, noise, timestep)
        return noisy_latent, timestep, target

    @ torch.no_grad()
    def _encode_mv_init_images(self, images):
        '''
        images should be B3HW
        '''
        images = images * 2 - 1
        self.latents = self.model._encode(images)
        self.latents,_,_ = self._add_noise_to_latent(self.latents)

    @ torch.no_grad()
    def _sd_forward(self, denoise_step, prompt_latent:torch.Tensor):
        # temp noise prediction
        t = self.timesteps[[-self.denoise_steps+denoise_step]].to(self.device)
        noise_pred = self.model._step_noise(self.latents, t, prompt_latent.repeat(len(self.latents),1,1))
        # solve image
        _,x0 = self.model._solve_x0(self.latents,noise_pred,t)   
        x0 = (x0 + 1) / 2 # in 0-1
        return t, noise_pred, x0   
    
    def _reset(self):
        self.model.re_init(num_inference_steps=50)

    @ torch.no_grad()
    def _denoise_to_x0(self, timestep_in_1000, prompt_latent:torch.Tensor):
        # temp noise prediction
        noise_pred = self.model._step_noise(self.latents, timestep_in_1000, prompt_latent.repeat(len(self.latents),1,1))
        # solve image
        _,x0 = self.model._solve_x0(self.latents,noise_pred,timestep_in_1000)   
        x0 = (x0 + 1) / 2 # in 0-1
        return noise_pred, x0   

    @ torch.no_grad()
    def _step_denoise(self, t, pred_noise, rect_x0, rect_w):
        '''
        pred_noise B4H//8W//8
        x0, rect_x0 B3HW
        '''
        # encoder rect_x0 to latent
        rect_x0 = rect_x0 * 2 - 1
        rect_latent = self.model._encode(rect_x0)
        # rectified noise
        rect_noise = self.model._solve_noise_given_x0_latent(self.latents,rect_latent,t)
        # noise rectification
        rect_noise = rect_noise / rect_noise.std(dim=list(range(1, rect_noise.ndim)),keepdim=True) \
                                * pred_noise.std(dim=list(range(1, pred_noise.ndim)),keepdim=True)
        
        pred_noise = pred_noise*(1.-rect_w) + rect_noise*rect_w
        # step forward
        self.latents = self.model._step_denoise(self.latents,pred_noise,t)

    @ torch.no_grad()
    def _decode_mv_imgs(self):
        imgs = self.model._decode(self.latents)
        imgs = (imgs + 1) / 2
        return imgs

