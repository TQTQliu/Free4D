import sys
sys.path.append('./lib/viewcrafter/extern')
from dust3r.dust3r.inference import inference, load_model
from dust3r.dust3r.utils.image import load_images
from dust3r.dust3r.image_pairs import make_pairs
from dust3r.dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.dust3r.utils.device import to_numpy
import torch
import numpy as np
import os
import copy
import shutil
from pytorch3d.structures import Pointclouds
import torch.nn.functional as F
from utils_vc.pvd_utils import *
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from utils_vc.diffusion_utils import instantiate_from_config,load_model_checkpoint,image_guided_synthesis

class ViewCrafter:
    def __init__(self, opts):
        self.opts = opts
        self.device = opts.device
        self.setup_dust3r()
        self.setup_diffusion()
        # initialize ref images, pcd
        if os.path.isfile(self.opts.image_dir):
            resize_images(self.opts.image_dir)
            self.images, self.img_ori = self.load_initial_images(image_dir=self.opts.image_dir)
            self.run_dust3r(input_images=self.images)
        else:
            print(f"{self.opts.image_dir} doesn't exist")           
        
    def run_dust3r(self, input_images, clean_pc = False):
        pairs = make_pairs(input_images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, self.dust3r, self.device, batch_size=self.opts.batch_size)
        mode = GlobalAlignerMode.PointCloudOptimizer #if len(self.images) > 2 else GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=self.device, mode=mode)
        if mode == GlobalAlignerMode.PointCloudOptimizer:
            loss = scene.compute_global_alignment(init='mst', niter=self.opts.niter, schedule=self.opts.schedule, lr=self.opts.lr)
        if clean_pc:
            self.scene = scene.clean_pointcloud()
        else:
            self.scene = scene

    def render_pcd(self,pts3d,imgs,masks,views,renderer,device,nbv=True):
        imgs = to_numpy(imgs)
        pts3d = to_numpy(pts3d)
        if masks == None:
            pts = torch.from_numpy(np.concatenate([p for p in pts3d])).view(-1, 3).to(device)
            col = torch.from_numpy(np.concatenate([p for p in imgs])).view(-1, 3).to(device)
        else:
            pts = torch.from_numpy(np.concatenate([p[m] for p, m in zip(pts3d, masks)])).to(device)
            col = torch.from_numpy(np.concatenate([p[m] for p, m in zip(imgs, masks)])).to(device)
        point_cloud = Pointclouds(points=[pts], features=[col]).extend(views)
        images = renderer(point_cloud)
        if nbv:
            color_mask = torch.ones(col.shape).to(device)
            point_cloud_mask = Pointclouds(points=[pts],features=[color_mask]).extend(views)
            view_masks = renderer(point_cloud_mask)
        else: 
            view_masks = None
        return images, view_masks
    
    def run_render(self, pcd, imgs,masks, H, W, camera_traj, num_views, nbv=True):
        render_setup = setup_renderer(camera_traj, image_size=(H,W))
        renderer = render_setup['renderer']
        render_results, viewmask = self.render_pcd(pcd, imgs, masks, num_views, renderer, self.device, nbv=nbv)
        return render_results, viewmask
    
    def run_diffusion(self, renderings, viewmask=None):
        prompts = [self.opts.prompt]
        videos = (renderings * 2. - 1.).permute(3,0,1,2).unsqueeze(0).to(self.device)
        condition_index = [0]
        with torch.no_grad(), torch.cuda.amp.autocast():
            batch_samples = image_guided_synthesis(self.diffusion, prompts, videos, self.noise_shape, self.opts.n_samples, self.opts.ddim_steps, self.opts.ddim_eta, \
                               self.opts.unconditional_guidance_scale, self.opts.cfg_img, self.opts.frame_stride, self.opts.text_input, self.opts.multiple_cond_cfg, self.opts.timestep_spacing, self.opts.guidance_rescale, condition_index, viewmask=viewmask)
        return torch.clamp(batch_samples[0][0].permute(1,2,3,0), -1., 1.) 

    def setup_diffusion(self):
        seed_everything(self.opts.seed)
        config = OmegaConf.load(self.opts.config)
        model_config = config.pop("model", OmegaConf.create())
        ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
        model_config['params']['unet_config']['params']['use_checkpoint'] = False
        model = instantiate_from_config(model_config)
        model = model.to(self.device)
        model.cond_stage_model.device = self.device
        model.perframe_ae = self.opts.perframe_ae
        assert os.path.exists(self.opts.ckpt_path), "Error: checkpoint Not Found!"
        model = load_model_checkpoint(model, self.opts.ckpt_path)
        model.eval()
        self.diffusion = model
        h, w = self.opts.height // 8, self.opts.width // 8
        channels = model.model.diffusion_model.out_channels
        n_frames = self.opts.video_length
        self.noise_shape = [self.opts.bs, channels, n_frames, h, w]

    def setup_dust3r(self):
        self.dust3r = load_model(self.opts.model_path, self.device)
    
    def load_initial_images(self, image_dir):
        ## load images
        images = load_images([image_dir], size=512,force_1024 = True)
        img_ori = (images[0]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2. # [576,1024,3] [0,1]
        if len(images) == 1:
            images = [images[0], copy.deepcopy(images[0])]
            images[1]['idx'] = 1
        return images, img_ori

    def nvs_single_view(self):
        c2ws = self.scene.get_im_poses().detach()[1:] 
        principal_points = self.scene.get_principal_points().detach()[1:] #cx cy
        focals = self.scene.get_focals().detach()[1:] 
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)] # a list of points of size whc
        depth = [i.detach() for i in self.scene.get_depthmaps()]
        depth_avg = depth[-1][H//2,W//2] 
        radius = depth_avg*self.opts.center_scale 
        ## change coordinate
        c2ws,pcd =  world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=-1, r=radius, elevation=self.opts.elevation, device=self.device)
        imgs = np.array(self.scene.imgs)
        masks = None
        if self.opts.mode == 'single_view_txt':
            with open(self.opts.traj_txt, 'r') as file:
                lines = file.readlines()
                phi = [float(i) for i in lines[0].split()]
                theta = [float(i) for i in lines[1].split()]
                r = [float(i) for i in lines[2].split()]
                shutil.copy(self.opts.traj_txt, self.opts.save_dir)
            camera_traj, num_views = generate_traj_txt(c2ws, H, W, focals, principal_points, phi, theta, r,self.opts.video_length, self.device,viz_traj=True, save_dir = self.opts.save_dir)
        else:
            raise KeyError(f"Invalid Mode: {self.opts.mode}")
        render_results, viewmask = self.run_render([pcd[-1]], [imgs[-1]],masks, H, W, camera_traj, num_views, nbv=True)
        viewmask = F.interpolate(viewmask.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        render_results[0] = self.img_ori
        if self.opts.mode == 'single_view_txt':
            if phi[-1]==0. and theta[-1]==0. and r[-1]==0.:
                render_results[-1] = self.img_ori
        save_video(render_results, os.path.join(self.opts.save_dir, 'render0.mp4'))
        save_video(viewmask, os.path.join(self.opts.save_dir, f'mask0.mp4'))
        # save_pointcloud_with_normals([imgs[-1]], [pcd[-1]], msk=None, save_path=os.path.join(self.opts.save_dir,'pcd0.ply') , mask_pc=False, reduce_pc=False)
        diffusion_results = self.run_diffusion(render_results, viewmask=viewmask)
        save_video((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, 'diffusion0.mp4'))
        img_save_dir = os.path.join(self.opts.save_dir, os.path.splitext(os.path.basename(self.opts.image_dir))[0])
        os.makedirs(img_save_dir, exist_ok=True)
        for j in range(len(diffusion_results)):
            image = ((diffusion_results[j]+ 1.0) / 2.0).float()
            image_path = os.path.join(img_save_dir, f"{j:03d}.jpg")
            save_image(image.permute(2, 0, 1), image_path)
        return diffusion_results