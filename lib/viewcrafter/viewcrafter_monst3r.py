import sys
sys.path.append('./lib/viewcrafter/extern/')
from monst3r.dust3r.inference import inference
from monst3r.dust3r.utils.image import load_images, enlarge_seg_masks
from monst3r.dust3r.image_pairs import make_pairs
from monst3r.dust3r.cloud_opt import global_aligner, GlobalAlignerMode, get_3D_model_from_scene
from monst3r.dust3r.utils.device import to_numpy
from monst3r.dust3r.model import AsymmetricCroCo3DStereo
import shutil
import torch
import numpy as np
import os
import cv2  
import glob
from PIL import Image
from pytorch3d.structures import Pointclouds
from torchvision.utils import save_image
import torch.nn.functional as F
from PIL import Image
from utils_vc.pvd_utils import *
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from utils_vc.diffusion_utils import instantiate_from_config,load_model_checkpoint,image_guided_synthesis,get_latent_z
from torchvision.utils import save_image
import functools
from tqdm import tqdm

class ViewCrafter:
    def __init__(self, opts):
        self.opts = opts
        self.device = opts.device
        self.setup_monst3r()
        self.setup_diffusion()
        # initialize ref images, pcd
        if os.path.isdir(self.opts.image_dir):
            resize_images(self.opts.image_dir)
            self.images, self.img_ori = self.load_initial_dir(image_dir=self.opts.image_dir)
            self.run_monst3r(input_images=self.images) 
        else:
            print(f"{self.opts.image_dir} doesn't exist")           

    def run_monst3r(self, input_images):
        recon_fun = functools.partial(self.get_reconstructed_scene, self.opts, self.monst3r, self.opts.device, self.opts.silent)
        # Call the function with default parameters
        self.scene = recon_fun(
            input_images=input_images,
            schedule='linear',
            niter=300,
            min_conf_thr=1.1,
            as_pointcloud=True,
            mask_sky=False,
            clean_depth=True,
            transparent_cams=False,
            cam_size=0.05,
            show_cam=True,
            scenegraph_type='swinstride',
            winsize=5,
            refid=0,
            seq_name=self.opts.seq_name,
            temporal_smoothing_weight=0.01,
            translation_weight=1.0,
            shared_focal=True,
            flow_loss_weight=0.01,
            flow_loss_start_iter=0.1,
            flow_loss_threshold=25,
            use_gt_mask=self.opts.use_gt_davis_masks,
        )
    
    def get_reconstructed_scene(self, args, model, device, silent, input_images, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, show_cam, scenegraph_type, winsize, refid, 
                            seq_name, temporal_smoothing_weight, translation_weight, shared_focal, 
                            flow_loss_weight, flow_loss_start_iter, flow_loss_threshold, use_gt_mask):
        """
        from a list of images, run monst3r inference, global aligner.
        then run get_3D_model_from_scene
        """
        if scenegraph_type == "swin" or scenegraph_type == "swinstride" or scenegraph_type == "swin2stride":
            scenegraph_type = scenegraph_type + "-" + str(winsize) + "-noncyclic"
        elif scenegraph_type == "oneref":
            scenegraph_type = scenegraph_type + "-" + str(refid)
        pairs = make_pairs(input_images, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=self.opts.batch_size)

        mode = GlobalAlignerMode.PointCloudOptimizer #if len(self.images) > 2 else GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=device, mode=mode, verbose=not silent, shared_focal = shared_focal, temporal_smoothing_weight=temporal_smoothing_weight, translation_weight=translation_weight,
                        flow_loss_weight=flow_loss_weight, flow_loss_start_epoch=flow_loss_start_iter, flow_loss_thre=flow_loss_threshold, use_self_mask=not use_gt_mask,
                        num_total_iter=niter, empty_cache=True, batchify=not self.opts.not_batchify)
        if mode == GlobalAlignerMode.PointCloudOptimizer:
            schedule='linear'
            lr=0.01
            loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)
        
        save_folder = f'{args.output_dir}/{seq_name}'
        os.makedirs(save_folder, exist_ok=True)
        
        outfile = get_3D_model_from_scene(save_folder, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                clean_depth, transparent_cams, cam_size, show_cam)
        poses = scene.save_tum_poses(f'{save_folder}/pred_traj.txt')
        K = scene.save_intrinsics(f'{save_folder}/pred_intrinsics.txt')
        depth_maps = scene.save_depth_maps(save_folder)
        dynamic_masks = scene.save_dynamic_masks(save_folder)
        conf = scene.save_conf_maps(save_folder)
        init_conf = scene.save_init_conf_maps(save_folder)
        rgbs = scene.save_rgb_imgs(save_folder)
        enlarge_seg_masks(save_folder, kernel_size=5 if use_gt_mask else 3) 
        return scene

    def render_pcd(self,pts3d,imgs,masks,views,renderer,device,nbv=False,fg_masks=None,index=None, bg_points=None, bg_point_colors=None):
        imgs = to_numpy(imgs)
        pts3d = to_numpy(pts3d)
        pts = torch.from_numpy(np.concatenate((bg_points,pts3d[index][fg_masks[index]]))).to(device)
        col = torch.from_numpy(np.concatenate((bg_point_colors,imgs[index][fg_masks[index]]))).to(device)
        point_cloud = Pointclouds(points=[pts], features=[col]).extend(views)
        images = renderer(point_cloud)
        if nbv:
            color_mask = torch.ones(col.shape).to(device)
            point_cloud_mask = Pointclouds(points=[pts],features=[color_mask]).extend(views)
            view_masks = renderer(point_cloud_mask)
        else: 
            view_masks = None
        return images, view_masks
    
    def run_render(self, pcd, imgs,masks, H, W, camera_traj, num_views, nbv=True, fg_masks=None, index=None, bg_points=None, bg_point_colors=None):
        render_setup = setup_renderer(camera_traj, image_size=(H,W))
        renderer = render_setup['renderer']
        render_results, viewmask = self.render_pcd(pcd, imgs, masks, num_views, renderer, self.device, nbv=nbv, fg_masks=fg_masks, index=index, bg_points=bg_points, bg_point_colors=bg_point_colors)
        return render_results, viewmask

    def run_diffusion(self, renderings, x0_ref=None, replace_mask=None, viewmask=None):
        prompts = [self.opts.prompt]
        videos = (renderings * 2. - 1.).permute(3,0,1,2).unsqueeze(0).to(self.device)
        condition_index = [0]
        with torch.no_grad(), torch.cuda.amp.autocast():
            batch_samples = image_guided_synthesis(self.diffusion, prompts, videos, self.noise_shape, self.opts.n_samples, self.opts.ddim_steps, self.opts.ddim_eta, \
                               self.opts.unconditional_guidance_scale, self.opts.cfg_img, self.opts.frame_stride, self.opts.text_input, self.opts.multiple_cond_cfg, self.opts.timestep_spacing, self.opts.guidance_rescale, condition_index, \
                               x0_ref=x0_ref, replace_mask=replace_mask, viewmask=viewmask)
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

    def setup_monst3r(self):
        self.monst3r = AsymmetricCroCo3DStereo.from_pretrained(self.opts.model_path).to(self.device)
        self.monst3r.eval()
    
    def load_initial_dir(self, image_dir):
        image_files = glob.glob(os.path.join(image_dir, "*"))
        if len(image_files) < 2:
            raise ValueError("Input views should not less than 2.")
        image_files = sorted(image_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        images = load_images(image_files, size=512, force_1024=True)
        img_gts = []
        for i in range(len(image_files)):
            img_gts.append((images[i]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2.) 
        return images, img_gts

    def nvs_single_view(self):
        c2ws = self.scene.get_im_poses().detach() 
        principal_points = self.scene.get_principal_points().detach() #cx cy
        focals = self.scene.get_focals().detach()
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)] # a list of points of size whc
        depth = [i.detach() for i in self.scene.get_depthmaps()]
        depth_avg = depth[0][H//2,W//2]
        radius = depth_avg*self.opts.center_scale

        ## change coordinate
        c2ws, pcd = world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=0, r=radius, elevation=self.opts.elevation, device=self.device)

        imgs = np.array(self.scene.imgs)
        
        masks = None
        if self.opts.mode == 'single_view_txt':
            with open(self.opts.traj_txt, 'r') as file:
                lines = file.readlines()
                phi = [float(i) for i in lines[0].split()]
                theta = [float(i) for i in lines[1].split()]
                r = [float(i) for i in lines[2].split()]
                shutil.copy(self.opts.traj_txt, self.opts.save_dir)
            camera_traj, num_views = generate_traj_txt(c2ws[:1], H, W, focals[:1], principal_points[:1], phi, theta, r,self.opts.video_length, self.device,viz_traj=True, save_dir = self.opts.save_dir)
        else:
            raise KeyError(f"Invalid Mode: {self.opts.mode}")
        
        del self.monst3r
        
        fg_masks=[]
        bg_masks=[]
        monst3r_dir = os.path.join(self.opts.save_dir, 'monst3r_out')
        for idx in range(len(pcd)):
            mask_dynamic = np.array(Image.open(f'{monst3r_dir}/dynamic_mask_{idx}.png'))
            mask_dynamic = (mask_dynamic/255).astype(np.bool_)
            fg_masks.append(mask_dynamic)
            bg_masks.append(~mask_dynamic)
            cv2.imwrite(f'{monst3r_dir}/fg_masks_{idx}.png', ((mask_dynamic) * 255).astype(np.uint8))
            cv2.imwrite(f'{monst3r_dir}/bg_masks_{idx}.png', ((~mask_dynamic) * 255).astype(np.uint8))
        
        # # direct concat
        # bg_points = np.concatenate([p[m] for p, m in zip(to_numpy(pcd), bg_masks)])
        # bg_point_colors = np.concatenate([p[m] for p, m in zip(to_numpy(imgs), bg_masks)])
        
        ## progressive aggregation
        stacked_point = np.stack(to_numpy(pcd))
        stacked_point_colors = np.stack(to_numpy(imgs))
        stacked_masks = np.stack(bg_masks)
        N, H, W, _ = stacked_point.shape
        fused_point = np.zeros((H, W, 3), dtype=np.float32)
        fused_point_colors = np.zeros((H, W, 3), dtype=np.float32)
        fused_mask = np.zeros((H, W), dtype=np.float32)
        for i in range(H):
            for j in range(W):
                valid_indices = np.nonzero(stacked_masks[:, i, j])[0]
                if len(valid_indices) > 0:
                    valid_points = stacked_point[valid_indices, i, j]
                    valid_colors = stacked_point_colors[valid_indices, i, j]
                    fused_point[i, j] = np.mean(valid_points, axis=0)
                    fused_point_colors[i, j] = np.mean(valid_colors, axis=0)
                    fused_mask[i, j] = 1
        bg_points = fused_point[fused_mask == 1]
        bg_point_colors = fused_point_colors[fused_mask == 1]
        
        img_save_dir = os.path.join(self.opts.save_dir, os.path.splitext(os.path.basename(self.opts.image_dir))[0])
        os.makedirs(img_save_dir, exist_ok=True)
        x0_ref=None
        ref_mask=None
        replace_mask=None
        for idx in tqdm(range(len(pcd))):
            if idx>0:
                self.opts.unconditional_guidance_scale = 1
            render_results, viewmask = self.run_render(pcd, imgs, masks, H, W, camera_traj, num_views, fg_masks=fg_masks, index=idx, bg_points=bg_points, bg_point_colors=bg_point_colors)
            viewmask = F.interpolate(viewmask.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
            render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
            render_results[0] = self.img_ori[idx]
            if self.opts.mode == 'single_view_txt':
                if phi[-1]==0. and theta[-1]==0. and r[-1]==0.:
                    render_results[-1] = self.img_ori[idx]
            save_video(render_results, os.path.join(self.opts.save_dir, f'render{idx}.mp4'))
            save_video(viewmask, os.path.join(self.opts.save_dir, f'mask{idx}.mp4'))
            # save_pointcloud_with_normals(imgs[idx:idx+1], pcd[idx:idx+1], msk=None, save_path=os.path.join(self.opts.save_dir,f'pcd{idx}.ply') , mask_pc=False, reduce_pc=False)
            if ref_mask is not None:
                replace_mask = ((ref_mask<0.1).float() + (viewmask<0.1).float())
                replace_mask = (replace_mask==2).float() # 25 h w 3
            diffusion_results = self.run_diffusion(render_results, x0_ref=x0_ref, replace_mask=replace_mask, viewmask=viewmask)
            save_video((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, f'diffusion{idx}.mp4'))
            sub_dir = os.path.join(img_save_dir, f'{idx:03d}')
            os.makedirs(sub_dir, exist_ok=True)
            for j in range(len(diffusion_results)):
                image = ((diffusion_results[j]+ 1.0) / 2.0).float()
                image_path = os.path.join(sub_dir, f"{j:03d}.jpg")
                save_image(image.permute(2, 0, 1), image_path)
            if idx==0:
                videos_input = diffusion_results.permute(3,0,1,2).unsqueeze(0).float()
                x0_ref = get_latent_z(self.diffusion, videos_input) # b c t h w
                ref_mask = viewmask
            
        return diffusion_results