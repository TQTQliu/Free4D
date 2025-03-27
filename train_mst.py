#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import sys
sys.path.append('lib')
import torch
import lpips
from tqdm import tqdm
from lib.gaussian_renderer import render, network_gui
from lib.scene import Scene, GaussianModel
from lib.utils.image_utils import psnr
from lib.utils.log_utils import prepare_output_and_logger, training_report
from lib.utils.dataloader_utils import *
from lib.utils.general_utils import safe_state, setup_seed, Timer
from lib.utils.loss_utils import l1_loss, lpips_loss
from lib.guidance.mcs import *
from configs.arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from argparse import ArgumentParser

def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint,
                         gaussians, scene, stage, tb_writer, train_iter, timer, refine_iteration=None):
    first_iter = 0
    gaussians.training_setup(opt)
    if checkpoint:
        if stage == "3dgs" and stage not in checkpoint:
            print("start from 4dgs stage, skip 3dgs stage.")
            return
        if stage in checkpoint: 
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)
        
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    final_iter = train_iter
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    batch_size = opt.batch_size
    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=16,collate_fn=list)
        loader = iter(viewpoint_stack_loader)
        
    if stage == "3dgs" and opt.zerostamp_init:
        images_folder = dataset.source_path
        vd_len = len(os.listdir(os.path.join(images_folder, 'cam01')))
        entries = os.listdir(images_folder)
        cam_folders = [entry for entry in entries 
                   if os.path.isdir(os.path.join(images_folder, entry)) and entry.startswith("cam")]
        mv_len = len(cam_folders)
        if mv_len+vd_len-1 == len(viewpoint_stack):
            temp_list = [viewpoint_stack[0]] + [viewpoint_stack[vd_len+k] for k in range(0, mv_len-1)]
        else:
            temp_list = [viewpoint_stack[k*vd_len] for k in range(0, mv_len)]
    else:
        temp_list = [viewpoint_stack[k] for k in range(len(viewpoint_stack))]
    viewpoint_stack = temp_list.copy()
    load_in_memory = True
        
    for iteration in range(first_iter, final_iter+1):
        if stage=='4dgs' and iteration==refine_iteration:
            lpips_model = lpips.LPIPS(net="alex").cuda()
            denoise_steps = 5
            refiner = HackSD_MCS(device='cuda',use_lcm=True,denoise_steps=denoise_steps,
                                sd_ckpt='stable-diffusion-v1-5/stable-diffusion-v1-5',
                                lcm_ckpt=args.lcm_ckpt)
            rgb_prompt_latent = refiner.model._encode_text_prompt()
            if opt.dataloader:
                test_viewpoint_stack = scene.getTestCameras()
                test_temp_list = [test_viewpoint_stack[k] for k in range(len(test_viewpoint_stack))]
                test_viewpoint_stack = test_temp_list.copy()
                                     
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        # Pick a random Camera
        if opt.dataloader and not load_in_memory:
            viewpoint_cams = load_loader(loader, viewpoint_stack, batch_size)
        else:
            viewpoint_cams, viewpoint_stack = load_memory(temp_list, viewpoint_stack, batch_size)
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage,cam_type=scene.dataset_type)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            images.append(image.unsqueeze(0))
            gt_image = viewpoint_cam.original_image.cuda()
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)

        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])
        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        loss = Ll1
        
        # fine stage of 4dgs optimization
        if stage == '4dgs' and iteration >= refine_iteration:
            images_folder = dataset.source_path
            vd_len = len(os.listdir(os.path.join(images_folder, 'cam01')))
            test_viewpoint_cams, test_viewpoint_stack = load_test_memory(test_temp_list, test_viewpoint_stack, batch_size, vd_len)
            render_image = []
            vc_image = []
            for viewpoint_cam in test_viewpoint_cams:
                render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage, cam_type=scene.dataset_type)
                image = render_pkg["render"]
                render_image.append(image)
                gt_image = viewpoint_cam.original_image.cuda()
                vc_image.append(gt_image)
            render_image = torch.stack(render_image)
            vc_image = torch.stack(vc_image)
            refiner._encode_mv_init_images(render_image)
            rect_w_ls = [0.5,0.4,0.3,0.2,0.1]
            for step in range(denoise_steps):
                rgb_t, rgb_noise_pr, rgb_denoise = step_gaussian_optimization(step, refiner, rgb_prompt_latent, denoise_steps)
                # rectification
                refiner._step_denoise(rgb_t,rgb_noise_pr,vc_image,rect_w=rect_w_ls[step]) 
            refine_image = rgb_denoise.permute(0, 3, 1, 2)
            refiner._reset()
            vc_loss = lpips_loss(render_image,refine_image,lpips_model)*0.1
            loss += vc_loss
          
        if stage == "4dgs" and hyper.time_smoothness_weight != 0:
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
            loss += tv_loss
 
        loss.backward()
        if torch.isnan(loss).any():
            print("loss is nan, end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        if iteration < opt.iterations:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)
        if (iteration in checkpoint_iterations):
            print("\n[ITER {}] Saving Checkpoint".format(iteration))
            torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" +f"_{stage}_" + str(iteration) + ".pth")
        if (iteration in saving_iterations):
            print("\n[ITER {}] Saving Gaussians".format(iteration))
            scene.save(iteration, stage)
                
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "psnr": f"{psnr_:.{2}f}", "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            # Log and save
            timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage, scene.dataset_type)
            timer.start()
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)
                if stage == "3dgs":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter)  
                if  iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)
                if  iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0]>200000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000 and opt.add_point:
                    gaussians.grow(5,5,scene.model_path,iteration,stage)
                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()
                           
def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, expname):
    tb_writer = prepare_output_and_logger(args, expname, __file__)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians)
    timer.start()
    # canonical 3dgs
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                        checkpoint_iterations, checkpoint,
                        gaussians, scene, "3dgs", tb_writer, opt.iterations_3d, timer)
    # 4dgs
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                        checkpoint_iterations, checkpoint,
                        gaussians, scene, "4dgs", tb_writer, opt.iterations, timer, refine_iteration=opt.refine_iteration)

if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "configs/arguments/multipleview/default.py")
    parser.add_argument("--lcm_ckpt", type=str, default = "ckpt/lcm/pytorch_lora_weights.safetensors")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from lib.utils.general_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.expname)

    # All done
    print("Training complete.")