from einops import rearrange
from splatting import splatting_function
import torch

def flow_warp(raft_model, args_raft, ref_depth_est, ref_img, src_img):
    ref_depth_est = ref_depth_est.cpu().data.numpy()
    B, _, H, W = ref_depth_est.shape
    # Calculate flow and importance for splatting.
    output = raft_model(ref_img*255, src_img*255, iters=args_raft.iters, test_mode=True)
    flow = output['flow'][-1]
    flow = flow.reshape(B,2,-1)
    flow = flow.permute(0,2,1)
    flow = flow.cpu().data.numpy()
    new_z = ref_depth_est.reshape(B,-1)
    ## Importance.
    alpha: float = 0.5
    importance = alpha / new_z
    importance = torch.from_numpy(importance)
    importance = importance[...,None]
    importance -= importance.amin((1, 2), keepdim=True)
    importance /= importance.amax((1, 2), keepdim=True) + 1e-6
    importance = importance * 10 - 10
    ## Rearrange.
    importance = rearrange(importance, 'b (h w) c -> b c h w', h=H, w=W)
    # Splatting.
    flow = torch.from_numpy(flow)
    flow = rearrange(flow, 'b (h w) c -> b c h w', h=H, w=W)
    flow = flow.to(ref_img)
    importance = importance.to(ref_img)
    # Splatting.
    warped = splatting_function('softmax', ref_img, flow, importance, eps=1e-6)
    ## mask is 1 where there is no splat
    mask = (warped == 0.).all(dim=1, keepdim=True)
    ## mask is 0 where there is no splat
    mask = ~mask
    return warped, mask.float()