import os
import sys
sys.path.append('./')
sys.path.append('./lib/viewcrafter')
from configs.vc_configs.infer_config import get_parser
from utils_vc.pvd_utils import *
from datetime import datetime

if __name__=="__main__":
    parser = get_parser()
    opts = parser.parse_args()
    if opts.exp_name == None:
        prefix = datetime.now().strftime("%Y%m%d_%H%M")
        opts.exp_name = f'{prefix}_{os.path.splitext(os.path.basename(opts.image_dir))[0]}'
    opts.save_dir = os.path.join(opts.out_dir,opts.exp_name)
    os.makedirs(opts.save_dir, exist_ok=True)
    opts.output_dir = opts.save_dir
    
    if opts.recon == 'monst3r':
        from viewcrafter_monst3r import ViewCrafter
        pvd = ViewCrafter(opts)
    elif opts.recon == 'dust3r':
        from viewcrafter_dust3r import ViewCrafter
        pvd = ViewCrafter(opts)
    else:
        raise NotImplementedError
    
    if opts.mode == 'single_view_txt':
        pvd.nvs_single_view()
    else:
        raise KeyError(f"Invalid Mode: {opts.mode}")