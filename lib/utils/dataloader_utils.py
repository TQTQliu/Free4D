from random import randint
from torch.utils.data import DataLoader

def load_loader(loader, viewpoint_stack, batch_size, random_loader=True):
    try:
        viewpoint_cams = next(loader)
    except StopIteration:
        print("reset dataloader into random dataloader.")
        if not random_loader:
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=32,collate_fn=list)
            random_loader = True
        loader = iter(viewpoint_stack_loader)
    return viewpoint_cams

def load_memory(temp_list, viewpoint_stack, batch_size):
    idx = 0
    viewpoint_cams = []
    while idx < batch_size:
        viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))
        if not viewpoint_stack:
            viewpoint_stack =  temp_list.copy()
        viewpoint_cams.append(viewpoint_cam)
        idx +=1
    return viewpoint_cams, viewpoint_stack

def load_test_memory(test_temp_list, test_viewpoint_stack, batch_size, vd_len):
    idx = 0
    test_viewpoint_cams = []
    while idx < batch_size :    
        viewpoint_cam = test_viewpoint_stack.pop(randint(0,len(test_viewpoint_stack)-1))
        if not test_viewpoint_stack :
            test_viewpoint_stack =  test_temp_list.copy()
        time_idx = viewpoint_cam.uid % vd_len
        cam_idx = viewpoint_cam.uid // vd_len
        if time_idx==0 or cam_idx==0:
            continue
        test_viewpoint_cams.append(viewpoint_cam)
        idx +=1
    return test_viewpoint_cams, test_viewpoint_stack