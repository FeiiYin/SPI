import os
import sys
sys.path.append(os.path.abspath('.'))

import eg3d.dnnlib as dnnlib
from eg3d.torch_utils import misc
import eg3d.legacy as legacy
import torch
from eg3d.training.triplane import TriPlaneGenerator
from eg3d.training.dual_discriminator import DualDiscriminator
from spi.configs import paths_config



def load_eg3d(reload_modules=True, device='cuda', network_pkl=None):
    if network_pkl is None:
        network_pkl = paths_config.EG3D_PATH
    
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        # print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new
        
    G.neural_rendering_resolution = 128
    G.eval()
    return G
    

def load_bisenet():
	from third_part.bisenet.bisenet import BiSeNet
	net = BiSeNet(19)
	path = paths_config.BISENET_PATH
	ckpt = torch.load(path, map_location='cpu')
	net.load_state_dict(ckpt)
	net.to('cuda')
	net.eval()
	return net


def load_sg_vgg():
    url = paths_config.VGG_PATH
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval()
    return vgg16