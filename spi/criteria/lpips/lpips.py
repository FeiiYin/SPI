from heapq import nsmallest
import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import get_network, LinLayers
from .utils import get_state_dict


class LPIPS(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).
    Arguments:
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    def __init__(self, net_type: str='alex', version: str='0.1', num_scales=1):

        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LPIPS, self).__init__()

        # pretrained network
        self.net = get_network(net_type).to("cuda")

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list).to("cuda")
        self.lin.load_state_dict(get_state_dict(net_type, version))
        self.num_scales = num_scales

    def forward(self, x: torch.Tensor, y: torch.Tensor, conf_sigma=None, mask=None):
        EPS = 1e-7
        loss = 0.0
        n_samples = x.shape[0]

        if x.shape[-1] > 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
            y = F.interpolate(y, size=(256, 256), mode='bilinear', align_corners=False)

        for _scale in range(self.num_scales):
            # import pdb; pdb.set_trace()
            feat_x, feat_y = self.net(x), self.net(y)
            diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]

            if conf_sigma is not None:
                ori_diff = diff
                diff = []
                for di in ori_diff:
                    H = di.shape[-1]
                    _conf_sigma = F.interpolate(conf_sigma, mode='area', size=(H, H))
                    di = di / (2 * _conf_sigma**2 + EPS) + (_conf_sigma + EPS).log()
                    diff.append(di)

            if mask is not None:
                ori_diff = diff
                diff = []
                for di in ori_diff:
                    H = di.shape[-1]
                    _mask = F.interpolate(mask, mode='area', size=(H, H))
                    di = di * _mask
                    diff.append(di)

            res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]
            loss += torch.sum(torch.cat(res, 0))

            if _scale != self.num_scales - 1:
                x = F.interpolate(x, mode='bilinear', scale_factor=0.5, align_corners=False, recompute_scale_factor=True)
                y = F.interpolate(y, mode='bilinear', scale_factor=0.5, align_corners=False, recompute_scale_factor=True)

        return loss / n_samples
