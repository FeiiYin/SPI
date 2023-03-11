from spi.criteria.l2_loss import l2_loss
from spi.criteria.lpips.lpips import LPIPS
from spi.criteria.id_loss.id_loss import IDLoss
from spi.configs import paths_config

class Metric():

    def __init__(self) -> None:
        self.lpips_loss = LPIPS(net_type='vgg').to('cuda').eval()
        path_ir_se50 = paths_config.IDLOSS_PATH
        self.id_loss = IDLoss(path_ir_se50).to('cuda').eval()

    def run(self, gt, fake):
        l2 = l2_loss(gt, fake)
        lpips = self.lpips_loss(gt, fake)
        id_sim = self.id_loss.calculate_similarity(gt, fake)
        return l2.item(), lpips.item(), id_sim.item()



def metric(gt, w, c, G, lpips_func):
    fake = G.synthesis(w, c, noise_mode='const')['image']

    l2 = l2_loss(gt, fake)
    lpips = lpips_func(gt, fake)
    return l2, lpips

