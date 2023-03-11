
from importlib.resources import path
import eg3d.dnnlib as dnnlib
import abc
import os
import os.path
import torch
from torchvision import transforms
from spi.criteria.lpips.lpips import LPIPS
from spi.training.projectors import w_plus_projector, mirror_projector, w_projector
from spi.configs import global_config, paths_config, hyperparameters
from criteria.l2_loss import l2_loss
from spi.utils.log_utils import log_image_from_w, log_image
from spi.utils.load_utils import load_eg3d as load_old_G
import spi.utils.load_utils as load_modules
from spi.utils.metric_utils import Metric
from spi.utils.video_utils import gen_interp_video
from spi.utils.camera_utils import cal_mirror_c
import numpy as np
from PIL import Image


def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def fix_seed():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BaseCoach:
    def __init__(self, data_loader, use_wandb):

        self.use_wandb = use_wandb
        self.data_loader = data_loader
        self.w_pivots = {}
        self.image_counter = 0
        self.metric = Metric()
        self.metric_dic = {}
        self.coach_name = 'Base_coach'

        # Initialize loss
        self.lpips_loss = LPIPS(net_type='vgg').to(global_config.device).eval()
        
        self.restart_training()
        self.vgg16 = load_modules.load_sg_vgg().to(global_config.device)

    def restart_training(self):
        self.G = load_old_G()
        toogle_grad(self.G, True)

        self.original_G = load_old_G()
        self.optimizer = self.configure_optimizers()
        
        fix_seed()

    def get_inversion(self, image_name, image, camera, fg_mask=None):
        embedding_dir = f'{paths_config.embedding_base_dir}/{self.coach_name}/'
        os.makedirs(embedding_dir, exist_ok=True)

        w_pivot = None
        if hyperparameters.load_embedding_coach_name is not None:
            load_embedding_dir = f'{paths_config.embedding_base_dir}/{hyperparameters.load_embedding_coach_name}/'
            w_pivot = self.load_inversions(load_embedding_dir, image_name)

        if w_pivot is None:
            w_pivot = self.calc_inversions(image_name, image, camera, fg_mask)

        torch.save(w_pivot, f'{embedding_dir}/{image_name}.pt')
        w_pivot = w_pivot.to(global_config.device)

        if self.use_wandb:
            camera_m = cal_mirror_c(camera)
            w_inv = log_image_from_w(w_pivot, camera, self.G, f'{image_name}_w_inv')
            w_inv_m = log_image_from_w(w_pivot, camera_m, self.G, f'{image_name}_w_inv_m')

            self.log_video(w_pivot, self.G, os.path.join(paths_config.experiments_output_dir, f'{image_name}_w_inv.mp4'))
            self.cal_metric(w_inv, image, 'w_inv', fake_m=w_inv_m)

        return w_pivot

    def load_inversions(self, embedding_dir, image_name):
        if image_name in self.w_pivots:
            return self.w_pivots[image_name]

        w_potential_path = f'{embedding_dir}/{image_name}.pt'

        if not os.path.isfile(w_potential_path):
            print('[ERROR]: No existing w codes.')
            return None
        
        w = torch.load(w_potential_path, map_location='cpu').to(global_config.device)
        self.w_pivots[image_name] = w
        return w

    def calc_inversions(self, image_name, image, camera, fg_mask=None):
        initial_w = None
        assert hyperparameters.first_inv_type in ['sg', 'sgw+', 'mir', 'reg']
        if hyperparameters.first_inv_type == 'sg':
            w = w_projector.project(self.G, image, camera, vgg16=self.vgg16,
                                     device=torch.device(global_config.device), 
                                    w_avg_samples=600, num_steps=hyperparameters.first_inv_steps, 
                                    verbose=self.use_wandb,
                                    w_name=image_name, initial_w=initial_w)
        elif hyperparameters.first_inv_type == 'sgw+':
            w = w_plus_projector.project(self.G, image, camera, lpips_func=self.lpips_loss,
                                     device=torch.device(global_config.device), 
                                        w_avg_samples=600, num_steps=hyperparameters.first_inv_steps, 
                                        verbose=self.use_wandb,
                                        w_name=image_name, initial_w=initial_w)
        elif hyperparameters.first_inv_type == 'mir':
            w = mirror_projector.project(self.G, image, camera, lpips_func=self.lpips_loss,
                                    fg_mask=fg_mask,
                                     device=torch.device(global_config.device), 
                                        w_avg_samples=600, num_steps=hyperparameters.first_inv_steps, 
                                        verbose=self.use_wandb,
                                        w_name=image_name, initial_w=initial_w)
        else:
            raise NotImplementedError
    
        return w

    @abc.abstractmethod
    def train(self):
        pass

    def configure_optimizers(self):
        print('[Note]: Fintune only All parameters')
        optimizer = torch.optim.Adam(self.G.parameters(), lr=hyperparameters.pti_learning_rate)
        return optimizer

    @abc.abstractmethod
    def calc_loss(self):
        return

    def cal_metric(self, fake, gt, name, fake_m=None):
        if name not in self.metric_dic:
            self.metric_dic[name] = {'l2': [], 'lpips': [], 'id': [], 'l2_m': [], 'lpips_m': [], 'id_m': []}

        l2, lpips, id_sim = self.metric.run(gt, fake)
        self.metric_dic[name]['l2'].append(l2)
        self.metric_dic[name]['lpips'].append(lpips)
        self.metric_dic[name]['id'].append(id_sim)

        if fake_m is not None:
            l2, lpips, id_sim = self.metric.run(torch.flip(gt, dims=[3]), fake_m)
            self.metric_dic[name]['l2_m'].append(l2)
            self.metric_dic[name]['lpips_m'].append(lpips)
            self.metric_dic[name]['id_m'].append(id_sim)

    def log_metric(self):
        with open(os.path.join(paths_config.experiments_output_dir, 'metric_log.txt'), "a") as log_file:
            message = f'Coach name: {self.coach_name}\n'
            message += f'hyperparameters.use_encoder: {hyperparameters.use_encoder}\n'
            message += f'hyperparameters.first_inv_type: {hyperparameters.first_inv_type}\n'
            message += f'hyperparameters.first_inv_steps: {hyperparameters.first_inv_steps}\n'
            message += f'hyperparameters.G_1_step: {hyperparameters.G_1_step}\n'
            message += f'hyperparameters.G_2_step: {hyperparameters.G_2_step}\n'
            log_file.write(message)
            log_file.write('\n')

            for key in self.metric_dic:
                message = f'Mode: {key}\n'
                l2_t, lp_t, lid_t, l2_m_t, lp_m_t, lid_m_t = 0,0,0,0,0,0
                current_dic = self.metric_dic[key]
                cnt = len(current_dic['l2'])
                for _i in range(cnt):
                    l2 = current_dic['l2'][_i]
                    lp = current_dic['lpips'][_i]
                    lid = current_dic['id'][_i]
                    l2_m = current_dic['l2_m'][_i]
                    lp_m = current_dic['lpips_m'][_i]
                    lid_m = current_dic['id_m'][_i]
                    message += f'ID: {_i} L2: {l2:.6f}; Lpips: {lp:.6f}; ID Sim: {lid:.6f}; L2 M: {l2_m:.6f}; Lpips M: {lp_m:.6f}; ID Sim M: {lid_m:.6f};\n'

                    l2_t += l2
                    lp_t += lp
                    lid_t += lid
                    l2_m_t += l2_m
                    lp_m_t += lp_m
                    lid_m_t += lid_m
                
                message += f'Mode: {key} AVG\n'
                l2_t /= cnt
                lp_t /= cnt
                lid_t /= cnt
                l2_m_t /= cnt
                lp_m_t /= cnt
                lid_m_t /= cnt
                message += f'L2: {l2_t:.6f}; Lpips: {lp_t:.6f}; ID Sim: {lid_t:.6f}; L2 M: {l2_m_t:.6f}; Lpips M: {lp_m_t:.6f}; ID Sim M: {lid_m_t:.6f};\n'
            
                log_file.write(message)
                log_file.write('\n')

    def save_latent(self, w, name):
        save_path = os.path.join(paths_config.embedding_base_dir, self.coach_name, f'{name}.pt')
        torch.save(w, save_path)

    def save(self, w, c, G, path):
        ckpt = {
            'w': w.detach().cpu(),
            'c': c.detach().cpu(),
            'G': G.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        ckpt = torch.load(path)
        self.G.load_state_dict(ckpt['G'])
        w = ckpt['w'].to(global_config.device)
        c = ckpt['c'].to(global_config.device)
        return w, c, self.G

    def post_process(self, w, c, G, name):
        self.save(w, c, G, path=os.path.join(paths_config.checkpoints_dir, self.coach_name, f'{name}.pt'))
        
        self.log_image(w, c, G, path=os.path.join(paths_config.images_output_dir, self.coach_name, name + '.jpg'))
        c_m = cal_mirror_c(c)
        self.log_image(w, c_m, G, path=os.path.join(paths_config.mirror_images_output_dir, self.coach_name, name + '.jpg'))

        self.log_video(w, G, path=os.path.join(paths_config.video_output_dir, self.coach_name, f'{name}.mp4'))

    def log_image(self, w, c, G, path):
        if len(w.size()) <= 2:
            w = w.unsqueeze(0)
        with torch.no_grad():
            img_tensor = G.synthesis(w, c, noise_mode='const')['image']
            img = img_tensor[0].permute(1, 2, 0)
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
        Image.fromarray(img).save(path)

    def log_video(self, w, G, path):
        gen_interp_video(G, {'w': w.detach().clone()}, mp4=path)

    def build_name(self):
        self.coach_name += '_' + hyperparameters.first_inv_type
        self.coach_name += '_' + str(hyperparameters.first_inv_steps)

        self.coach_name += '_' + hyperparameters.G_1_type
        self.coach_name += '_' + str(hyperparameters.G_1_step)
        
        if hyperparameters.use_encoder:
            self.coach_name += '_wenc'
        if hyperparameters.use_G_avg:
            self.coach_name += '_wgavg'
        
        self.coach_name += f'_rot_{hyperparameters.pt_rot_lambda}'
        self.coach_name += f'_mirrorrot_{hyperparameters.pt_mirror_rot_lambda}'
        self.coach_name += f'_depth_{hyperparameters.pt_depth_lambda}'
        self.coach_name += f'_tv_{hyperparameters.pt_tv_lambda}'
    
        if hyperparameters.use_adapt_yaw_range:
            self.coach_name += '_wadyaw'

        if hyperparameters.description is not None:
            self.coach_name += '_' + hyperparameters.description
        

        print('[COACH]:', self.coach_name)
        os.makedirs(os.path.join(paths_config.checkpoints_dir, self.coach_name), exist_ok=True)
        os.makedirs(os.path.join(paths_config.embedding_base_dir, self.coach_name), exist_ok=True)
        os.makedirs(os.path.join(paths_config.experiments_output_dir, self.coach_name), exist_ok=True)
        os.makedirs(os.path.join(paths_config.images_output_dir, self.coach_name), exist_ok=True)
        os.makedirs(os.path.join(paths_config.mirror_images_output_dir, self.coach_name), exist_ok=True)
        os.makedirs(os.path.join(paths_config.video_output_dir, self.coach_name), exist_ok=True)