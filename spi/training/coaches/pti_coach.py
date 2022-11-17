import os
import torch
from tqdm import tqdm
from spi.utils.camera_utils import cal_mirror_c
from spi.configs import paths_config, hyperparameters, global_config
from .base_coach import BaseCoach
from spi.utils.log_utils import log_image_from_w, log_image
from spi.criteria.l2_loss import l2_loss



class SingleIDCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)
        self.coach_name = 'PTI_coach'
        self.build_name()

    def calc_loss(self, x, gt, use_ball_holder, new_G, w, c):
        loss = 0.0
        if hyperparameters.pt_l2_lambda > 0:
            l2_loss_val = l2_loss(x, gt)
            loss += l2_loss_val * hyperparameters.pt_l2_lambda
        if hyperparameters.pt_lpips_lambda > 0:
            loss_lpips = self.lpips_loss(x, gt)
            loss_lpips = torch.squeeze(loss_lpips)
            loss += loss_lpips * hyperparameters.pt_lpips_lambda

        if use_ball_holder and hyperparameters.use_locality_regularization:
            ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(new_G, w, c)
            loss += ball_holder_loss_val
        return loss, loss_lpips

    def train(self):
        use_ball_holder = True
        paths_config.experiments_output_dir += f'{self.coach_name}'
        output_dir = paths_config.experiments_output_dir

        for idx, data in tqdm(enumerate(self.data_loader)):
            if self.image_counter >= hyperparameters.max_images_to_invert:
                break
            image_name = data['name'][0]
            # if image_name not in ['chris_evans']:
            #     continue

            image = data['img'].to(global_config.device)
            camera = data['c'].to(global_config.device)
            mask = data['mask'].to(global_config.device)[:, 0]
            fg_mask = 1 - (mask == 0).float()

            paths_config.experiments_output_dir = os.path.join(output_dir, image_name)
            os.makedirs(paths_config.experiments_output_dir, exist_ok=True)

            if self.use_wandb:
                log_image(image, 'target_image')

            self.restart_training()

            w_pivot = self.get_inversion(image_name, image, camera, fg_mask=fg_mask)
            
            log_images_counter = 0
            real_images_batch = image.to(global_config.device)

            for i in tqdm(range(hyperparameters.G_1_step)):
                generated_images = self.G.synthesis(w_pivot, camera, noise_mode='const')['image']
                target_images = real_images_batch

                loss, loss_lpips = self.calc_loss(generated_images, target_images, image_name, self.G, use_ball_holder, w_pivot)

                self.optimizer.zero_grad()

                if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                    break

                loss.backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                if self.use_wandb and log_images_counter % global_config.log_snapshot == 0:
                    log_image_from_w(w_pivot, camera, self.G,  f'{image_name}_G1_inv_{log_images_counter}')

                global_config.training_step += 1
                log_images_counter += 1

            self.image_counter += 1

            if self.use_wandb and hyperparameters.G_1_step > 0:
                camera_m = cal_mirror_c(camera=camera)
                G1_inv = log_image_from_w(w_pivot, camera, self.G,  f'{image_name}_G1_inv')
                G1_inv_m = log_image_from_w(w_pivot, camera_m, self.G,  f'{image_name}_G1_inv_m')
                self.log_video(w_pivot, self.G, path=os.path.join(paths_config.experiments_output_dir, f'{image_name}_G1_inv.mp4'))
                
                self.cal_metric(G1_inv, image, 'G1_inv', fake_m=G1_inv_m)
                # torch.save(self.G, f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_{image_name}.pt')
            
            self.post_process(w_pivot, camera, self.G, image_name)
        
        paths_config.experiments_output_dir = output_dir
        if self.use_wandb:
            self.log_metric()
