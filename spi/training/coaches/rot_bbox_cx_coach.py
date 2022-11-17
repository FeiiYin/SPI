import os
import torch
from tqdm import tqdm
from spi.configs import paths_config, hyperparameters, global_config
from .base_coach import BaseCoach
from spi.utils.log_utils import log_image_from_w, log_image
from spi.criteria.l2_loss import l2_loss
from spi.criteria.bbox_cx_loss import BoxCXLoss
from spi.utils.rotate import rotate, rotate_with_conffidence
from spi.utils.mask_utils import calculate_face_mask
from spi.utils.camera_utils import cal_mirror_c, cal_camera_weight, sample_surrounding_camera, sample_camera, cal_sequence_c, cal_camera_gauss_weight
from criteria.tv_loss import cal_tv_loss


class RotBboxCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)
        self.coach_name = 'RotBboxCoach'
        self.build_name()
        self.max_pool = torch.nn.MaxPool2d(kernel_size=21, stride=1, padding=10)
        # tensor_dilate = self.max_pool(image)
        # tensor_erode = - self.max_pool(-image)
        self.box_cx_loss = BoxCXLoss().cuda().eval()

    def train(self):
        use_ball_holder = True
        paths_config.experiments_output_dir += f'{self.coach_name}'
        output_dir = paths_config.experiments_output_dir

        for idx, data in tqdm(enumerate(self.data_loader)):
            if self.image_counter >= hyperparameters.max_images_to_invert:
                break
            image_name = data['name'][0]

            image = data['img'].to(global_config.device)
            camera = data['c'].to(global_config.device)
            mask = data['mask'].to(global_config.device)[:, 0]
            fg_mask = 1 - (mask == 0).float()
            face_mask = calculate_face_mask(mask).float()

            fg_mask_m = torch.flip(fg_mask, dims=[3])
            face_mask_m = torch.flip(face_mask, dims=[3])
            camera_m = cal_mirror_c(camera=camera)
            image_m = torch.flip(image, dims=[3])

            paths_config.experiments_output_dir = os.path.join(output_dir, image_name)
            os.makedirs(paths_config.experiments_output_dir, exist_ok=True)

            if self.use_wandb:
                log_image(image, 'target_image')
                log_image(image_m, 'mirror_image')

            self.restart_training()
            w_pivot = self.get_inversion(image_name, image, camera, fg_mask=fg_mask)
                        
            log_images_counter = 0
            target_camera = camera
            target_image = image
            weight_m = cal_camera_weight(camera)
            # Fix yaw range
            if hyperparameters.use_adapt_yaw_range:
                adapt_yaw_range = cal_camera_gauss_weight(camera)[0].item()
            else:
                adapt_yaw_range = 0.2
            # Adapt yaw range
            # 0.2 ~ 0.5 check 0.2 ~ 1 is ok or not
            ws = w_pivot
            rot_bs = 4

            for i in tqdm(range(hyperparameters.G_1_step)):
                self.optimizer.zero_grad()

                gen_results = self.G.synthesis(ws, target_camera, noise_mode='const')
                generated_images = gen_results['image']
                generated_depths = gen_results['image_depth']
                
                loss = 0.0
                if hyperparameters.pt_l2_lambda > 0:
                    l2_loss_val = l2_loss(generated_images, target_image)
                    loss += l2_loss_val * hyperparameters.pt_l2_lambda

                if hyperparameters.pt_lpips_lambda > 0:
                    loss_lpips = self.lpips_loss(generated_images, target_image)
                    loss_lpips = torch.squeeze(loss_lpips)
                    loss += loss_lpips * hyperparameters.pt_lpips_lambda

                # if use_ball_holder and hyperparameters.use_locality_regularization:
                #     ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(self.G, ws, target_camera)
                #     loss += ball_holder_loss_val

                loss.backward()

                if i % rot_bs == 0:
                    if hyperparameters.pt_rot_lambda > 0:
                        new_surrounding_camera = sample_surrounding_camera(camera, batch_size=rot_bs, yaw_range=adapt_yaw_range, pitch_range=0.1)
                        new_ws = w_pivot.repeat(rot_bs, 1, 1)
                        gen_samples = self.G.synthesis(new_ws, new_surrounding_camera, noise_mode='const')

                        with torch.no_grad():
                            warp_img, warp_mask = rotate(target_camera=new_surrounding_camera, target_depth=gen_samples['image_depth'], 
                                        src_image=image.repeat(rot_bs, 1, 1, 1), 
                                        src_camera=camera.repeat(rot_bs, 1), 
                                        src_depth=generated_depths.repeat(rot_bs,1,1,1), 
                                        # src_mask=fg_mask.repeat(rot_bs,1,1,1),
                                        src_mask=face_mask.repeat(rot_bs,1,1,1),
                                        EPS=5e-2)

                        loss_rot = self.lpips_loss(gen_samples['image'] * warp_mask, warp_img)
                        # lm = data['lm'].cuda()
                        # loss_rot = self.box_cx_loss(gen_samples['image'] * warp_mask, warp_img, lm.repeat(rot_bs, 1, 1))
                        loss_rot = loss_rot * hyperparameters.pt_rot_lambda * rot_bs
                        loss_rot.backward()
                        # loss += loss_rot * hyperparameters.pt_rot_lambda * rot_bs

                    if hyperparameters.pt_mirror_rot_lambda > 0 and weight_m > 0:
                        rot_bs_m = rot_bs
                        new_surrounding_camera_m = sample_surrounding_camera(camera_m, batch_size=rot_bs_m, yaw_range=adapt_yaw_range, pitch_range=0.1)
                        new_ws_m = w_pivot.repeat(rot_bs_m, 1, 1)
                        gen_samples_m = self.G.synthesis(new_ws_m, new_surrounding_camera_m, noise_mode='const')

                        # # TODO to set this as a constant
                        # _, warp_mask_m_from_gt = rotate(target_camera=new_surrounding_camera_m, 
                        #             target_depth=gen_samples_m['image_depth'],
                        #             src_image=image.repeat(rot_bs_m, 1, 1, 1), 
                        #             src_camera=camera.repeat(rot_bs_m, 1), 
                        #             src_depth=generated_depths.repeat(rot_bs_m,1,1,1), 
                        #             # src_mask=fg_mask.repeat(rot_bs_m,1,1,1),
                        #             src_mask=face_mask.repeat(rot_bs_m,1,1,1),
                        #             EPS=5e-2)

                        # warp_mask_m_from_gt = - self.max_pool(- warp_mask_m_from_gt)
                        with torch.no_grad():
                            generated_depths_m = torch.flip(generated_depths, dims=[3])
                            warp_img_m, warp_mask_m_from_gtm = rotate(target_camera=new_surrounding_camera_m, 
                                        target_depth=gen_samples_m['image_depth'], 
                                        src_image=image_m.repeat(rot_bs_m, 1, 1, 1), 
                                        src_camera=camera_m.repeat(rot_bs_m, 1), 
                                        src_depth=generated_depths_m.repeat(rot_bs_m,1,1,1), 
                                        src_mask=face_mask_m.repeat(rot_bs_m,1,1,1),
                                        EPS=5e-2)
                            flip_warp_img_m = torch.flip(warp_img_m, dims=[3])
                            flip_warp_mask_m_from_gtm = torch.flip(warp_mask_m_from_gtm, dims=[3])

                        # warp_mask_m = (1 - warp_mask_m_from_gt) * warp_mask_m_from_gtm
                        # warp_img_m = warp_mask_m * warp_img_m
                        # ratio = torch.sum(rest_face) / torch.sum(mirror_face_mask) / (bs - 1)

                        # if hyperparameters.use_cx_loss:
                        #     loss_rot_m = self.cx_loss(gen_samples_m['image'] * warp_mask_m, warp_img_m)
                        # else:
                        #     loss_rot_m = self.lpips_loss(gen_samples_m['image'] * warp_mask_m, warp_img_m)
                        lm = data['lm'].to(global_config.device).repeat(rot_bs, 1, 1)
                        # image_m = self.G.synthesis(w_pivot, camera_m, noise_mode='const')['image']
                        # flip_image_m = torch.flip(image_m, dims=[3])
                        # loss_rot_m = self.box_cx_loss(image, flip_image_m, lm)
                        flip_gen_image = torch.flip(gen_samples_m['image'], dims=[3])
                        loss_rot_m = self.box_cx_loss(flip_gen_image * flip_warp_mask_m_from_gtm, flip_warp_img_m, lm)
                        
                        loss_rot_m = loss_rot_m * hyperparameters.pt_mirror_rot_lambda * rot_bs  # weight_m
                        loss_rot_m.backward()
                        # loss += loss_rot_m * hyperparameters.pt_mirror_rot_lambda * rot_bs  # weight_m

                    if hyperparameters.pt_depth_lambda > 0:
                        new_camera = sample_camera(batch_size=4, yaw_range=0.7, pitch_range=0.4, device=global_config.device)
                        new_ws = w_pivot.repeat(4, 1, 1)
                        sample_depth = self.G.synthesis(new_ws, new_camera, noise_mode='const')['image_depth']
                        with torch.no_grad():
                            stable_depth = self.original_G.synthesis(new_ws, new_camera, noise_mode='const')['image_depth']
                        loss_depth = l2_loss(stable_depth, sample_depth) 
                        loss_depth = loss_depth * hyperparameters.pt_depth_lambda
                        loss_depth.backward()
                        # loss += loss_depth * hyperparameters.pt_depth_lambda

                    if hyperparameters.pt_tv_lambda > 0:
                        loss_tv = cal_tv_loss(w_pivot, self.G)
                        loss_tv = loss_tv * hyperparameters.pt_tv_lambda
                        loss_tv.backward()
                        # loss += tv_loss * hyperparameters.pt_tv_lambda

                if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                    break
                
                self.optimizer.step()

                # use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                if self.use_wandb and log_images_counter % global_config.log_snapshot == 0:
                    log_image_from_w(w_pivot, camera, self.G,  f'{image_name}_G1_inv_{log_images_counter}')
                    # if hyperparameters.use_rot:
                    #     log_image(warp_img[:1], f'{image_name}_warp_img_{log_images_counter}')

                global_config.training_step += 1
                log_images_counter += 1

            self.image_counter += 1

            if self.use_wandb and hyperparameters.G_1_step > 0:
                G1_inv = log_image_from_w(w_pivot, camera, self.G,  f'{image_name}_G1_inv')
                G1_inv_m = log_image_from_w(w_pivot, camera_m, self.G,  f'{image_name}_G1_inv_m')
                self.log_video(w_pivot, self.G, path=os.path.join(paths_config.experiments_output_dir, f'{image_name}_G1_inv.mp4'))
                self.cal_metric(G1_inv, image, 'G1_inv', fake_m=G1_inv_m)
                # torch.save(self.G, f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_{image_name}.pt')

            self.post_process(w_pivot, camera, self.G, image_name)

        paths_config.experiments_output_dir = output_dir
        if self.use_wandb:
            self.log_metric()


