import os
import torch
from tqdm import tqdm
from spi.configs import paths_config, hyperparameters, global_config
from .base_coach import BaseCoach
from spi.utils.mask_utils import calculate_face_mask
from spi.utils.camera_utils import cal_mirror_c


class InferenceCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)
        self.coach_name = 'InferenceCoach'
        self.build_name()
        
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

            ckpt_path = os.path.join(paths_config.checkpoints_dir, hyperparameters.load_embedding_coach_name, f'{image_name}.pt')

            w_pivot, camera, self.G = self.load(ckpt_path)
            self.log_video(w_pivot, self.G, os.path.join(paths_config.video_output_dir, f'{image_name}.mp4'))


