import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./eg3d/'))
import argparse
from torch.utils.data import DataLoader
from eg3d.torch_utils.ops import grid_sample_gradfix

from spi.configs import global_config, hyperparameters, paths_config
from spi.training.coaches.pti_coach import SingleIDCoach
from spi.training.coaches.rot_bbox_cx_coach import RotBboxCoach
from spi.training.coaches.inference_coach import InferenceCoach
from spi.data.images_dataset import PTIDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--data_root', type=str, default='/apdcephfs_cq2/share_1290939/feiiyin/dataset/CelebAHQ_test_camera/')
    parser.add_argument('--data_mode', type=str, default='png')
    parser.add_argument('--output_root', type=str, default=None)
    parser.add_argument('--use_encoder', action='store_true', default=False)
    parser.add_argument('--use_G_avg', action='store_true', default=False)

    parser.add_argument('--use_adapt_yaw_range', action='store_true', default=False)
    parser.add_argument('--not_use_wandb', action='store_true', default=False)
    
    parser.add_argument('--first_inv_type', type=str, default='pti')
    parser.add_argument('--first_inv_steps', type=int, default=500)
    parser.add_argument('--G_1_step', type=int, default=500)
    parser.add_argument('--G_1_type', type=str, default='space')
    parser.add_argument('--G_2_step', type=int, default=500)
    parser.add_argument('--load_embedding_coach_name', type=str, default=None)

    parser.add_argument('--pt_rot_lambda', type=float, default=0)
    parser.add_argument('--pt_mirror_rot_lambda', type=float, default=0)
    parser.add_argument('--pt_depth_lambda', type=float, default=0)
    parser.add_argument('--pt_tv_lambda', type=float, default=0)

    parser.add_argument('--description', type=str, default=None)
    parser.add_argument('--dataset_block', type=str, default=None, help='1/20')
    parser.add_argument('--select_range', type=int, default=None, help='100')
    parser.add_argument('--filter_index', type=str, default=None, help='1,2,3')
    args = parser.parse_args()

    hyperparameters.use_encoder = args.use_encoder
    hyperparameters.use_G_avg = args.use_G_avg
    hyperparameters.first_inv_type = args.first_inv_type
    hyperparameters.first_inv_steps = args.first_inv_steps
    hyperparameters.G_1_step = args.G_1_step
    hyperparameters.G_1_type = args.G_1_type
    hyperparameters.G_2_step = args.G_2_step
    hyperparameters.load_embedding_coach_name = args.load_embedding_coach_name
    hyperparameters.use_adapt_yaw_range = args.use_adapt_yaw_range
    hyperparameters.description = args.description
    hyperparameters.pt_rot_lambda = args.pt_rot_lambda
    hyperparameters.pt_mirror_rot_lambda = args.pt_mirror_rot_lambda
    hyperparameters.pt_depth_lambda = args.pt_depth_lambda
    hyperparameters.pt_tv_lambda = args.pt_tv_lambda

    if args.output_root is not None:
        paths_config.root = args.output_root

        paths_config.checkpoints_dir = paths_config.root + 'checkpoints/'
        paths_config.embedding_base_dir = paths_config.root + 'embedding/' # root + 'embeddings'
        paths_config.experiments_output_dir = paths_config.root + 'experiments/'

        # Used in the final test
        paths_config.images_output_dir = paths_config.root + 'image/'
        paths_config.mirror_images_output_dir = paths_config.root + 'image_m/'
        paths_config.video_output_dir = paths_config.root + 'video/'

        for dir in [paths_config.checkpoints_dir, 
            paths_config.embedding_base_dir, 
            paths_config.experiments_output_dir, 
            paths_config.images_output_dir, 
            paths_config.mirror_images_output_dir, 
            paths_config.video_output_dir
        ]:
            os.makedirs(dir, exist_ok=True)
    
    return args


def build_dataset(args):
    root = args.data_root

    if args.filter_index is not None:
        args.filter_index = args.filter_index.split(',')
    dataset = PTIDataset(
        source_root=os.path.join(root, 'crop'), 
        c_root=os.path.join(root, 'c'), 
        w_root=None,
        mask_root=os.path.join(root, 'mask'),
        lm_root=os.path.join(root, 'lm'),
        target_name='target',
        mode=args.data_mode,
        dataset_block=args.dataset_block,
        select_range=args.select_range,
        filter_index=args.filter_index,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataset, dataloader


def run():
    args = parse_args()
    use_wandb = not args.not_use_wandb
    
    os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    _, dataloader = build_dataset(args)

    if args.G_1_type == 'pti':
        coach = SingleIDCoach(dataloader, use_wandb)
    elif args.G_1_type == 'RotBbox':
        coach = RotBboxCoach(dataloader, use_wandb)
    elif args.G_1_type == 'Inference':
        coach = InferenceCoach(dataloader, use_wandb)
    else:
        raise NotImplementedError
    coach.train()
    return global_config.run_name



if __name__ == '__main__':
    grid_sample_gradfix.enabled = True
    run()

