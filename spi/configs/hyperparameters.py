## Architechture
lpips_type = 'vgg'
max_images_to_invert = 3000

# w stage
use_encoder = False
use_G_avg = False
first_inv_type = 'sg'  # 'mir', 'sgw+'
optim_type = 'adam'
first_inv_steps = 500

# G stage 1
LPIPS_value_threshold = 0.05
G_1_step = 0
G_1_type = None
G_2_step = 0
use_adapt_yaw_range = False
description = None


## Locality regularization
latent_ball_num_of_samples = 1
locality_regularization_interval = 1
use_locality_regularization = False
regulizer_l2_lambda = 0.1
regulizer_lpips_lambda = 0.1
regulizer_alpha = 30
reg_w_loss_weight = 1

## Loss
pt_l2_lambda = 1
pt_lpips_lambda = 1
pt_tv_lambda = 0  # 0.25
pt_rot_lambda = 0.1
pt_mirror_rot_lambda = 0.05
pt_depth_lambda = 1


## Optimization
pti_learning_rate = 3e-4
first_inv_lr = 5e-3
train_batch_size = 1
use_last_w_pivots = False
load_embedding_coach_name = None
w_space_index = 14
