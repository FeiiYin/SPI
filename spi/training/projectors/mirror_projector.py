# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from spi.configs import global_config, hyperparameters
from spi.utils.log_utils import log_image
from spi.utils.camera_utils import sample_camera, cal_canonical_c, cal_mirror_c, cal_camera_weight


def project(
		G,
		target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
		c,
		lpips_func,
		# wloss_func,
		fg_mask,
		*,
		initial_w=None,
		num_steps=1000,
		w_avg_samples=10000,
		initial_learning_rate=0.01,
		initial_noise_factor=0.05,
		lr_rampdown_length=0.25,
		lr_rampup_length=0.05,
		noise_ramp_length=0.75,
		regularize_noise_weight=1e5,
		verbose=False,
		device: torch.device,
		image_log_step=global_config.log_snapshot,
		w_name: str
):
	# print('w_plus_projector.')
	assert target.shape[1:] == (G.img_channels, G.img_resolution, G.img_resolution)

	# def logprint(*args):
	# 	if verbose:
	# 		print(*args)

	G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float() # type: ignore

	# Compute w stats.
	# logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
	z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
	c_samples = c.repeat(w_avg_samples, 1)
	w_samples = G.mapping(torch.from_numpy(z_samples).to(device), c_samples)  # [N, L, C]
	w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
	
	w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
	w_avg_tensor = torch.from_numpy(w_avg).to(global_config.device).repeat(1, 14, 1)
	w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

	# Setup noise inputs.
	noise_bufs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}

	if initial_w is not None:
		start_w = initial_w
	else:
		start_w = w_avg
		start_w = np.repeat(start_w, G.backbone.mapping.num_ws, axis=1)

	w_opt = torch.tensor(start_w, dtype=torch.float32, device=device, requires_grad=False)  # pylint: disable=not-callable
	w_opt.requires_grad_(True)
	
	optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=hyperparameters.first_inv_lr)
	
	# Init noise.
	for buf in noise_bufs.values():
		buf[:] = torch.randn_like(buf)
		buf.requires_grad = True

	# w_gt = wloss_func.cal_w(target, c)
	batch_size = 2

	# canonical_camera = [c]
	# for x in [-0.4, 0, 0.4]:
	# 	cam = cal_canonical_c(yaw_angle=x, pitch_angle=-0.2, batch_size=1, device='cuda')
	# 	canonical_camera.append(cam)
	# canonical_camera = torch.cat(canonical_camera, dim=0)

	target_m = torch.flip(target, dims=[3])

	camera_m = cal_mirror_c(camera=c)
	target_camera = torch.cat([c, camera_m], dim=0)
	weight_m = cal_camera_weight(camera_m)[0]


	pool = torch.nn.MaxPool2d(kernel_size=15, stride=1, padding=7)
	dilated_mask = pool(fg_mask)
	log_image(dilated_mask, 'dilated_mask')
	bg_mask = 1 - dilated_mask
	bg_mask = torch.nn.functional.interpolate(bg_mask, (128, 128), mode='bilinear', align_corners=False)
	bg_mask_m = torch.flip(bg_mask, dims=[3])

	# print('original_ weight m', weight_m)  # 0.5306
	# weight_m = 1
	# target_weight = torch.tensor([1, weight_m[0]]).to(device)

	for step in tqdm(range(num_steps)):

		# Learning rate schedule.
		t = step / num_steps
		w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
		lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
		lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
		lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
		lr = initial_learning_rate * lr_ramp
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

		# Synth images from opt_w.
		w_noise = torch.randn_like(w_opt) * w_noise_scale
		ws = (w_opt + w_noise)

		# sample_c = sample_camera(batch_size=batch_size - 4, yaw_range=0.5, pitch_range=0.2, device=device)
		# new_c = torch.cat([canonical_camera, sample_c], dim=0)
		ws = ws.repeat(batch_size, 1, 1)

		gen_results = G.synthesis(ws, target_camera, noise_mode='const')
		synth_images = gen_results['image']
		depth = gen_results['image_depth']

		# Features for synth images.
		dist = lpips_func(synth_images[:1], target) + lpips_func(synth_images[1:], target_m) * weight_m

		# Noise regularization.
		reg_loss = 0.0
		for v in noise_bufs.values():
			noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
			while True:
				reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
				reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
				if noise.shape[2] <= 8:
					break
				noise = F.avg_pool2d(noise, kernel_size=2)
		
		# w_loss = wloss_func.cal_loss(synth_images, new_c, w_gt.repeat(batch_size, 1, 1))
		threshold = torch.max(depth).detach() # + 0.05
		bg_loss = threshold - torch.mean(depth[:1] * bg_mask + depth[1:] * bg_mask_m)
		# bg_loss = 0
		# 0.5 for side effect
		loss = dist + reg_loss * regularize_noise_weight # + bg_loss * 0.01

		# Step
		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()

		# Normalize noise.
		with torch.no_grad():
			for buf in noise_bufs.values():
				buf -= buf.mean()
				buf *= buf.square().mean().rsqrt()

		if verbose and step % image_log_step == 0:
			print(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')
			with torch.no_grad():
				global_config.training_step += 1
				log_image(synth_images, f'w_{w_name}_{step}')
	
	del G
	return w_opt


