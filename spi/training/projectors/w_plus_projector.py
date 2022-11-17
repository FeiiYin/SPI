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


def project(
		G,
		target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
		c,
		lpips_func,
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
	# np.save('./mpti/training/w_avg.npy')
	# exit()
	# w_avg_tensor = torch.from_numpy(w_avg).to(global_config.device).repeat(1, 14, 1)
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
		synth_images = G.synthesis(ws, c, noise_mode='const')['image']

		# Features for synth images.
		dist = lpips_func(synth_images, target)

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
		
		loss = dist + reg_loss * regularize_noise_weight  # + reg_avg_loss * 1

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
