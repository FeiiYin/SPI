# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""
import os
import sys
sys.path.append(os.path.abspath('.'))

import eg3d.dnnlib as dnnlib
import imageio
import numpy as np
import scipy.interpolate
import torch
from tqdm import tqdm
import mrcfile
import math
import eg3d.legacy as legacy

from eg3d.camera_utils import LookAtPoseSampler
from eg3d.torch_utils import misc

#----------------------------------------------------------------------------


def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
	batch_size, channels, img_h, img_w = img.shape
	if grid_w is None:
		grid_w = batch_size // grid_h
	assert batch_size == grid_w * grid_h
	if float_to_uint8:
		img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
	img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
	img = img.permute(2, 0, 3, 1, 4)
	img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
	if chw_to_hwc:
		img = img.permute(1, 2, 0)
	if to_numpy:
		img = img.cpu().numpy()
	return img

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
	# NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
	voxel_origin = np.array(voxel_origin) - cube_length/2
	voxel_size = cube_length / (N - 1)

	overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
	samples = torch.zeros(N ** 3, 3)

	# transform first 3 columns
	# to be the x, y, z index
	samples[:, 2] = overall_index % N
	samples[:, 1] = (overall_index.float() / N) % N
	samples[:, 0] = ((overall_index.float() / N) / N) % N

	# transform first 3 columns
	# to be the x, y, z coordinate
	samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
	samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
	samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

	num_samples = N ** 3

	return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------

def gen_interp_video(
	G, 
	G_kwargs,
	mp4: str, 
	images=None, 
	shuffle_seed=None, 
	w_frames=30*4, 
	kind='cubic', 
	grid_dims=(1,1), 
	num_keyframes=None, 
	wraps=2, 
	psi=1, 
	truncation_cutoff=14, 
	cfg='FFHQ', 
	image_mode='image', 
	gen_shapes=False, 
	device=torch.device('cuda'), 
	**video_kwargs
):
	grid_w = grid_dims[0]
	grid_h = grid_dims[1]

	if images is None:
		images = torch.empty(len(G_kwargs['w']), 1, 1, 1).cuda()
	if num_keyframes is None:
		if len(images) % (grid_w*grid_h) != 0:
			raise ValueError('Number of input seeds must be divisible by grid W*H')
		num_keyframes = len(images) // (grid_w*grid_h)

	# all_seeds = np.zeros(num_keyframes*grid_h*grid_w, dtype=np.int64)
	# for idx in range(num_keyframes*grid_h*grid_w):
	#     all_seeds[idx] = seeds[idx % len(seeds)]

	# if shuffle_seed is not None:
	#     rng = np.random.RandomState(seed=shuffle_seed)
	#     rng.shuffle(all_seeds)

	camera_lookat_point = torch.tensor([0, 0, 0.2], device=device) if cfg == 'FFHQ' else torch.tensor([0, 0, 0], device=device)

	# zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])).to(device)
	cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point, radius=2.7, device=device)
	intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
	c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
	c = c.repeat(len(images), 1)

	ws = G_kwargs['w']
	if 'wt' in G_kwargs:
		wt = G_kwargs['wt']
	else:
		wt = None
	# plane_supp = G_kwargs['plane_supp'] if 'plane_supp' in G_kwargs else None
	# plane = G_kwargs['plane'] if 'plane' in G_kwargs else None
	# ws = G.mapping(z=zs, c=c, truncation_psi=psi, truncation_cutoff=truncation_cutoff)
	_ = G.synthesis(ws[:1], c[:1]) # warm up
	ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])
	if wt is not None:
		wt = wt.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])

	# Interpolation.
	grid = []
	grid_t = []
	for yi in range(grid_h):
		row = []
		rowt = []
		for xi in range(grid_w):
			x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
			y = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
			interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
			row.append(interp)
			if wt is not None:
				yt = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
				interp_t = scipy.interpolate.interp1d(x, yt, kind=kind, axis=0)
				rowt.append(interp_t)
		grid.append(row)
		if wt is not None:
			grid_t.append(rowt)

	# Render video.
	max_batch = 10000000
	voxel_resolution = 512
	video_out = imageio.get_writer(mp4, mode='I', fps=60, codec='libx264', **video_kwargs)

	if gen_shapes:
		outdir = os.path.dirname(mp4)
		outdir = os.path.join(outdir, 'interpolation_shape')
		os.makedirs(outdir, exist_ok=True)
	all_poses = []
	for frame_idx in tqdm(range(num_keyframes * w_frames)):
		imgs = []
		for yi in range(grid_h):
			for xi in range(grid_w):
				pitch_range = 0.4
				yaw_range = 0.7
				cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
														3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
														camera_lookat_point, radius=2.7, device=device)
				all_poses.append(cam2world_pose.squeeze().cpu().numpy())
				intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
				c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

				interp = grid[yi][xi]
				w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)
				
				if 'planes_s' in G_kwargs:
					img = G.synthesis_planes(ws=w.unsqueeze(0), c=c[0:1], planes=G_kwargs['planes_s'], planes_t=G_kwargs['planes_t'], noise_mode='const')[image_mode][0]
				elif 'wt' in G_kwargs:
					img = G.synthesis(ws=w.unsqueeze(0), c=c[0:1], noise_mode='const', wt=G_kwargs['wt'])[image_mode][0]
				else:
					img = G.synthesis(ws=w.unsqueeze(0), c=c[0:1], noise_mode='const')[image_mode][0]

				if image_mode == 'image_depth':
					img = -img
					img = (img - img.min()) / (img.max() - img.min()) * 2 - 1

				imgs.append(img)

				if gen_shapes:
					# generate shapes
					print('Generating shape for frame %d / %d ...' % (frame_idx, num_keyframes * w_frames))
					
					samples, voxel_origin, voxel_size = create_samples(N=voxel_resolution, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'])
					samples = samples.to(device)
					sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
					transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
					transformed_ray_directions_expanded[..., -1] = -1

					head = 0
					with tqdm(total = samples.shape[1]) as pbar:
						with torch.no_grad():
							while head < samples.shape[1]:
								torch.manual_seed(0)
								sigma = G.sample_mixed(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], w.unsqueeze(0), truncation_psi=psi, noise_mode='const')['sigma']
								sigmas[:, head:head+max_batch] = sigma
								head += max_batch
								pbar.update(max_batch)

					sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
					sigmas = np.flip(sigmas, 0)
					
					pad = int(30 * voxel_resolution / 256)
					pad_top = int(38 * voxel_resolution / 256)
					sigmas[:pad] = 0
					sigmas[-pad:] = 0
					sigmas[:, :pad] = 0
					sigmas[:, -pad_top:] = 0
					sigmas[:, :, :pad] = 0
					sigmas[:, :, -pad:] = 0

					output_ply = True
					if output_ply:
						from shape_utils import convert_sdf_samples_to_ply
						convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, os.path.join(outdir, f'{frame_idx:04d}_shape.ply'), level=10)
					else: # output mrc
						with mrcfile.new_mmap(outdir + f'{frame_idx:04d}_shape.mrc', overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
							mrc.data[:] = sigmas

		video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
	video_out.close()
	all_poses = np.stack(all_poses)

	if gen_shapes:
		print(all_poses.shape)
		with open(mp4.replace('.mp4', '_trajectory.npy'), 'wb') as f:
			np.save(f, all_poses)



def gen_interp_normal_video(
	G, 
	G_kwargs,
	mp4: str, 
	images=None, 
	shuffle_seed=None, 
	w_frames=30*4, 
	kind='cubic', 
	grid_dims=(1,1), 
	num_keyframes=None, 
	wraps=2, 
	psi=1, 
	truncation_cutoff=14, 
	cfg='FFHQ', 
	image_mode='image', 
	gen_shapes=False, 
	device=torch.device('cuda'), 
	**video_kwargs
):
	grid_w = grid_dims[0]
	grid_h = grid_dims[1]

	if images is None:
		images = torch.empty(len(G_kwargs['w']), 1, 1, 1).cuda()
	if num_keyframes is None:
		if len(images) % (grid_w*grid_h) != 0:
			raise ValueError('Number of input seeds must be divisible by grid W*H')
		num_keyframes = len(images) // (grid_w*grid_h)

	# all_seeds = np.zeros(num_keyframes*grid_h*grid_w, dtype=np.int64)
	# for idx in range(num_keyframes*grid_h*grid_w):
	#     all_seeds[idx] = seeds[idx % len(seeds)]

	# if shuffle_seed is not None:
	#     rng = np.random.RandomState(seed=shuffle_seed)
	#     rng.shuffle(all_seeds)

	camera_lookat_point = torch.tensor([0, 0, 0.2], device=device) if cfg == 'FFHQ' else torch.tensor([0, 0, 0], device=device)

	# zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])).to(device)
	cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point, radius=2.7, device=device)
	intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
	c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
	c = c.repeat(len(images), 1)

	ws = G_kwargs['w']
	if 'wt' in G_kwargs:
		wt = G_kwargs['wt']
	else:
		wt = None
	# plane_supp = G_kwargs['plane_supp'] if 'plane_supp' in G_kwargs else None
	# plane = G_kwargs['plane'] if 'plane' in G_kwargs else None
	# ws = G.mapping(z=zs, c=c, truncation_psi=psi, truncation_cutoff=truncation_cutoff)
	_ = G.synthesis(ws[:1], c[:1]) # warm up
	ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])
	if wt is not None:
		wt = wt.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])

	# Interpolation.
	grid = []
	grid_t = []
	for yi in range(grid_h):
		row = []
		rowt = []
		for xi in range(grid_w):
			x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
			y = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
			interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
			row.append(interp)
			if wt is not None:
				yt = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
				interp_t = scipy.interpolate.interp1d(x, yt, kind=kind, axis=0)
				rowt.append(interp_t)
		grid.append(row)
		if wt is not None:
			grid_t.append(rowt)

	# Render video.
	max_batch = 10000000
	voxel_resolution = 512
	video_out = imageio.get_writer(mp4, mode='I', fps=60, codec='libx264', **video_kwargs)

	all_poses = []
	for frame_idx in tqdm(range(num_keyframes * w_frames)):
		imgs = []
		for yi in range(grid_h):
			for xi in range(grid_w):
				pitch_range = 0.25
				yaw_range = 0.35
				cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
														3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
														camera_lookat_point, radius=2.7, device=device)
				all_poses.append(cam2world_pose.squeeze().cpu().numpy())
				intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
				c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

				interp = grid[yi][xi]
				w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)
			  
				# if 'planes_s' in G_kwargs:
				#     img = G.synthesis_planes(ws=w.unsqueeze(0), c=c[0:1], planes=G_kwargs['planes_s'], planes_t=G_kwargs['planes_t'], noise_mode='const')[image_mode][0]
				# elif 'wt' in G_kwargs:
				#     img = G.synthesis(ws=w.unsqueeze(0), c=c[0:1], noise_mode='const', wt=G_kwargs['wt'])[image_mode][0]
				# else:
				img = G.synthesis(ws=w.unsqueeze(0), c=c[0:1], noise_mode='const', if_normal=True)['normal'][0]

				# if image_mode == 'image_depth':
				#     img = -img
				#     img = (img - img.min()) / (img.max() - img.min()) * 2 - 1

				imgs.append(img)

		video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
	video_out.close()
	all_poses = np.stack(all_poses)




def gen_video(
	G, 
	G_kwargs,
	mp4: str, 
	w_frames=30*4, 
	cfg='FFHQ', 
	image_mode='image', 
	device=torch.device('cuda'), 
	**video_kwargs
):
	B = 1
	camera_lookat_point = torch.tensor([0, 0, 0.2], device=device) if cfg == 'FFHQ' else torch.tensor([0, 0, 0], device=device)

	# zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])).to(device)
	cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point, radius=2.7, device=device)
	intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
	c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
	c = c.repeat(B, 1)
	
	# _ = G.synthesis(ws[:1], c[:1]) # warm up
	
	# Render video.
	video_out = imageio.get_writer(mp4, mode='I', fps=30, codec='libx264', **video_kwargs)

	all_poses = []
	for frame_idx in tqdm(range(w_frames)):
		imgs = []

		pitch_range = 0.25
		yaw_range = 0.35
		cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (w_frames)),
												3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (w_frames)),
												camera_lookat_point, radius=2.7, device=device)
		all_poses.append(cam2world_pose.squeeze().cpu().numpy())
		intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
		c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

		if 'ws1' in G_kwargs:
			ws1 = G_kwargs['ws1']
			ws2 = G_kwargs['ws2']
			ws3 = G_kwargs['ws3']
			img = G.synthesis_triw(ws1=ws1, ws2=ws2, ws3=ws3, c=c[0:1], noise_mode='const')[image_mode][0]
		elif 'w' in G_kwargs:
			w = G_kwargs['w']
			img = G.synthesis(ws=w, c=c[0:1], noise_mode='const')[image_mode][0]
		else:
			assert False
		# if 'planes_s' in G_kwargs:
		#     img = G.synthesis_planes(ws=w.unsqueeze(0), c=c[0:1], planes=G_kwargs['planes_s'], planes_t=G_kwargs['planes_t'], noise_mode='const')[image_mode][0]
		# elif 'wt' in G_kwargs:
		#     img = G.synthesis(ws=w.unsqueeze(0), c=c[0:1], noise_mode='const', wt=G_kwargs['wt'])[image_mode][0]
		# else:
		#     img = G.synthesis(ws=w.unsqueeze(0), c=c[0:1], noise_mode='const')[image_mode][0]

		if image_mode == 'image_depth':
			img = -img
			img = (img - img.min()) / (img.max() - img.min()) * 2 - 1

		imgs.append(img)

		video_out.append_data(layout_grid(torch.stack(imgs), grid_w=1, grid_h=1))
	video_out.close()
	all_poses = np.stack(all_poses)

	

def gen_video_ide3d(
	G, 
	G_kwargs,
	mp4: str, 
	w_frames=60*4, 
	cfg='FFHQ', 
	image_mode='image', 
	device=torch.device('cuda'), 
	**video_kwargs
):
	B = 1
	camera_lookat_point = torch.tensor([0, 0, 0.2], device=device) if cfg == 'FFHQ' else torch.tensor([0, 0, 0], device=device)

	# zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])).to(device)
	cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point, radius=2.7, device=device)
	intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
	c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
	c = c.repeat(B, 1)
	
	# _ = G.synthesis(ws[:1], c[:1]) # warm up
	
	# Render video.
	video_out = imageio.get_writer(mp4, mode='I', fps=60, codec='libx264', **video_kwargs)

	all_poses = []
	for frame_idx in tqdm(range(w_frames)):
		imgs = []

		h = math.pi*(0.5 + 0.1*math.cos(2*math.pi*frame_idx/(0.5 * 240)))
		v = math.pi*(0.5 - 0.05*math.sin(2*math.pi*frame_idx/(0.5 * 240)))
	
		render_params = {
			"h_mean": h,
			"v_mean": v,
			"h_stddev": 0.,
			"v_stddev": 0.,
			"fov": 18,
			"num_steps": 96
		}

		cam2world_pose = LookAtPoseSampler.sample(
			horizontal_mean=h, vertical_mean=v, lookat_position=camera_lookat_point, horizontal_stddev=0, vertical_stddev=0, 
			radius=2.7, batch_size=1, device=device)
			
		# all_poses.append(cam2world_pose.squeeze().cpu().numpy())
		intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
		c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

		if 'ws1' in G_kwargs:
			ws1 = G_kwargs['ws1']
			ws2 = G_kwargs['ws2']
			ws3 = G_kwargs['ws3']
			img = G.synthesis_triw(ws1=ws1, ws2=ws2, ws3=ws3, c=c[0:1], noise_mode='const')[image_mode][0]
		elif 'w' in G_kwargs:
			w = G_kwargs['w']
			img = G.synthesis(ws=w, c=c[0:1], noise_mode='const')[image_mode][0]
		else:
			assert False
		# if 'planes_s' in G_kwargs:
		#     img = G.synthesis_planes(ws=w.unsqueeze(0), c=c[0:1], planes=G_kwargs['planes_s'], planes_t=G_kwargs['planes_t'], noise_mode='const')[image_mode][0]
		# elif 'wt' in G_kwargs:
		#     img = G.synthesis(ws=w.unsqueeze(0), c=c[0:1], noise_mode='const', wt=G_kwargs['wt'])[image_mode][0]
		# else:
		#     img = G.synthesis(ws=w.unsqueeze(0), c=c[0:1], noise_mode='const')[image_mode][0]

		if image_mode == 'image_depth':
			img = -img
			img = (img - img.min()) / (img.max() - img.min()) * 2 - 1

		imgs.append(img)

		video_out.append_data(layout_grid(torch.stack(imgs), grid_w=1, grid_h=1))
	video_out.close()
	# all_poses = np.stack(all_poses)

