import torch
import os
from PIL import Image
import spi.configs.paths_config as paths_config


def log_image_from_w(w, c, G, name):
	if len(w.size()) <= 2:
		w = w.unsqueeze(0)
	with torch.no_grad():
		img_tensor = G.synthesis(w, c, noise_mode='const')['image']
		img = img_tensor[0].permute(1, 2, 0)
		img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
	Image.fromarray(img).save(os.path.join(paths_config.experiments_output_dir, name + '.jpg'))
	return img_tensor


def tensor2im(var, vmin=-1, vmax=1):
	# var shape: (3, H, W)
	if len(var.shape) == 4:
		var = var[0]
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var - vmin) / (vmax - vmin))
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))


def tensor2depth(var):
	# var shape: (3, H, W)
	if len(var.shape) == 4:
		var = var[0]
	var = var.cpu().detach()[0].numpy()
	vmax = var.max()
	vmin = var.min()
	var = ((var - vmin) / (vmax - vmin))
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))


def log_image(t, name, vmin=-1, vmax=1, mode='jpg'):
	t = t.type(torch.FloatTensor)
	if len(t.shape) == 4:
		t = t[0]
	if t.shape[0] == 1:
		img = tensor2depth(t)
	else:
		img = tensor2im(t, vmin=vmin, vmax=vmax)
	img.save(os.path.join(paths_config.experiments_output_dir, name + '.' + mode))
	return


def log_mask(t, name):
	img = tensor2im(t)
	img.save(os.path.join(paths_config.experiments_output_dir, name + '.jpg'))
	return