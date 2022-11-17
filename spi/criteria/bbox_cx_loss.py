from torchvision.ops import roi_align
import torch
from PIL import Image
import torch.nn.functional as F
import torchvision
# from mpti.utils.log_utils import tensor2im

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


def get_landmark_bbox(lm, scale=1):
	l_eye_id = [36, 42]
	r_eye_id = [42, 48]
	nose_id = [27, 36]
	mouth_id = [48, 68]
	p = 8
	bbox = []
	for _i, box_id in enumerate([mouth_id, l_eye_id, r_eye_id, nose_id]):
		box_lm = lm[:, box_id[0]:box_id[1]]
		ly, ry = torch.min(box_lm[:, :, 0], dim=1)[0], torch.max(box_lm[:, :, 0], dim=1)[0]
		lx, rx = torch.min(box_lm[:, :, 1], dim=1)[0], torch.max(box_lm[:, :, 1], dim=1)[0]  # shape: [b]
		lx, rx, ly, ry = (lx * scale).long(), (rx * scale).long(), (ly * scale).long(), (ry * scale).long()
		if _i == 1 or _i == 2:
			p = 15
		lx, rx, ly, ry = lx - p, rx + p, ly - p, ry + p
		lx, rx, ly, ry = lx.unsqueeze(1), rx.unsqueeze(1), ly.unsqueeze(1), ry.unsqueeze(1)
		bbox.append(torch.cat([ly, lx, ry, rx], dim=1))
	return bbox



def get_bbox(image, fake_image, lm):
	assert image.shape[-1] == 256
	# image = F.interpolate(image, size=(256, 256), mode='area')
	# fake_image = F.interpolate(fake_image, size=(256, 256), mode='area')

	bbox = get_landmark_bbox(lm)
	_i = torch.arange(image.shape[0]).unsqueeze(1).cuda()

	gt_mouth_bbox = torch.cat([_i, bbox[0]], dim=1).float().cuda()
	gt_mouth = roi_align(image, boxes=gt_mouth_bbox, output_size=80)
	fake_mouth = roi_align(fake_image, boxes=gt_mouth_bbox, output_size=80)

	gt_l_eye_bbox = torch.cat([_i, bbox[1]], dim=1).float().cuda()
	gt_l_eye = roi_align(image, boxes=gt_l_eye_bbox, output_size=80)
	fake_l_eye = roi_align(fake_image, boxes=gt_l_eye_bbox, output_size=80)

	gt_r_eye_bbox = torch.cat([_i, bbox[2]], dim=1).float().cuda()
	gt_r_eye = roi_align(image, boxes=gt_r_eye_bbox, output_size=80)
	fake_r_eye = roi_align(fake_image, boxes=gt_r_eye_bbox, output_size=80)

	return gt_mouth, gt_l_eye, gt_r_eye, fake_mouth, fake_l_eye, fake_r_eye
	# vgg = VGG19().cuda()
	# print(gt_r_eye.shape)
	# gt_r_eye = vgg(gt_r_eye)
	# print(gt_r_eye.shape)
	
	# print(gt_l_eye.shape, gt_l_eye.max())
	# tensor2im(gt_l_eye, vmin=0).save('./criterion/gt_l_eye.jpg')
	# tensor2im(fake_l_eye, vmin=0).save('./criterion/fake_l_eye.jpg')
	# tensor2im(gt_r_eye, vmin=0).save('./criterion/gt_r_eye.jpg')
	# tensor2im(fake_r_eye, vmin=0).save('./criterion/fake_r_eye.jpg')
	# tensor2im(gt_mouth, vmin=0).save('./criterion/gt_mouth.jpg')
	# tensor2im(fake_mouth, vmin=0).save('./criterion/fake_mouth.jpg')


class VGG19(torch.nn.Module):
	def __init__(self, requires_grad=False):
		super(VGG19, self).__init__()
		vgg_pretrained_features = torchvision.models.vgg.vgg19(pretrained=True).features
		self.slice1 = torch.nn.Sequential()
		for x in range(6):
			self.slice1.add_module(str(x), vgg_pretrained_features[x])
	  
		if not requires_grad:
			for param in self.parameters():
				param.requires_grad = False

	def forward(self, X):
		h = self.slice1(X)
		return h


def compute_cosine_distance(x, y):
	# mean shifting by channel-wise mean of `y`.
	y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
	x_centered = x - y_mu
	y_centered = y - y_mu

	# L2 normalization
	x_normalized = F.normalize(x_centered, p=2, dim=1)
	y_normalized = F.normalize(y_centered, p=2, dim=1)

	# channel-wise vectorization
	N, C, *_ = x.size()
	x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
	y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)

	# consine similarity
	cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)  # (N, H*W, H*W)

	# convert to distance
	dist = 1 - cosine_sim

	return dist


def compute_relative_distance(dist_raw):
	dist_min, _ = torch.min(dist_raw, dim=2, keepdim=True)
	dist_tilde = dist_raw / (dist_min + 1e-5)
	# add here.
	dist_tilde = torch.clamp(dist_tilde, max=10., min=-10)
	return dist_tilde


def compute_cx(dist_tilde, band_width):
	# Easy to get NaN
	w = torch.exp((1 - dist_tilde) / band_width)  # Eq(3)
	cx = w / torch.sum(w, dim=2, keepdim=True)  # Eq(4)
	return cx



def flip_landmark(lm):
	# TODO: since flip will make left eye become right eye, the landmark index should be re-arranged; this is not right
	n_lm = lm.view(-1, 68, 2).clone()
	n_lm[:, :, 0] = 256 - lm[:, :, 0]
	return lm



class BoxCXLoss(torch.nn.Module):
	def __init__(self,
				 band_width: float = 0.5):
		super(BoxCXLoss, self).__init__()
		self.band_width = band_width

		self.vgg_model = VGG19()
		self.register_buffer(
			name='vgg_mean',
			tensor=torch.tensor(
				[[[0.485]], [[0.456]], [[0.406]]], requires_grad=False)
		)
		self.register_buffer(
			name='vgg_std',
			tensor=torch.tensor(
				[[[0.229]], [[0.224]], [[0.225]]], requires_grad=False)
		)

	def forward(self, x, y, lm):
		if x.shape[-1] > 256:
			x = F.interpolate(x, (256, 256), mode='bilinear', align_corners=False)
		if y.shape[-1] > 256:
			y = F.interpolate(y, (256, 256), mode='bilinear', align_corners=False)

		x = x.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
		y = y.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())

		gt_mouth, gt_l_eye, gt_r_eye, fake_mouth, fake_l_eye, fake_r_eye = get_bbox(x, y, lm)
		loss = 0
		for (_x, _y) in [(gt_mouth, fake_mouth), (gt_l_eye, fake_l_eye), (gt_r_eye, fake_r_eye)]:
			_x = self.vgg_model(_x)
			_y = self.vgg_model(_y)
			# N, C, H, W = _x.size()
			dist_raw = compute_cosine_distance(_x, _y)
			dist_tilde = compute_relative_distance(dist_raw)
			cx = compute_cx(dist_tilde, self.band_width)
			cx = torch.mean(torch.max(cx, dim=1)[0], dim=1)  # Eq(1)
			cx_loss = torch.mean(-torch.log(cx + 1e-5))  # Eq(5)

			loss += cx_loss
		loss = loss * 0.1
		return loss


class BoxLoss(torch.nn.Module):
	def __init__(self,
				 band_width: float = 0.5):
		super(BoxLoss, self).__init__()
		self.band_width = band_width

		self.vgg_model = VGG19()
		self.register_buffer(
			name='vgg_mean',
			tensor=torch.tensor(
				[[[0.485]], [[0.456]], [[0.406]]], requires_grad=False)
		)
		self.register_buffer(
			name='vgg_std',
			tensor=torch.tensor(
				[[[0.229]], [[0.224]], [[0.225]]], requires_grad=False)
		)
		self.metric = torch.nn.SmoothL1Loss(reduction='mean')

	def forward(self, x, y, lm):
		if x.shape[-1] > 256:
			x = F.interpolate(x, (256, 256), mode='bilinear', align_corners=False)
		if y.shape[-1] > 256:
			y = F.interpolate(y, (256, 256), mode='bilinear', align_corners=False)

		x = x.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
		y = y.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())

		gt_mouth, gt_l_eye, gt_r_eye, fake_mouth, fake_l_eye, fake_r_eye = get_bbox(x, y, lm)
		loss = 0
		for (_x, _y) in [(gt_mouth, fake_mouth), (gt_l_eye, fake_l_eye), (gt_r_eye, fake_r_eye)]:
			_x = self.vgg_model(_x)
			_y = self.vgg_model(_y)
			# N, C, H, W = _x.size()
			
			loss += self.metric(_x, _y)
		return loss

def test():
	from PIL import Image
	import numpy as np
	from torchvision import transforms
	image_path = '/apdcephfs_cq2/share_1290939/feiiyin/InvEG3D_result/mpti_output/pti_input/hua_2/cropped/0.png'
	lm_path = '/apdcephfs_cq2/share_1290939/feiiyin/InvEG3D_result/mpti_output/pti_input/hua_2/lm/0.npy'
	image = Image.open(image_path).convert('RGB')
	t = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
	image = np.asarray(image)
	image = t(image)
	image = image.unsqueeze(0).cuda().float()
	# image = 
	lm = torch.from_numpy(np.load(lm_path)).unsqueeze(0).cuda().float()
	# import pdb; pdb.set_trace()
	get_bbox(image, image, lm)

	# test_flip 
	
	flip_lm = flip_landmark(lm)
	flip_image = torch.flip(image, dims=[3])
	get_bbox(flip_image, image, flip_lm)

	
# test()

def test_loss():
	from PIL import Image
	import numpy as np
	from torchvision import transforms
	image_path = '/apdcephfs_cq2/share_1290939/feiiyin/InvEG3D_result/mpti_output/pti_input/hua_2/cropped/0.png'
	lm_path = '/apdcephfs_cq2/share_1290939/feiiyin/InvEG3D_result/mpti_output/pti_input/hua_2/lm/0.npy'
	image = Image.open(image_path).convert('RGB')
	t = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
	image = np.asarray(image)
	image = t(image)
	image = image.unsqueeze(0).cuda().float()
	# image = 
	lm = torch.from_numpy(np.load(lm_path)).unsqueeze(0).cuda().float()

	loss_func = BoxCXLoss().cuda()
	fake = torch.randn_like(image)
	loss = loss_func(image, fake, lm)
	print(loss)

# test_loss()

# def get_mask_bbox(image, mask, scale=1):
# 	#   glasses seems not ok
# 	# atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
# 	# 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
# 	att_list = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]
# 	# parsing = torch.argmax(out, dim=1)
# 	# mask = torch.zeros(1, 1, 512, 512).cuda()
# 	# for att in att_list:
# 	# 	mask += (parsing == att)
# 	l_eye = (mask == 4)
# 	r_eye = (mask == 5)
# 	r_eye = (mask == 5)
# 	r_eye = (mask == 5)
# 	mask = F.interpolate(mask, size=(256, 256), mode='nearest')
	
# 	l_eye_id = [36, 42]
# 	r_eye_id = [42, 48]
# 	nose_id = [27, 36]
# 	mouth_id = [48, 68]
# 	p = 8
# 	bbox = []
# 	for _i, box_id in enumerate([mouth_id, l_eye_id, r_eye_id, nose_id]):
# 		box_lm = lm[:, box_id[0]:box_id[1]]
# 		ly, ry = torch.min(box_lm[:, :, 0], dim=1)[0], torch.max(box_lm[:, :, 0], dim=1)[0]
# 		lx, rx = torch.min(box_lm[:, :, 1], dim=1)[0], torch.max(box_lm[:, :, 1], dim=1)[0]  # shape: [b]
# 		lx, rx, ly, ry = (lx * scale).long(), (rx * scale).long(), (ly * scale).long(), (ry * scale).long()
# 		if _i == 1 or _i == 2:
# 			p = 15
# 		lx, rx, ly, ry = lx - p, rx + p, ly - p, ry + p
# 		lx, rx, ly, ry = lx.unsqueeze(1), rx.unsqueeze(1), ly.unsqueeze(1), ry.unsqueeze(1)
# 		bbox.append(torch.cat([ly, lx, ry, rx], dim=1))
# 	return bbox


