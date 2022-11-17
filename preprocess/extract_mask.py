import os
import sys
sys.path.append(os.path.abspath('.'))

import numpy as np
from PIL import Image
import torch
import glob
import tqdm
import torch.nn.functional as F
from spi.utils.load_utils import load_bisenet


def cal_face_mask(bisenet, image):
	# print(image.shape)
	target_images = image.clone()
	out = bisenet(target_images)[0]  # 1, 19, 512, 512
	# atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
	# 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
	att_list = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]
	parsing = torch.argmax(out, dim=1, keepdim=True)
	mask = torch.zeros(image.shape[0], 1, 512, 512).cuda()
	for att in att_list:
		mask += (parsing == att)
	# log_image(mask, 'predict_mask_512')
	mask = F.interpolate(mask, size=(256, 256), mode='nearest')
	# print('mask shape', mask.shape)
	# log_image(mask, 'predict_mask')
	return mask


bisenet = load_bisenet()


def cal_mask(bisenet, image):
	# print(image.shape)
	target_images = image.clone()
	out = bisenet(target_images)[0]  # 1, 19, 512, 512
	# atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
	# 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
	# att_list = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]
	parsing = torch.argmax(out, dim=1, keepdim=True)
	# mask = torch.zeros(image.shape[0], 1, 512, 512).cuda()
	# for att in att_list:
		# mask += (parsing == att)
	# log_image(mask, 'predict_mask_512')
	# mask = F.interpolate(mask, size=(256, 256), mode='nearest')
	# print('mask shape', mask.shape)
	# log_image(mask, 'predict_mask')
	return parsing

def extract_mask(input_dir, output_dir, mode='png'):
	image_list = sorted(glob.glob(f'{input_dir}/*.{mode}'))
	# for image_path in tqdm.tqdm(image_list):
	for image_path in image_list:
		image_name = os.path.basename(image_path).split('.')[0]
		image = Image.open(image_path).resize((512, 512))
		image = torch.from_numpy(np.asarray(image)).unsqueeze(0).permute(0, 3, 1, 2)
		# print(image.shape)
		image = (image.to('cuda').to(torch.float32) / 127.5 - 1)
		mask = cal_mask(bisenet, image)
		torch.save(mask.cpu(), os.path.join(output_dir, image_name + '.pt'))


def main():
	root = '/apdcephfs_cq2/share_1290939/feiiyin/InvEG3D_result/mpti_output/pti_input/cropped/'
	out = '/apdcephfs_cq2/share_1290939/feiiyin/InvEG3D_result/mpti_output/pti_input/mask/'

	
	# root = '/apdcephfs_cq2/share_1290939/feiiyin/InvEG3D_result/personal/origin/barack_obama/'

	image_list = sorted(glob.glob(f'{root}/*.png'))
	for image_path in tqdm.tqdm(image_list):
		image_name = os.path.basename(image_path).split('.')[0]
		image = Image.open(image_path).resize((512, 512))
		image = torch.from_numpy(np.asarray(image)).unsqueeze(0).permute(0, 3, 1, 2)
		# print(image.shape)
		image = (image.to('cuda').to(torch.float32) / 127.5 - 1)
		mask = cal_face_mask(bisenet, image)
		torch.save(mask.cpu(), os.path.join(out, image_name + '.pt'))