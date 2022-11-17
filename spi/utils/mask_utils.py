import torch


def calculate_face_mask(mask):
	att_list = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]
	face_mask = torch.zeros_like(mask)
	for att in att_list:
		face_mask += (mask == att)
	return face_mask






def calculate_face_mask_w_model(bisenet, image):
	# bisenet = load_bisenet()
	out = bisenet(image)[0]  # 1, 19, 512, 512
	# atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
	# 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
	att_list = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]
	parsing = torch.argmax(out, dim=1, keepdim=True)
	N, _, H, W = image.shape
	mask = torch.zeros(N, 1, H, W).float().to(image.device)
	for att in att_list:
		mask += (parsing == att)
	# log_image(mask, 'predict_mask_512')
	# mask = torch.nn.functional.interpolate(mask, size=(256, 256), mode='nearest')
	# log_image(mask, 'predict_mask')
	return mask