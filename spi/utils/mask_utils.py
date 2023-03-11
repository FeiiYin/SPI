import torch


def calculate_face_mask(mask):
	att_list = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]
	face_mask = torch.zeros_like(mask)
	for att in att_list:
		face_mask += (mask == att)
	return face_mask






def calculate_face_mask_w_model(bisenet, image):
	out = bisenet(image)[0]  # 1, 19, 512, 512
	att_list = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]
	parsing = torch.argmax(out, dim=1, keepdim=True)
	N, _, H, W = image.shape
	mask = torch.zeros(N, 1, H, W).float().to(image.device)
	for att in att_list:
		mask += (parsing == att)
	return mask