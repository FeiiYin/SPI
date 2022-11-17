import copy
import os
from preprocess.extract_landmark import get_landmark
from preprocess.extract_3dmm import Extract3dmm, align_img
from preprocess.process_camera import process_camera
# from third_part.Deep3DFaceRecon_pytorch.util.preprocess import align_img
import numpy as np
from PIL import Image
import torch




def compute_rotation(angles):
		"""
		Return:
			rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

		Parameters:
			angles           -- torch.tensor, size (B, 3), radian
		"""

		batch_size = angles.shape[0]
		ones = torch.ones([batch_size, 1])
		zeros = torch.zeros([batch_size, 1])
		x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],
		
		rot_x = torch.cat([
			ones, zeros, zeros,
			zeros, torch.cos(x), -torch.sin(x), 
			zeros, torch.sin(x), torch.cos(x)
		], dim=1).reshape([batch_size, 3, 3])
		
		rot_y = torch.cat([
			torch.cos(y), zeros, torch.sin(y),
			zeros, ones, zeros,
			-torch.sin(y), zeros, torch.cos(y)
		], dim=1).reshape([batch_size, 3, 3])

		rot_z = torch.cat([
			torch.cos(z), -torch.sin(z), zeros,
			torch.sin(z), torch.cos(z), zeros,
			zeros, zeros, ones
		], dim=1).reshape([batch_size, 3, 3])

		rot = rot_z @ rot_y @ rot_x
		return rot.permute(0, 2, 1)[0]



class CameraExtractor:

	def __init__(self, crop_outdir, c_outdir, mode):
		self.crop_outdir = crop_outdir
		self.c_outdir = c_outdir

		self.model_3dmm = Extract3dmm({
			'BFM': '/apdcephfs/share_1290939/feiiyin/TH/PIRender_bak/Deep3DFaceRecon_pytorch/BFM/', 
			'3DMM': '/apdcephfs/share_1290939/feiiyin/TH/PIRender_bak/Deep3DFaceRecon_pytorch/checkpoints/model_name/epoch_20.pth',
		})
		self.lm3d_std = self.model_3dmm.lm3d_std
		self.mode = mode

	def set_path(self, crop_outdir, c_outdir, mode):
		self.crop_outdir = crop_outdir
		self.c_outdir = c_outdir
		self.mode = mode

	def __len__(self):
		return len(self.video_list)

	def crop(self, img_pil, lm, image_name):
		rescale_factor = 300
		center_crop_size = 700
		output_size = 512
		_, im_pil, lm, _, im_high = align_img(img_pil, lm, self.lm3d_std, rescale_factor=rescale_factor)

		left = int(im_high.size[0]/2 - center_crop_size/2)
		upper = int(im_high.size[1]/2 - center_crop_size/2)
		right = left + center_crop_size
		lower = upper + center_crop_size
		im_cropped = im_high.crop((left, upper, right, lower))
		im_cropped = im_cropped.resize((output_size, output_size), resample=Image.LANCZOS)
		
		im_cropped.save(os.path.join(self.crop_outdir, f'{image_name}.{self.mode}'), compress_level=0)
	
	def cal_camera(self, coeff_3dmm):
		angle = coeff_3dmm['angle']
		trans = coeff_3dmm['trans'][0]
		
		R = compute_rotation(angle).numpy()
		# R = compute_rotation(torch.from_numpy(angle)).numpy()
	
		trans[2] += -10
		c = -np.dot(R, trans)
		pose = np.eye(4)
		pose[:3, :3] = R

		c *= 0.27 # factor to match tripleganger
		c[1] += 0.006 # offset to align to tripleganger
		c[2] += 0.161 # offset to align to tripleganger
		pose[0,3] = c[0]
		pose[1,3] = c[1]
		pose[2,3] = c[2]

		focal = 2985.29 # = 1015*1024/224*(300/466.285)#
		pp = 512#112
		w = 1024#224
		h = 1024#224

		count = 0
		K = np.eye(3)
		K[0][0] = focal
		K[1][1] = focal
		K[0][2] = w/2.0
		K[1][2] = h/2.0
		K = K.tolist()

		Rot = np.eye(3)
		Rot[0, 0] = 1
		Rot[1, 1] = -1
		Rot[2, 2] = -1        
		pose[:3, :3] = np.dot(pose[:3, :3], Rot)

		pose = pose.tolist()
		out = {}
		out["intrinsics"] = K
		out["pose"] = pose
		out["angle"] = (angle * torch.tensor([1, -1, 1])).flatten().tolist()
		# intrinsics = np.asarray(K).reshape(-1)
		# pose = np.asarray(pose).reshape(-1)
		# out = np.concatenate([pose, intrinsics], axis=0).reshape(-1)
		# out = out.astype(np.float32)
		# print(out.shape)
		# print(out)
		return out

	def __extract(self, image_pil, image_name):
		lm_np = get_landmark(image_pil)
		# print(lm_np.shape)
		coeff_3dmm = self.model_3dmm.get_3dmm([image_pil], [lm_np])
		self.crop(image_pil, lm_np, image_name)
		# print(coeff_3dmm.shape)
		camera_parameter = self.cal_camera(coeff_3dmm)
		camera = process_camera(pose=camera_parameter['pose'], intrinsics=camera_parameter['intrinsics'])
		np.save(os.path.join(self.c_outdir, f'{image_name}.npy'), camera)
		# print(camera)
		return camera
		

	def extract(self, image_path):
		image_pil = Image.open(image_path).convert("RGB")  # prevent png exist channel error
		image_name = os.path.basename(image_path).split('.')[0]
		camera = self.__extract(image_pil, image_name)
		
		return camera

	def flip_yaw(self, pose_matrix):
		flipped = copy.deepcopy(pose_matrix)
		flipped[0, 1] *= -1
		flipped[0, 2] *= -1
		flipped[1, 0] *= -1
		flipped[2, 0] *= -1
		flipped[0, 3] *= -1
		return flipped

	def _cal_mirror_c(self, c):
		# c = c[0]
		pose, intrinsics = c[:16].reshape(4,4), c[16:].reshape(3, 3)
		flipped_pose = self.flip_yaw(pose)
		mirror_c = np.concatenate([flipped_pose.reshape(-1), intrinsics.reshape(-1)])
		assert mirror_c.shape == c.shape
		return mirror_c

	def cal_mirror_c(self, image_path):
		image_name = os.path.basename(image_path).split('.')[0]
		
		crop_image_path = os.path.join(self.crop_outdir, f'{image_name}.{self.mode}')
		crop_image_pil = Image.open(crop_image_path).convert("RGB")
		mirror_image_path = os.path.join(self.crop_outdir, f'{image_name}_m.{self.mode}')

		crop_image_pil.transpose(Image.FLIP_LEFT_RIGHT).save(mirror_image_path)

		camera = np.load(os.path.join(self.c_outdir, f'{image_name}.npy'))
		mirror_camera = self._cal_mirror_c(camera)
		np.save(os.path.join(self.c_outdir, f'{image_name}_m.npy'), mirror_camera)
