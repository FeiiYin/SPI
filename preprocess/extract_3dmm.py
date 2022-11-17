from email.mime import image
import os
import face_alignment
from third_part.Deep3DFaceRecon_pytorch.options.test_options import TestOptions
# from data import create_dataset
from third_part.Deep3DFaceRecon_pytorch.models import create_model
# from third_part.Deep3DFaceRecon_pytorch.util.visualizer import MyVisualizer
from third_part.Deep3DFaceRecon_pytorch.util.preprocess import align_img
from PIL import Image
import numpy as np
from third_part.Deep3DFaceRecon_pytorch.util.load_mats import load_lm3d
import torch


# calculating least square problem for image alignment
def POS(xp, x):
	npts = xp.shape[1]

	A = np.zeros([2*npts, 8])

	A[0:2*npts-1:2, 0:3] = x.transpose()
	A[0:2*npts-1:2, 3] = 1

	A[1:2*npts:2, 4:7] = x.transpose()
	A[1:2*npts:2, 7] = 1

	b = np.reshape(xp.transpose(), [2*npts, 1])

	k, _, _, _ = np.linalg.lstsq(A, b)

	R1 = k[0:3]
	R2 = k[4:7]
	sTx = k[3]
	sTy = k[7]
	s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
	t = np.stack([sTx, sTy], axis=0)

	return t, s

def extract_5p(lm):
	lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
	lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
		lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
	lm5p = lm5p[[1, 2, 0, 3, 4], :]
	return lm5p

# resize and crop images for face reconstruction
def resize_n_crop_img(img, lm, t, s, target_size=1024., mask=None):
	w0, h0 = img.size
	w = (w0*s).astype(np.int32)
	h = (h0*s).astype(np.int32)
	left = (w/2 - target_size/2 + float((t[0] - w0/2)*s)).astype(np.int32)
	right = left + target_size
	up = (h/2 - target_size/2 + float((h0/2 - t[1])*s)).astype(np.int32)
	below = up + target_size
	img = img.resize((w, h), resample=Image.LANCZOS)
	img = img.crop((left, up, right, below))

	if mask is not None:
		mask = mask.resize((w, h), resample=Image.LANCZOS)
		mask = mask.crop((left, up, right, below))

	lm = np.stack([lm[:, 0] - t[0] + w0/2, lm[:, 1] -
				  t[1] + h0/2], axis=1)*s
	lm = lm - np.reshape(
			np.array([(w/2 - target_size/2), (h/2-target_size/2)]), [1, 2])
	return img, lm, mask


# utils for face reconstruction
def align_img(img, lm, lm3D, mask=None, target_size=1024., rescale_factor=466.285):
	"""
	Return:
		transparams        --numpy.array  (raw_W, raw_H, scale, tx, ty)
		img_new            --PIL.Image  (target_size, target_size, 3)
		lm_new             --numpy.array  (68, 2), y direction is opposite to v direction
		mask_new           --PIL.Image  (target_size, target_size)
	
	Parameters:
		img                --PIL.Image  (raw_H, raw_W, 3)
		lm                 --numpy.array  (68, 2), y direction is opposite to v direction
		lm3D               --numpy.array  (5, 3)
		mask               --PIL.Image  (raw_H, raw_W, 3)
	"""

	w0, h0 = img.size
	if lm.shape[0] != 5:
		lm5p = extract_5p(lm)
	else:
		lm5p = lm

	# calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
	t, s = POS(lm5p.transpose(), lm3D.transpose())
	s = rescale_factor/s

	# processing the image
	img_new, lm_new, mask_new = resize_n_crop_img(img, lm, t, s, target_size=target_size, mask=mask)
	# img.save("/home/koki/Projects/Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/iphone/epoch_20_000000/img_new.jpg")    
	trans_params = np.array([w0, h0, s, t[0], t[1]])
	lm_new *= 224/1024.0
	img_new_low = img_new.resize((224, 224), resample=Image.LANCZOS)

	return trans_params, img_new_low, lm_new, mask_new, img_new


class Extract3dmm:

    def __init__(self, PRETRAINED_MODELS_PATH):
        bfm_path = PRETRAINED_MODELS_PATH['BFM']
        deep3d_path = PRETRAINED_MODELS_PATH['3DMM']
        deep3d_path = os.path.dirname(deep3d_path)
        deep3d_dir = os.path.dirname(deep3d_path)
        deep3d_name = os.path.basename(deep3d_path)
        cmd = f'--checkpoints_dir {deep3d_dir} ' \
            f'--bfm_folder {bfm_path} --name={deep3d_name} ' \
            f'--epoch=20  --img_folder=temp'
        opt = TestOptions(cmd_line=cmd).parse()  # get test options

        self.model = create_model(opt)
        self.model.setup(opt)
        self.model.device = 'cuda'
        self.model.parallelize()
        self.model.eval()
        self.lm3d_std = load_lm3d(opt.bfm_folder)

    def image_transform(self, images, lm):
        # W, H = images.size
        # W, H = 256, 256
        # imsize = 256  # Note this hyper-param is key for downloading optical model
        # images = images.resize((imsize, imsize))
        # lm = lm * imsize / W  # lm coordinate is corresponding to the image size
        # lm = lm.copy()  # Note that lm has been extracted at the size of 256
        _, H = images.size
        # if np.mean(lm) == -1:
        #     lm = (self.lm3d_std[:, :2] + 1) / 2.
        #     lm = np.concatenate(
        #         [lm[:, :1] * W, lm[:, 1:2] * H], 1
        #     )
        # else:
        lm = lm.reshape(-1, 2)
        lm[:, -1] = H - 1 - lm[:, -1]

        rescale_factor = 466.285
        # _, img, lm, img_high = align_img(images, lm, self.lm3d_std, rescale_factor=rescale_factor)
        _, im_pil, lm, _, im_high = align_img(images, lm, self.lm3d_std, rescale_factor=rescale_factor)

        img = torch.tensor(np.array(im_pil) / 255., dtype=torch.float32).permute(2, 0, 1)
        lm = torch.tensor(lm)
        # trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)])
        # trans_params = torch.tensor(trans_params.astype(np.float32))

        return img, lm  #, trans_params

    def get_3dmm(self, images_pil, lms_np):
        """
        :param images: PIL list
        :return:
        """
        images = []
        # trans_params = []
        lms = []
        for i, img in enumerate(images_pil):
            lm = lms_np[i]
            img_tensor, lm_tensor = self.image_transform(img, lm)
            lms.append(lm_tensor)
            images.append(img_tensor)
            # trans_params.append(p)

        images = torch.stack(images)
        lms = torch.stack(lms)
        # trans_params = torch.stack(trans_params)

        batch_size = 20
        num_batch = images.shape[0] // batch_size + 1
        pred_coeffs = []
        for _i in range(num_batch):
            _images = images[_i * batch_size: (_i+1) * batch_size]
            _lms = lms[_i * batch_size: (_i+1) * batch_size]

            if len(_images) == 0:
                break
            data_input = {
                'imgs': _images,
                'lms': _lms
            }
            self.model.set_input(data_input)
            with torch.no_grad():
                self.model.test()
            pred_coeff = {key: self.model.pred_coeffs_dict[key] for key in self.model.pred_coeffs_dict}
            # pred_coeff = torch.cat([
            #     pred_coeff['id'],
            #     pred_coeff['exp'],
            #     pred_coeff['tex'],
            #     pred_coeff['angle'],
            #     pred_coeff['gamma'],
            #     pred_coeff['trans']], 1
            # )
            # _trans_params = np.array(trans_params[_i * batch_size: (_i+1) * batch_size])
            # _, _, ratio, t0, t1 = np.hsplit(_trans_params, 5)
            # crop_param = np.concatenate([ratio, t0, t1], 1)
            # pred_coeff = np.concatenate([pred_coeff.cpu().numpy(), crop_param], 1)

            pred_coeffs.append(pred_coeff)

        coeff_3dmm = pred_coeffs[0]
        for key in coeff_3dmm:
            coeff_3dmm[key] = coeff_3dmm[key].cpu()
            for _i in range(1, len(pred_coeffs)):
                coeff_3dmm[key] = torch.cat(coeff_3dmm[key], pred_coeffs[_i][key].cpu())

        # coeff_3dmm = np.concatenate(pred_coeffs, 0)

        # # extract 73 feature from 260
        # # id_coeff = coeff_3dmm[:,:80] #identity
        # ex_coeff = coeff_3dmm[:, 80:144]  # expression
        # # tex_coeff = coeff_3dmm[:,144:224] #texture
        # angles = coeff_3dmm[:, 224:227]  # euler angles for pose
        # # gamma = coeff_3dmm[:,227:254] #lighting
        # translation = coeff_3dmm[:, 254:257]  # translation
        # crop = coeff_3dmm[:, 257:300]  # crop param
        # coeff_3dmm = np.concatenate([ex_coeff, angles, translation, crop], 1)
        return coeff_3dmm


