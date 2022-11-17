import glob
import os
import numpy as np
import PIL.Image as Image
import tqdm
import scipy.ndimage

# input_dir = '/home/jachoon/Desktop/Code/pose-free_3dgan_inversion/*.jpg'
# out_dir = './padded'
input_dir = '/apdcephfs_cq2/share_1290939/feiiyin/dataset/CelebAHQ_test/*.jpg'
out_dir = '/apdcephfs_cq2/share_1290939/feiiyin/dataset/CelebAHQ_test_padding/'
if os.path.isdir(out_dir) == 0:
    os.mkdir(out_dir)

image_list = glob.glob(input_dir)
stripstr = input_dir.split('/')[-1].lstrip('*')
for i in tqdm.tqdm(image_list):
    image_name = i.split('/')[-1].rstrip(stripstr)
    # if image_name not in ['10252', '27065']:
    #     continue
    image = np.array(Image.open(i).convert('RGB'))
    pad_num = 250
    image_pad= np.pad(image, ((pad_num, pad_num),(pad_num, pad_num),(0, 0)), 'reflect')
    h, w, _ = image_pad.shape
    y, x, _ = np.mgrid[:h, :w, :1]
    # mask = np.ones_like(image_pad)[:, :, :1]
    
    # mask[pad_num:pad_num+1024, pad_num:pad_num+1024, :] = np.zeros_like(mask[pad_num:pad_num+1024, pad_num:pad_num+1024, :])
    
    mask = 1.0 - np.minimum(np.minimum(np.float32(x) / pad_num, np.float32(y) / pad_num), np.minimum(np.float32(w-1-x) / pad_num, np.float32(h-1-y) / pad_num))

    #for j in range(1, 20):
        # import pdb; pdb.set_trace()
    image_pad = image_pad.astype(np.float32)
    image_pad += (scipy.ndimage.gaussian_filter(image_pad, [5,5,0]) - image_pad) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
    
    #image_pad += ((np.median(image_pad, axis=(0,1)) - image_pad) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)).astype(np.uint8)
    Image.fromarray(image_pad.astype(np.uint8)).save(out_dir + image_name + '.png')