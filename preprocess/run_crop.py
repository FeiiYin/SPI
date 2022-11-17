import os
import sys
sys.path.append(os.path.abspath('.'))
import glob
import tqdm


from preprocess.extract_camera import CameraExtractor

root = '/apdcephfs_cq2/share_1290939/feiiyin/InvEG3D_result/mpti_output/pti_input/hua_2/frames/'
out = '/apdcephfs_cq2/share_1290939/feiiyin/InvEG3D_result/mpti_output/pti_input/hua_2/cropped/'
os.makedirs(out, exist_ok=True)
extractor = CameraExtractor(outdir=out)
# root = '/apdcephfs_cq2/share_1290939/feiiyin/InvEG3D_result/personal/origin/barack_obama/'



image_list = sorted(glob.glob(f'{root}/*.jpg'))
for image_path in tqdm.tqdm(image_list):
    print(image_path)
    extractor.extract(image_path)