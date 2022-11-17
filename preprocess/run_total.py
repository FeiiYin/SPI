import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./eg3d/'))
import argparse
import glob
import tqdm

from preprocess.extract_camera import CameraExtractor
from preprocess.extract_landmark import extract_landmark
from preprocess.extract_mask import extract_mask
from preprocess.video2frames import video2frames, video2frames_mirror


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--input_root', type=str, default='./test/images/')
    parser.add_argument('--output_root', type=str, default='./test/dataset/')
    parser.add_argument('--mode', type=str, default='jpg')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # Real dataset
    input_dir = args.input_root
    root = args.output_root

    image_input = os.path.join(root, f'input')
    c_out = os.path.join(root, f'c')
    crop_out = os.path.join(root, f'crop')
    lm_out = os.path.join(root, f'lm')
    mask_out = os.path.join(root, f'mask')

    for out in [image_input, c_out, crop_out, lm_out, mask_out]:
        os.makedirs(out, exist_ok=True)

    extractor = CameraExtractor(crop_outdir=None, c_outdir=None, mode=None)

    mode = args.mode
    image_list = sorted(glob.glob(f'{input_dir}/*.{mode}'))

    # filter_name_list = ['taylor_swift_4']
    # image_list = [f'{input_dir}/{name_i}.{mode}' for name_i in filter_name_list]

    # Transform the png into jpg
    # for image_path in tqdm.tqdm(image_list):
    #     Image.open(image_path).convert('RGB').save(image_path.replace(mode, 'png'))

    for image_path in tqdm.tqdm(image_list):
        try:
            name = os.path.basename(image_path).split('.')[0]
            
            frame_root = os.path.join(image_input, name)
            os.makedirs(frame_root, exist_ok=True)
            
            frame_crop_path = os.path.join(crop_out, name, f'target.{mode}')
            if not os.path.exists(frame_crop_path):
                os.system(f'cp {image_path} {frame_root}/target.{mode}')
            
            # video2frames
            # video_path = os.path.join(video_dir, f'{name}.mp4')
            # if os.path.exists(video_path):
            #     os.system(f'cp {video_path} {video_input}/{name}.mp4')
            #     video2frames(video_path=video_path, output_dir=frame_root, mode=mode)
            
            # Crop & C
            _crop_outdir = os.path.join(crop_out, name)
            _c_outdir = os.path.join(c_out, name)
            os.makedirs(_crop_outdir, exist_ok=True)
            os.makedirs(_c_outdir, exist_ok=True)
            extractor.set_path(crop_outdir=_crop_outdir, c_outdir=_c_outdir, mode=mode)
            
            frame_list = sorted(glob.glob(f'{frame_root}/*.{mode}'))
            for f in frame_list:
                extractor.extract(f)

            # Lm
            _lm_outdir = os.path.join(lm_out, name)
            os.makedirs(_lm_outdir, exist_ok=True)
            extract_landmark(input_dir=_crop_outdir, output_dir=_lm_outdir, mode=mode)

            # Mask
            _mask_outdir = os.path.join(mask_out, name)
            os.makedirs(_mask_outdir, exist_ok=True)
            extract_mask(input_dir=_crop_outdir, output_dir=_mask_outdir, mode=mode)
        except Exception:
            print(image_path, name)


if __name__ == '__main__':
    main()