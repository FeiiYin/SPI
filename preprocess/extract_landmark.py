from PIL import Image
import os
import tqdm
import glob
import face_alignment
import numpy as np
import torch


detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D)


def get_landmark(image):
    """
    :param images: PIL
    :return: numpy (1, 68)
    """
    lm = detector.get_landmarks_from_image(np.array(image))
    assert lm is not None, 'No face detect error!'
    # lm_np = np.expand_dims(lm[0], axis=0)
    return lm[0]


def extract_landmark(input_dir, output_dir, mode='png'):
    os.makedirs(output_dir, exist_ok=True)
    # root = '/apdcephfs_cq2/share_1290939/feiiyin/InvEG3D_result/mpti_output/pti_input/cropped/'
    # out = '/apdcephfs_cq2/share_1290939/feiiyin/InvEG3D_result/mpti_output/pti_input/mask/'

    image_list = sorted(glob.glob(f'{input_dir}/*.{mode}'))
    # for image_path in tqdm.tqdm(image_list):
    for image_path in image_list:
        image_name = os.path.basename(image_path).split('.')[0]
        image = Image.open(image_path).convert("RGB").resize((256, 256))
        lm = get_landmark(image)
        np.save(os.path.join(output_dir, image_name + '.npy'), lm)


# extract_landmark(input_dir='/apdcephfs_cq2/share_1290939/feiiyin/InvEG3D_result/mpti_output/pti_input/hua_2/cropped', 
#                 output_dir='/apdcephfs_cq2/share_1290939/feiiyin/InvEG3D_result/mpti_output/pti_input/hua_2/lm/')