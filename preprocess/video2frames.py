import os
from PIL import Image
import cv2


def read_video(filename, uplimit=1000):
    frames = []
    cap = cv2.VideoCapture(filename)
    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (512, 512))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            break
        cnt += 1
        if cnt >= uplimit:
            break
    cap.release()
    assert len(frames) > 0, f'{filename}: video with no frames!'
    return frames



def video2frames(video_path, output_dir, mode, prefix=None):
    os.makedirs(output_dir, exist_ok=True)

    start_i = 0
    frames = read_video(video_path)
    for _i in range(len(frames)):
        path = os.path.join(output_dir, f'{start_i:03d}.{mode}')
        if prefix is not None:
            path = os.path.join(output_dir, f'{prefix}#{start_i:03d}.{mode}')
        Image.fromarray(frames[_i]).save(path)
        start_i += 1


def video2frames_mirror(video_path, output_dir, mode):
    os.makedirs(output_dir, exist_ok=True)

    start_i = 0
    frames = read_video(video_path)
    for _i in range(len(frames)):
        Image.fromarray(frames[_i]).save(os.path.join(output_dir, f'{start_i:03d}.{mode}'))
        start_i += 1

    frames = read_video(video_path)
    for _i in range(len(frames)):
        Image.fromarray(frames[_i]).transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(output_dir, f'{start_i:03d}.{mode}'))
        start_i += 1




def main():
    input_path = '/apdcephfs_cq2/share_1290939/feiiyin/InvEG3D_result/mpti_output/pti_input/hua_2'
    output_dir = '/apdcephfs_cq2/share_1290939/feiiyin/InvEG3D_result/mpti_output/pti_input/hua_2/frames'

    os.makedirs(output_dir, exist_ok=True)

    # videos = [os.path.join(input_path, 'hua_gen.mp4'), os.path.join(input_path, 'hua_flip_shape.mp4')]
    videos = [os.path.join(input_path, 'hua_flip_shape.mp4')]

    start_i = 0
    for video_path in videos:
        print(video_path)
        frames = read_video(video_path)
        for _i in range(len(frames)):
            Image.fromarray(frames[_i]).save(os.path.join(output_dir, f'{start_i:03d}.jpg'))
            start_i += 1
