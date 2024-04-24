import os
import json
import glob
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--input_root', type=str, default=None)
    parser.add_argument('--output_root', type=str, default=None)
    args = parser.parse_args()
    return args

args = parse_args()
spi_output_root = args.input_root # "/weka/home-feiyin/code/GOAE/example/spi_output"
output_root = args.output_root # "example/input"
os.makedirs(output_root, exist_ok=True)

image_path_list = sorted(glob.glob(os.path.join(spi_output_root, "crop", "*")))
output_json = {"labels": []}

for image_root in image_path_list:
    image_name = os.path.basename(image_root)
    mode = "jpg"
    image_path = os.path.join(image_root, f"target.{mode}")
    if not os.path.exists(image_path):
        mode = "png"
        image_path = os.path.join(image_root, f"target.{mode}")

    target_path = os.path.join(output_root, f"{image_name}.{mode}")
    cmd = f"cp {image_path} {target_path}"
    os.system(cmd)

    camera_path = os.path.join(spi_output_root, "c", image_name, "target.npy")
    camera = np.load(camera_path)
    
    output_json["labels"].append([f"{image_name}.{mode}", camera.tolist()])

output_json_path = os.path.join(output_root, "label.json")
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(output_json, f, ensure_ascii=False, indent=4)

