## SPI: 3D GAN Inversion with Facial Symmetry Prior [CVPR 2023]

> **SPI: 3D GAN Inversion with Facial Symmetry Prior** <br>
> Fei Yin, Yong Zhang, Xuan Wang, Tengfei Wang, Xiaoyu Li, Yuan Gong, Yanbo Fan, Xiaodong Cun, Ying Shan, Cengiz Öztireli, Yujiu Yang <br>
> CVPR 2023 <br>
> [Project](https://feiiyin.github.io/SPI/) <br>
> [Paper](https://arxiv.org/pdf/2211.16927.pdf) <br>

<div align=center>
<img src="./assets/pipeline-2.jpg" width=800px>
</div>

#### Abstract 

Although with the facial prior preserved in pre-trained 3D GANs, 3D GAN Inversion, i.e., reconstructing a 3D portrait with only one monocular image, is still an ill-pose problem. The straightforward application of 2D GAN inversion methods focuses on texture similarity only while ignoring the correctness of 3D geometry shapes. It may raise geometry collapse effects, especially when reconstructing a side face under an extreme pose. Besides, the synthetic results in novel views are prone to be blurry. In this work, we propose a novel method to promote 3D GAN inversion by introducing facial symmetry prior. We design a pipeline and constraints to make full use of the pseudo auxiliary view obtained via image flipping, which helps obtain a view-consistent and well-structured geometry shape during the inversion process. To enhance texture fidelity in unobserved viewpoints, pseudo labels from depth-guided 3D warping can provide extra supervision. We design constraints to filter out conflict areas for optimization in asymmetric situations.

#### Environment

Follow [EG3D](https://github.com/NVlabs/eg3d) to install the environment.


# Quick Start

#### Pretrained Models

Please download our [pre-trained model](https://drive.google.com/drive/folders/1T4aZ65I23gEs6tDkSK5HJfB7yrp_0pLX?usp=sharing) and put it in `./checkpoints`. 3DMM files (BFM/ and epoch_20.pth) can be obtained in this [Link](https://drive.google.com/drive/folders/1BtETxoWlVRoPI6vcySAzimfhemmpc4px).

| Model | Description
| :--- | :----------
|checkpoints/ffhqrebalanced512-128.pkl | Pre-trained EG3D Generator.
|checkpoints/model_ir_se50.pth | Pre-trained arc-face encoder.
|checkpoints/bisenet.pth | Pre-trained face parsing encoder.
|checkpoints/vgg16.pt | Pre-trained VGG-16 network.
|checkpoints/BFM/ | 3DMM related file.
|checkpoints/model_name/epoch_20.pth | Pre-trained Deep-Face-Recon network.

Please download the checkpoint and place them at `./checkpoints/`.
You can also change the path in `spi/configs/path_config.py` with your custom path.

#### Inference

1. Pre-process the image, please run:

```
python preprocess/run_total.py --input_root=input_root --output_root=output_root --mode=jpg
```

2. Inversion

For PTI, please run:
```
CUDA_VISIBLE_DEVICES=1 python spi/run_inversion.py \
 --first_inv_type=sg --first_inv_steps=500 \
 --G_1_type=pti --G_1_step=1000 \
 --not_use_wandb \
 --data_root=./test/dataset/ --output_root=./test/output/ --data_mode=jpg
```

For SPI (Ours), please run:
```
CUDA_VISIBLE_DEVICES=0 python spi/run_inversion.py \
 --first_inv_type=mir --first_inv_steps=500 \
 --G_1_type=RotBbox --G_1_step=1000 \
 --pt_rot_lambda=0.1 --pt_mirror_rot_lambda=0.05 --pt_depth_lambda=1 --pt_tv_lambda=0 \
 --not_use_wandb \
 --data_root=./test/dataset/ --output_root=./test/output/ --data_mode=jpg
```

3. Editing

We employ the ZSSGAN to perform editing. Please change the `PATH_DICT` in `ZSSGAN/criteria/clip_loss.py` to your local CLIP checkpoint first. Then use the following example code to editing.

```
CUDA_VISIBLE_DEVICES=1 python ZSSGAN/train.py --size 512 \
                --batch 2 --n_sample 2 \
                --output_dir OUTPUT_PATH \
                --lr 0.002  \
                --frozen_gen_ckpt EG3D_PATH \
                --iter 301  \
                --source_class "Human"  \
                --target_class "Zombie"  \
                --auto_layer_k 18 \
                --auto_layer_iters 1  \
                --auto_layer_batch 8  \
                --output_interval 50  \
                --clip_models "ViT-B/32" "ViT-B/16"  \
                --clip_model_weights 1.0 1.0  \
                --mixing 0.0 \
                --save_interval 300
```


## Citation
If you find this work useful for your research, please cite:

``` 
@article{yin20223d,
  title={3D GAN Inversion with Facial Symmetry Prior},
  author={Yin, Fei and Zhang, Yong and Wang, Xuan and Wang, Tengfei and Li, Xiaoyu and Gong, Yuan and Fan, Yanbo and Cun, Xiaodong and Shan, Ying and Oztireli, Cengiz and Yang, Yujiu},
  journal={arXiv preprint arXiv:2211.16927},
  year={2022}
}
```

## Acknowledgement
Thanks to 
[EG3D](https://github.com/NVlabs/eg3d), 
[PTI](https://github.com/danielroich/PTI), 
[StyleGAN-NADA](https://github.com/rinongal/StyleGAN-nada), 
for sharing their code.
