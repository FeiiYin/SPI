SPI

1. Pre-process the image

```
python preprocess/run_total.py --input_root=input_root --output_root=output_root --mode=jpg
```

2. Inversion

PTI
```
CUDA_VISIBLE_DEVICES=1 python spi/run_inversion.py \
 --first_inv_type=sg --first_inv_steps=500 \
 --G_1_type=pti --G_1_step=1000 \
 --not_use_wandb \
 --data_root=./test/dataset/ --output_root=./test/output/ --data_mode=jpg
```

Ours
```
CUDA_VISIBLE_DEVICES=0 python spi/run_inversion.py \
 --first_inv_type=mir --first_inv_steps=500 \
 --G_1_type=RotBbox --G_1_step=1000 \
 --pt_rot_lambda=0.1 --pt_mirror_rot_lambda=0.05 --pt_depth_lambda=1 --pt_tv_lambda=0 \
 --not_use_wandb \
 --data_root=./test/dataset/ --output_root=./test/output/ --data_mode=jpg
```

3. Editing
TODO