python train_crop_mask.py --gpu 0 --net_type Unet --backbone efficientnetb6 --epoch 4800 --dim 384 --dataset wbc_1024x1024 --loss focal+dice --batch 8 --lr 5e-4 --focal_weight 4 --bk 1 --rot 10 --max_min max --valid False

