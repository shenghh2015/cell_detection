python train_crop_mask.py --gpu 2 --net_type Unet --backbone efficientnetb1 --epoch 2400 --dim 384 --dataset wbc_1024x1024 --loss focal+dice --batch 10 --lr 1e-4 --focal_weight 4 --bk 1
python train_crop_mask.py --gpu 0 --net_type Unet --backbone efficientnetb2 --epoch 2400 --dim 384 --dataset wbc2_1024x1024 --loss focal+dice --batch 8 --lr 1e-4 --focal_weight 4 --bk 1
python train_crop_mask.py --gpu 1 --net_type Unet --backbone efficientnetb2 --epoch 2400 --dim 384 --dataset wbc3_1024x1024 --loss focal+dice --batch 8 --lr 1e-4 --focal_weight 4 --bk 1

# Mar 7, 2021
python train_crop_mask.py --gpu 0 --net_type FPN --backbone efficientnetb1 --epoch 2400 --dim 352 --dataset wbc_1024x1024 --loss focal+dice --batch 10 --lr 1e-3 --focal_weight 4 --bk 1
python train_crop_mask.py --gpu 1 --net_type Unet --backbone efficientnetb1 --epoch 2400 --dim 352 --dataset wbc_1024x1024 --loss focal+dice --batch 10 --lr 1e-4 --focal_weight 4 --bk 1