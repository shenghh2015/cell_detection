# Mar 4, 2021
# JOB: python train_mask.py --gpu 0 --net_type Unet --backbone efficientnetb7 --epoch 2400 --dim 1024 --batch_size 1 --lr 1e-4 --dataset wbc2_1024x1024
# JOB: python train_mask.py --gpu 1 --net_type Unet --backbone efficientnetb7 --epoch 2400 --dim 1024 --batch_size 1 --lr 1e-4 --dataset wbc3_1024x1024
# JOB: python train_mask.py --gpu 2 --net_type Unet --backbone efficientnetb6 --epoch 2400 --dim 1024 --batch_size 1 --lr 1e-4 --dataset wbc2_1024x1024
# JOB: python train_mask.py --gpu 3 --net_type Unet --backbone efficientnetb6 --epoch 2400 --dim 1024 --batch_size 1 --lr 1e-4 --dataset wbc3_1024x1024

# JOB: python train_mask2.py --gpu 0 --net_type Unet --backbone efficientnetb5 --epoch 2400 --dim 1024 --batch_size 2 --lr 1e-4 --dataset wbc2_1024x1024
# JOB: python train_mask2.py --gpu 1 --net_type Unet --backbone efficientnetb5 --epoch 2400 --dim 1024 --batch_size 2 --lr 1e-4 --dataset wbc3_1024x1024
# JOB: python train_mask2.py --gpu 2 --net_type Unet --backbone efficientnetb4 --epoch 2400 --dim 1024 --batch_size 2 --lr 1e-4 --dataset wbc2_1024x1024
# JOB: python train_mask2.py --gpu 3 --net_type Unet --backbone efficientnetb4 --epoch 2400 --dim 1024 --batch_size 2 --lr 1e-4 --dataset wbc3_1024x1024

# Mar 5, 2021
# JOB: python train_crop_mask.py --gpu 0 --net_type Unet --backbone efficientnetb7 --epoch 4800 --dim 384 --dataset wbc_1024x1024 --loss focal+dice --batch 8 --lr 1e-4 --focal_weight 4 --bk 1 --rot 20
# JOB: python train_crop_mask.py --gpu 1 --net_type Unet --backbone efficientnetb7 --epoch 4800 --dim 384 --dataset wbc2_1024x1024 --loss focal+dice --batch 8 --lr 1e-4 --focal_weight 4 --bk 1 --rot 20
# JOB: python train_crop_mask.py --gpu 2 --net_type Unet --backbone efficientnetb7 --epoch 4800 --dim 384 --dataset wbc3_1024x1024 --loss focal+dice --batch 8 --lr 1e-4 --focal_weight 4 --bk 1 --rot 20
# JOB: python train_crop_mask.py --gpu 3 --net_type Unet --backbone efficientnetb6 --epoch 4800 --dim 384 --dataset wbc_1024x1024 --loss focal+dice --batch 8 --lr 1e-4 --focal_weight 4 --bk 1 --rot 20

# JOB: python train_crop_mask.py --gpu 0 --net_type Unet --backbone efficientnetb7 --epoch 4800 --dim 384 --dataset wbc_1024x1024 --loss focal --batch 8 --lr 1e-4 --focal_weight 4 --bk 1 --rot 20
# JOB: python train_crop_mask.py --gpu 1 --net_type Unet --backbone efficientnetb7 --epoch 4800 --dim 384 --dataset wbc3_1024x1024 --loss focal  --batch 8 --lr 1e-4 --focal_weight 4 --bk 1 --rot 20
# JOB: python train_crop_mask.py --gpu 2 --net_type Unet --backbone efficientnetb6 --epoch 4800 --dim 384 --dataset wbc_1024x1024 --loss focal --batch 8 --lr 1e-4 --focal_weight 4 --bk 1 --rot 20
# JOB: python train_crop_mask.py --gpu 3 --net_type Unet --backbone efficientnetb6 --epoch 4800 --dim 384 --dataset wbc3_1024x1024 --loss focal --batch 8 --lr 1e-4 --focal_weight 4 --bk 1 --rot 20

JOB: python train_crop_mask.py --gpu 0 --net_type Unet --backbone efficientnetb6 --epoch 4800 --dim 384 --dataset wbc_1024x1024 --loss focal+dice --batch 8 --lr 1e-4 --focal_weight 4 --bk 1 --rot 25
JOB: python train_crop_mask.py --gpu 1 --net_type Unet --backbone efficientnetb6 --epoch 4800 --dim 384 --dataset wbc3_1024x1024 --loss focal+dice --batch 8 --lr 1e-4 --focal_weight 4 --bk 1 --rot 25
JOB: python train_crop_mask.py --gpu 2 --net_type Unet --backbone efficientnetb7 --epoch 4800 --dim 384 --dataset wbc_1024x1024 --loss focal+dice --batch 8 --lr 1e-4 --focal_weight 4 --bk 1 --rot 25
JOB: python train_crop_mask.py --gpu 3 --net_type Unet --backbone efficientnetb7 --epoch 4800 --dim 384 --dataset wbc3_1024x1024 --loss focal+dice --batch 8 --lr 1e-4 --focal_weight 4 --bk 1 --rot 25