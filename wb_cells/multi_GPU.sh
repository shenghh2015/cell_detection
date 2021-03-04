# Mar 4, 2021
# JOB: python train_mask.py --gpu 0 --net_type Unet --backbone efficientnetb7 --epoch 2400 --dim 1024 --batch_size 1 --lr 1e-4 --dataset wbc2_1024x1024
# JOB: python train_mask.py --gpu 1 --net_type Unet --backbone efficientnetb7 --epoch 2400 --dim 1024 --batch_size 1 --lr 1e-4 --dataset wbc3_1024x1024
# JOB: python train_mask.py --gpu 2 --net_type Unet --backbone efficientnetb6 --epoch 2400 --dim 1024 --batch_size 1 --lr 1e-4 --dataset wbc2_1024x1024
# JOB: python train_mask.py --gpu 3 --net_type Unet --backbone efficientnetb6 --epoch 2400 --dim 1024 --batch_size 1 --lr 1e-4 --dataset wbc3_1024x1024

JOB: python train_mask2.py --gpu 0 --net_type Unet --backbone efficientnetb7 --epoch 2400 --dim 1024 --batch_size 2 --lr 1e-4 --dataset wbc2_1024x1024
JOB: python train_mask2.py --gpu 1 --net_type Unet --backbone efficientnetb7 --epoch 2400 --dim 1024 --batch_size 2 --lr 1e-4 --dataset wbc3_1024x1024
JOB: python train_mask2.py --gpu 2 --net_type Unet --backbone efficientnetb6 --epoch 2400 --dim 1024 --batch_size 2 --lr 1e-4 --dataset wbc2_1024x1024
JOB: python train_mask2.py --gpu 3 --net_type Unet --backbone efficientnetb6 --epoch 2400 --dim 1024 --batch_size 2 --lr 1e-4 --dataset wbc3_1024x1024