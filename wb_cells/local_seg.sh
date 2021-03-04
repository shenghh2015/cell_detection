python train_seg.py --docker True --gpu 0 --net_type Unet --backbone efficientnetb0 --epoch 2400 --dim 1024 --dataset wbc2_1024x1024 --batch 2 --lr 1e-4 --focal_weight 4 --bk 1
python train_seg.py --docker True --gpu 0 --net_type Unet --backbone efficientnetb0 --epoch 2400 --dim 1024 --dataset wbc2_1024x1024 --loss ce --batch 2 --lr 1e-4 --focal_weight 4 --bk 1
python train_seg.py --docker True --gpu 1 --net_type Unet --backbone efficientnetb1 --epoch 2400 --dim 928 --dataset wbc2_1024x1024 --batch 2 --lr 1e-4 --focal_weight 4 --bk 1
python train_seg.py --docker True --gpu 1 --net_type Unet --backbone efficientnetb1 --epoch 2400 --dim 928 --dataset wbc_1024x1024 --loss ce --batch 2 --lr 1e-4 --focal_weight 4 --bk 1

# Feb 23, 2021
python train_crop.py --docker True --gpu 1 --net_type Unet --backbone efficientnetb0 --epoch 2400 --dim 352 --rot 20 --dataset wbc_1024x1024 --loss ce --batch 8 --lr 1e-4 --focal_weight 4 --bk 1 --crop True
python train_crop.py --docker True --gpu 0 --net_type Unet --backbone efficientnetb0 --epoch 2400 --dim 352 --rot 20 --dataset wbc_1024x1024 --loss focal+dice --batch 8 --lr 1e-4 --focal_weight 4 --bk 1 --crop True

# Feb 27, 2021
python train_crop.py --docker True --gpu 0 --net_type Unet --backbone efficientnetb0 --epoch 2400 --dim 352 --rot 20 --dataset wbc_1024x1024 --loss focal+dice --batch 10 --lr 1e-4 --focal_weight 4 --bk 1 --crop True --cls 2
python train_crop.py --docker True --gpu 1 --net_type Unet --backbone efficientnetb0 --epoch 2400 --dim 352 --rot 20 --dataset wbc_1024x1024 --loss bce --batch 10 --lr 1e-4 --focal_weight 4 --bk 1 --crop True --cls 2

# Mar 3, 2021
python train_mask2.py --docker True --gpu 0 --net_type Unet --backbone efficientnetb0 --epoch 2400 --dim 1024 --dataset wbc_1024x1024 --loss focal+dice --batch 2 --lr 1e-4 --focal_weight 4 --bk 1
python train_mask2.py --docker True --gpu 1 --net_type Unet --backbone efficientnetb0 --epoch 2400 --dim 1024 --dataset wbc2_1024x1024 --loss focal+dice --batch 2 --lr 1e-4 --focal_weight 4 --bk 1
python train_mask2.py --docker True --gpu 2 --net_type Unet --backbone efficientnetb0 --epoch 2400 --dim 1024 --dataset wbc3_1024x1024 --loss focal+dice --batch 2 --lr 1e-4 --focal_weight 4 --bk 1

# Mar 4, 2021
python train_mask2.py --docker True --gpu 0 --net_type Unet --backbone resnet50 --epoch 2400 --dim 1024 --dataset wbc2_1024x1024 --loss focal+dice --batch 2 --lr 1e-4 --focal_weight 4 --bk 1