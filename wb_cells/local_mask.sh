python train_crop_mask.py --gpu 2 --net_type Unet --backbone efficientnetb1 --epoch 2400 --dim 384 --dataset wbc_1024x1024 --loss focal+dice --batch 10 --lr 1e-4 --focal_weight 4 --bk 1
python train_crop_mask.py --gpu 0 --net_type Unet --backbone efficientnetb2 --epoch 2400 --dim 384 --dataset wbc2_1024x1024 --loss focal+dice --batch 8 --lr 1e-4 --focal_weight 4 --bk 1
python train_crop_mask.py --gpu 1 --net_type Unet --backbone efficientnetb2 --epoch 2400 --dim 384 --dataset wbc3_1024x1024 --loss focal+dice --batch 8 --lr 1e-4 --focal_weight 4 --bk 1

# Mar 7, 2021
python train_crop_mask.py --gpu 0 --net_type FPN --backbone efficientnetb1 --epoch 2400 --dim 352 --dataset wbc_1024x1024 --loss focal+dice --batch 10 --lr 1e-3 --focal_weight 4 --bk 1
python train_crop_mask.py --gpu 1 --net_type Unet --backbone efficientnetb1 --epoch 2400 --dim 352 --dataset wbc_1024x1024 --loss focal+dice --batch 10 --lr 1e-4 --focal_weight 4 --bk 1

# Mar 8, 2021
python train_crop_mask.py --gpu 0 --net_type FPN --backbone efficientnetb1 --epoch 2400 --dim 352 --dataset wbc_1024x1024 --loss focal+dice --reduce_factor 0.8 --batch 10 --lr 1e-4 --focal_weight 4 --bk 1 --max_min max
python train_crop_mask.py --gpu 1 --net_type FPN --backbone efficientnetb0 --epoch 2400 --dim 352 --dataset wbc3_1024x1024 --loss focal+dice --reduce_factor 0.8 --batch 10 --lr 1e-4 --focal_weight 4 --bk 1 --max_min max
python train_crop_mask.py --gpu 2 --net_type AtUnet --backbone efficientnetb0 --epoch 2400 --dim 384 --dataset wbc3_1024x1024 --loss focal+dice --reduce_factor 0.8 --batch 10 --lr 1e-4 --focal_weight 4 --bk 1 --max_min max

# Mar 9, 2021
# python train_crop_mask.py --gpu 0 --net_type unet1 --backbone efficientnetb0 --epoch 2400 --dim 384 --dataset wbc_1024x1024 --loss focal+dice --reduce_factor 0.95 --batch 10 --lr 1e-4 --focal_weight 4 --bk 1 --max_min max
# python train_crop_mask.py --gpu 1 --net_type unet1 --filters 32 --backbone efficientnetb0 --epoch 2400 --dim 384 --dataset wbc3_1024x1024 --loss bce --reduce_factor 0.95 --batch 10 --lr 1e-4 --focal_weight 4 --bk 1 --max_min max
# python train_crop_mask.py --gpu 2 --net_type unet1 --backbone efficientnetb0 --epoch 2400 --dim 384 --dataset wbc3_1024x1024 --loss focal+dice --reduce_factor 0.95 --batch 10 --lr 1e-4 --focal_weight 4 --bk 1 --max_min max

# Mar 9, 2021
python train_crop_mask.py --gpu 0 --net_type Unet --backbone efficientnetb0 --epoch 2400 --dim 384 --dataset wbc_1024x1024 --loss focal+dice --batch 16 --lr 1e-4 --focal_weight 4 --bk 1 --max_min max --valid False
python train_crop_mask.py --gpu 1 --net_type Unet --backbone efficientnetb0 --epoch 2400 --dim 384 --dataset wbc2_1024x1024 --loss focal+dice --batch 16 --lr 1e-4 --focal_weight 4 --bk 1 --max_min max --valid False
python train_crop_mask.py --gpu 2 --net_type Unet --backbone efficientnetb0 --epoch 2400 --dim 384 --dataset wbc_1024x1024 --loss focal+dice --batch 32 --lr 1e-4 --focal_weight 4 --bk 1 --max_min max --valid False

python train_crop_mask.py --docker False --gpu 6 --net_type Unet --backbone efficientnetb0 --epoch 2400 --dim 384 --dataset wbc4_1024x1024 --loss focal+dice --batch 16 --lr 1e-4 --focal_weight 4 --bk 1 --max_min max --valid False
python train_crop_mask.py --docker False --gpu 0 --net_type Unet --backbone efficientnetb0 --epoch 2400 --dim 384 --dataset wbc4_1024x1024 --loss focal+dice --batch 16 --lr 1e-4 --focal_weight 4 --bk 1 --max_min max --valid True