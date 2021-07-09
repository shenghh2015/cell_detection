# python3 train_wbc.py --docker --snapshot /data/coco/efficientdet-d0.h5 --dataset wbc_1024x1024 --phi 0 --gpu 0 --batch-size 8 --steps 100 --epoch 200
# python3 train_wbc.py --docker --snapshot /data/coco/efficientdet-d0.h5 --dataset wbc_1024x1024 --phi 0 --gpu 0 --batch-size 8 --steps 100 --epoch 200 --weighted-bifpn
# python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc5_1024x1024 --valid False --phi 0 --gpu 1 --batch-size 8 --steps 100 --epoch 200 --aug False --bstrp 3
# python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc5_1024x1024 --valid False --phi 0 --gpu 1 --batch-size 8 --steps 100 --epoch 200 --aug False --bstrp 4
# python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --valid False --phi 0 --gpu 1 --batch-size 8 --steps 100 --epoch 200 --aug False --bstrp 0
# python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --valid False --phi 0 --gpu 2 --batch-size 8 --steps 100 --epoch 200 --aug False --bstrp 0 --lr 1e-4 --lw 1.0
# python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --valid False --phi 0 --gpu 0 --batch-size 8 --steps 100 --epoch 200 --aug False --bstrp 0 --lr 1e-4 --lw 0.5
# python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --valid False --phi 0 --gpu 1 --batch-size 8 --steps 100 --epoch 200 --aug False --bstrp 0 --lr 4e-5 --lw 0.5
# python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc_1024x1024 --valid False --phi 0 --gpu 0 --batch-size 8 --steps 100 --epoch 200 --aug False --bstrp 0 --lr 4e-5 --lw 0.5

# python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --valid True --phi 0 --gpu 1 --batch-size 8 --steps 100 --epoch 200 --aug False --bstrp 0 --lr 1e-4 --lw 0.5
# python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --valid True --phi 0 --gpu 1 --batch-size 8 --steps 100 --epoch 200 --aug False --bstrp 1 --lr 1e-4 --lw 0.5

# python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --valid True --phi 0 --gpu 1 --batch-size 8 --steps 100 --epoch 200 --aug False --bstrp 2 --lr 1e-4 --lw 0.5
# python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --valid True --phi 0 --gpu 1 --batch-size 8 --steps 100 --epoch 200 --aug False --bstrp 3 --lr 1e-4 --lw 0.5

python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc_1024x1024 --valid True --phi 0 --gpu 1 --batch-size 8 --steps 100 --epoch 200 --aug False --bstrp 1 --lr 5e-5 --lw 0.5 --nb 350
python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc_1024x1024 --valid True --phi 0 --gpu 1 --batch-size 8 --steps 100 --epoch 200 --aug False --bstrp 2 --lr 5e-5 --lw 0.5 --nb 350