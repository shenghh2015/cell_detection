# python3 train_wbc.py --docker --snapshot imagenet --dataset wbc3_1024x1024 --phi 0 --gpu 2 --batch-size 8 --steps 100 --epoch 200
# python3 train_wbc.py --docker --snapshot imagenet --dataset wbc3_1024x1024 --phi 0 --gpu 2 --batch-size 8 --steps 100 --epoch 200 --weighted-bifpn

python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --valid False --phi 0 --gpu 0 --batch-size 8 --steps 100 --epoch 200 --aug False --bstrp 1
python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --valid False --phi 0 --gpu 0 --batch-size 8 --steps 100 --epoch 200 --aug False --bstrp 2
python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --valid False --phi 0 --gpu 0 --batch-size 8 --steps 100 --epoch 200 --aug False --bstrp 3