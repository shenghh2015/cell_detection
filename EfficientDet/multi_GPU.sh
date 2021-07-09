# Feb 28, 2021
# JOB: python3 train.py --snapshot imagenet --phi 4 --gpu 0 --epochs 200 --snapshot-path /data/results/wbc2/det4 --random-transform --compute-val-loss --freeze-backbone --batch-size 8 --steps 1000 csv /data/datasets/wbc2_1024x1024/train.csv /data/datasets/wbc2_1024x1024/class.csv --val-annotations-path /data/datasets/wbc2_1024x1024/valid.csv
# JOB: python3 train.py --snapshot imagenet --phi 5 --gpu 1 --epochs 200 --snapshot-path /data/results/wbc2/det5 --random-transform --compute-val-loss --freeze-backbone --batch-size 4 --steps 1000 csv /data/datasets/wbc2_1024x1024/train.csv /data/datasets/wbc2_1024x1024/class.csv --val-annotations-path /data/datasets/wbc2_1024x1024/valid.csv
# JOB: python3 train.py --snapshot imagenet --phi 6 --gpu 2 --epochs 200 --snapshot-path /data/results/wbc2/det6 --random-transform --compute-val-loss --freeze-backbone --batch-size 2 --steps 1000 csv /data/datasets/wbc2_1024x1024/train.csv /data/datasets/wbc2_1024x1024/class.csv --val-annotations-path /data/datasets/wbc2_1024x1024/valid.csv
# JOB: python3 train.py --snapshot imagenet --phi 0 --gpu 3 --epochs 200 --snapshot-path /data/results/wbc2/det0 --random-transform --compute-val-loss --freeze-backbone --batch-size 64 --steps 1000 csv /data/datasets/wbc2_1024x1024/train.csv /data/datasets/wbc2_1024x1024/class.csv --val-annotations-path /data/datasets/wbc2_1024x1024/valid.csv

# Mar 2, 2021
# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 4 --gpu 0 --batch-size 2 --epoch 200 --steps 1000 --dataset wbc_1024x1024
# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 4 --gpu 1 --batch-size 2 --epoch 200 --steps 1000 --dataset wbc2_1024x1024
# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 4 --gpu 2 --batch-size 2 --epoch 200 --steps 1000 --dataset wbc3_1024x1024
# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 3 --gpu 3 --batch-size 4 --epoch 200 --steps 1000 --dataset wbc_1024x1024

# Mar 3, 2021
# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 0 --gpu 0 --batch-size 16 --epoch 400 --steps 100 --dataset wbc_1024x1024
# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 0 --gpu 1 --batch-size 16 --epoch 400 --steps 100 --dataset wbc2_1024x1024
# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 0 --gpu 2 --batch-size 16 --epoch 400 --steps 100 --dataset wbc3_1024x1024
# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 1 --gpu 3 --batch-size 10 --epoch 400 --steps 100 --dataset wbc_1024x1024

# JOB: python3 train_wbc.py --docker --snapshot /data/coco/efficientdet-d0.h5 --phi 0 --gpu 0 --batch-size 16 --epoch 400 --steps 100 --dataset wbc_1024x1024
# JOB: python3 train_wbc.py --docker --snapshot /data/coco/efficientdet-d0.h5 --phi 0 --gpu 1 --batch-size 16 --epoch 400 --steps 100 --dataset wbc2_1024x1024
# JOB: python3 train_wbc.py --docker --snapshot /data/coco/efficientdet-d0.h5 --phi 0 --gpu 2 --batch-size 16 --epoch 400 --steps 100 --dataset wbc3_1024x1024
# JOB: python3 train_wbc.py --docker --snapshot /data/coco/efficientdet-d1.h5 --phi 1 --gpu 3 --batch-size 10 --epoch 400 --steps 100 --dataset wbc_1024x1024

# Mar 5, 2021
# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 1 --gpu 0 --batch-size 10 --epoch 400 --steps 100 --dataset wbc_1024x1024 --cls 5
# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 1 --gpu 1 --batch-size 10 --epoch 400 --steps 100 --dataset wbc2_1024x1024 --cls 5
# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 1 --gpu 2 --batch-size 10 --epoch 400 --steps 100 --dataset wbc3_1024x1024 --cls 5
# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 2 --gpu 3 --batch-size 6 --epoch 400 --steps 100 --dataset wbc_1024x1024 --cls 5

# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 2 --gpu 0 --batch-size 6 --epoch 400 --steps 100 --dataset wbc_1024x1024 --cls 5
# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 2 --gpu 1 --batch-size 6 --epoch 400 --steps 100 --dataset wbc2_1024x1024 --cls 5
# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 2 --gpu 2 --batch-size 6 --epoch 400 --steps 100 --dataset wbc3_1024x1024 --cls 5
# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 2 --gpu 3 --batch-size 4 --epoch 400 --steps 100 --dataset wbc_1024x1024 --cls 5

# Mar 7, 2021
# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 2 --weighted-bifpn --gpu 0 --batch-size 6 --epoch 400 --steps 100 --dataset wbc_1024x1024 --cls 5
# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 2 --weighted-bifpn --gpu 1 --batch-size 6 --epoch 400 --steps 100 --dataset wbc2_1024x1024 --cls 5
# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 2 --weighted-bifpn --gpu 2 --batch-size 6 --epoch 400 --steps 100 --dataset wbc3_1024x1024 --cls 5
# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 3 --weighted-bifpn --gpu 3 --batch-size 4 --epoch 400 --steps 100 --dataset wbc_1024x1024 --cls 5

# Mar 9, 2021
# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 1 --weighted-bifpn --gpu 0 --batch-size 10 --epoch 400 --steps 100 --dataset wbc_1024x1024 --cls 4 --valid False
# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 1 --weighted-bifpn --gpu 1 --batch-size 10 --epoch 400 --steps 100 --dataset wbc2_1024x1024 --cls 4 --valid False
# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 1 --weighted-bifpn --gpu 2 --batch-size 10 --epoch 400 --steps 100 --dataset wbc_1024x1024 --cls 5 --valid False
# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 1 --weighted-bifpn --gpu 3 --batch-size 10 --epoch 400 --steps 100 --dataset wbc2_1024x1024 --cls 5 --valid False

# Mar 9, 2021
# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 1 --gpu 0 --batch-size 10 --epoch 400 --steps 100 --dataset wbc4_1024x1024 --cls 4 --valid False
# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 1 --gpu 1 --batch-size 10 --epoch 400 --steps 100 --dataset wbc4_1024x1024 --cls 5 --valid False
# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 1 --gpu 2 --batch-size 10 --epoch 400 --steps 100 --dataset wbc4_1024x1024 --cls 4 --valid True
# JOB: python3 train_wbc.py --docker --snapshot imagenet --phi 1 --gpu 3 --batch-size 10 --epoch 400 --steps 100 --dataset wbc4_1024x1024 --cls 5 --valid True

# Mar 17, 2021
# general
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 4 --phi 0 --gpu 0 --batch-size 12 --steps 100 --epoch 200 --valid False --cross 1
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 4 --phi 0 --gpu 1 --batch-size 12 --steps 100 --epoch 200 --valid False --cross 2
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 4 --phi 0 --gpu 2 --batch-size 12 --steps 100 --epoch 200 --valid False --cross 5
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 5 --phi 0 --gpu 3 --batch-size 12 --steps 100 --epoch 200 --valid False --cross 1
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 4 --phi 0 --gpu 2 --batch-size 16 --steps 100 --epoch 200 --valid False --cross 3
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 4 --phi 0 --gpu 3 --batch-size 16 --steps 100 --epoch 200 --valid False --cross 4

# general
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 5 --phi 0 --gpu 0 --batch-size 12 --steps 100 --epoch 200 --valid False --cross 2
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 5 --phi 0 --gpu 1 --batch-size 12 --steps 100 --epoch 200 --valid False --cross 3
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 5 --phi 0 --gpu 2 --batch-size 12 --steps 100 --epoch 200 --valid False --cross 4
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 5 --phi 0 --gpu 3 --batch-size 12 --steps 100 --epoch 200 --valid False --cross 5

# Mar 20, 2021
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc_1024x1024 --cls 4 --phi 0 --gpu 0 --batch-size 12 --steps 100 --epoch 200 --valid False --cross 1
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc_1024x1024 --cls 4 --phi 0 --gpu 1 --batch-size 12 --steps 100 --epoch 200 --valid False --cross 2
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc_1024x1024 --cls 4 --phi 0 --gpu 2 --batch-size 12 --steps 100 --epoch 200 --valid False --cross 5
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc_1024x1024 --cls 5 --phi 0 --gpu 3 --batch-size 12 --steps 100 --epoch 200 --valid False --cross 1

# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc_1024x1024 --cls 5 --phi 0 --gpu 0 --batch-size 12 --steps 100 --epoch 200 --valid False --cross 2
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc_1024x1024 --cls 5 --phi 0 --gpu 1 --batch-size 12 --steps 100 --epoch 200 --valid False --cross 3
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc_1024x1024 --cls 5 --phi 0 --gpu 2 --batch-size 12 --steps 100 --epoch 200 --valid False --cross 4
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc_1024x1024 --cls 5 --phi 0 --gpu 3 --batch-size 12 --steps 100 --epoch 200 --valid False --cross 5

# Mar 21, 2021
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc4_1024x1024 --cls 4 --phi 0 --gpu 0 --batch-size 12 --steps 100 --epoch 200 --valid False --cross 1
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc4_1024x1024 --cls 4 --phi 0 --gpu 1 --batch-size 12 --steps 100 --epoch 200 --valid False --cross 2
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc4_1024x1024 --cls 4 --phi 0 --gpu 2 --batch-size 12 --steps 100 --epoch 200 --valid False --cross 3
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc4_1024x1024 --cls 4 --phi 0 --gpu 3 --batch-size 12 --steps 100 --epoch 200 --valid False --cross 4

# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc4_1024x1024 --cls 5 --phi 0 --gpu 0 --batch-size 12 --steps 100 --epoch 200 --valid False --cross 1
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc4_1024x1024 --cls 5 --phi 0 --gpu 1 --batch-size 12 --steps 100 --epoch 200 --valid False --cross 2
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc4_1024x1024 --cls 5 --phi 0 --gpu 2 --batch-size 12 --steps 100 --epoch 200 --valid False --cross 3
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc4_1024x1024 --cls 5 --phi 0 --gpu 3 --batch-size 12 --steps 100 --epoch 200 --valid False --cross 4

# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc4_1024x1024 --cls 4 --phi 0 --gpu 0 --batch-size 12 --steps 100 --epoch 200 --valid False --cross 5
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc4_1024x1024 --cls 5 --phi 0 --gpu 1 --batch-size 12 --steps 100 --epoch 200 --valid False --cross 5
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc4_1024x1024 --cls 4 --phi 0 --gpu 2 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 1
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc4_1024x1024 --cls 4 --phi 0 --gpu 3 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 2

# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 4 --phi 1 --gpu 0 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 1
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 5 --phi 1 --gpu 1 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 2
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 4 --phi 1 --gpu 2 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 3
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 4 --phi 1 --gpu 3 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 4

# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 4 --phi 1 --gpu 0 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 1
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 4 --phi 1 --gpu 1 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 2
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 4 --phi 1 --gpu 2 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 3
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 4 --phi 1 --gpu 3 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 4

# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 5 --phi 1 --gpu 0 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 1
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 5 --phi 1 --gpu 1 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 2
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 5 --phi 1 --gpu 2 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 3
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 5 --phi 1 --gpu 3 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 4

# Mar 22, 2021
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 4 --phi 1 --gpu 0 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 5
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 5 --phi 1 --gpu 1 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 5
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc_1024x1024 --cls 4 --phi 1 --gpu 2 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 1
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc_1024x1024 --cls 4 --phi 1 --gpu 3 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 2

# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc_1024x1024 --cls 4 --phi 1 --gpu 0 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 3
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc_1024x1024 --cls 4 --phi 1 --gpu 1 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 4
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc_1024x1024 --cls 4 --phi 1 --gpu 2 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 5
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc_1024x1024 --cls 5 --phi 1 --gpu 3 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 1

# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc_1024x1024 --cls 5 --phi 1 --gpu 0 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 2
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc_1024x1024 --cls 5 --phi 1 --gpu 1 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 3
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc_1024x1024 --cls 5 --phi 1 --gpu 2 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 4
# JOB: python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc_1024x1024 --cls 5 --phi 1 --gpu 3 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 5

# July 6, 2021
# JOB: python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc_1024x1024 --valid False --phi 0 --gpu 0 --batch-size 8 --steps 100 --epoch 200 --aug False --bstrp 0
# JOB: python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc_1024x1024 --valid False --phi 0 --gpu 1 --batch-size 8 --steps 100 --epoch 200 --aug False --bstrp 1
# JOB: python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc_1024x1024 --valid False --phi 0 --gpu 2 --batch-size 8 --steps 100 --epoch 200 --aug False --bstrp 2
# JOB: python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc_1024x1024 --valid False --phi 0 --gpu 3 --batch-size 8 --steps 100 --epoch 200 --aug False --bstrp 3

# JOB: python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc_1024x1024 --valid False --phi 0 --gpu 0 --batch-size 8 --steps 100 --epoch 200 --aug False --bstrp 4
# JOB: python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc_1024x1024 --valid False --phi 1 --gpu 1 --batch-size 8 --steps 100 --epoch 200 --aug False --bstrp 0
# JOB: python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc_1024x1024 --valid False --phi 1 --gpu 2 --batch-size 8 --steps 100 --epoch 200 --aug False --bstrp 1
# JOB: python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc_1024x1024 --valid False --phi 1 --gpu 3 --batch-size 8 --steps 100 --epoch 200 --aug False --bstrp 2

# JOB: python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc_1024x1024 --valid False --phi 0 --gpu 0 --batch-size 8 --steps 50 --epoch 400 --aug False --bstrp 0 --lr 1e-4 --lw 0.5
# JOB: python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc_1024x1024 --valid False --phi 1 --gpu 1 --batch-size 8 --steps 50 --epoch 400 --aug False --bstrp 0 --lr 1e-4 --lw 0.1
# JOB: python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc_1024x1024 --valid False --phi 1 --gpu 2 --batch-size 8 --steps 50 --epoch 400 --aug False --bstrp 0 --lr 5e-5 --lw 0.5
# JOB: python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc_1024x1024 --valid False --phi 1 --gpu 3 --batch-size 8 --steps 50 --epoch 400 --aug False --bstrp 0 --lr 5e-5 --lw 0.1

JOB: python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc_1024x1024 --valid False --phi 0 --gpu 0 --batch-size 16 --steps 50 --epoch 400 --aug False --bstrp 1 --lr 1e-4 --lw 0.5
JOB: python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc_1024x1024 --valid False --phi 0 --gpu 1 --batch-size 16 --steps 50 --epoch 400 --aug False --bstrp 2 --lr 1e-4 --lw 0.5
JOB: python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc_1024x1024 --valid False --phi 0 --gpu 2 --batch-size 16 --steps 50 --epoch 400 --aug False --bstrp 3 --lr 1e-4 --lw 0.5
JOB: python3 train_wbc_bootstrap.py --docker --snapshot imagenet --dataset wbc_1024x1024 --valid False --phi 0 --gpu 3 --batch-size 16 --steps 50 --epoch 400 --aug False --bstrp 4 --lr 1e-4 --lw 0.5