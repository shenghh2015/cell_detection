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

JOB: python3 train_wbc.py --docker --snapshot /data/coco/efficientdet-d0.h5 --phi 0 --gpu 0 --batch-size 16 --epoch 400 --steps 100 --dataset wbc_1024x1024
JOB: python3 train_wbc.py --docker --snapshot /data/coco/efficientdet-d0.h5 --phi 0 --gpu 1 --batch-size 16 --epoch 400 --steps 100 --dataset wbc2_1024x1024
JOB: python3 train_wbc.py --docker --snapshot /data/coco/efficientdet-d0.h5 --phi 0 --gpu 2 --batch-size 16 --epoch 400 --steps 100 --dataset wbc3_1024x1024
JOB: python3 train_wbc.py --docker --snapshot /data/coco/efficientdet-d1.h5 --phi 1 --gpu 3 --batch-size 10 --epoch 400 --steps 100 --dataset wbc_1024x1024