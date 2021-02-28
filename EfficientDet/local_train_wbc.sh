# python3 train.py --snapshot imagenet --phi 0 --gpu 0 --random-transform --compute-val-loss --freeze-backbone --batch-size 32 --steps 1000 csv ../../datasets/wbc_1024x1024/annotation.csv ../../datasets/wbc_1024x1024/class.csv
# Feb 27, 2021
python3 train.py --snapshot imagenet --phi 0 --gpu 0 --snapshot-path ../../results/det0 --random-transform --compute-val-loss --freeze-backbone --batch-size 32 --steps 1000 csv ../../datasets/wbc_1024x1024/train.csv ../../datasets/wbc_1024x1024/class.csv --val-annotations-path ../../datasets/wbc_1024x1024/valid.csv

python3 train.py --snapshot imagenet --phi 1 --gpu 2 --snapshot-path ../../results/det1 --random-transform --compute-val-loss --freeze-backbone --batch-size 32 --steps 1000 csv ../../datasets/wbc_1024x1024/train.csv ../../datasets/wbc_1024x1024/class.csv --val-annotations-path ../../datasets/wbc_1024x1024/valid.csv

python3 train.py --snapshot imagenet --phi 2 --gpu 2 --snapshot-path ../../results/det2 --random-transform --compute-val-loss --freeze-backbone --batch-size 20 --steps 1000 csv ../../datasets/wbc_1024x1024/train.csv ../../datasets/wbc_1024x1024/class.csv --val-annotations-path ../../datasets/wbc_1024x1024/valid.csv

python3 train.py --snapshot imagenet --phi 3 --gpu 1 --snapshot-path ../../results/det3 --random-transform --compute-val-loss --freeze-backbone --batch-size 8 --steps 1000 csv ../../datasets/wbc_1024x1024/train.csv ../../datasets/wbc_1024x1024/class.csv --val-annotations-path ../../datasets/wbc_1024x1024/valid.csv

python3 train.py --snapshot imagenet --phi 4 --gpu 3 --snapshot-path ../../results/det4 --random-transform --compute-val-loss --freeze-backbone --batch-size 4 --steps 1000 csv ../../datasets/wbc_1024x1024/train.csv ../../datasets/wbc_1024x1024/class.csv --val-annotations-path ../../datasets/wbc_1024x1024/valid.csv

python3 train.py --snapshot imagenet --phi 5 --gpu 4 --snapshot-path ../../results/det5 --random-transform --compute-val-loss --freeze-backbone --batch-size 2 --steps 1000 csv ../../datasets/wbc_1024x1024/train.csv ../../datasets/wbc_1024x1024/class.csv --val-annotations-path ../../datasets/wbc_1024x1024/valid.csv

python3 train.py --snapshot imagenet --phi 6 --gpu 1 --snapshot-path ../../results/det6 --random-transform --compute-val-loss --freeze-backbone --batch-size 2 --steps 1000 csv ../../datasets/wbc_1024x1024/train.csv ../../datasets/wbc_1024x1024/class.csv --val-annotations-path ../../datasets/wbc_1024x1024/valid.csv

# Feb 27, 2021
python3 train.py --snapshot imagenet --phi 1 --gpu 0 --snapshot-path ../../results/det1 --random-transform --compute-val-loss --freeze-backbone --batch-size 32 --steps 1000 csv ../../datasets/wbc2_1024x1024/train.csv ../../datasets/wbc2_1024x1024/class.csv --val-annotations-path ../../datasets/wbc2_1024x1024/valid.csv

python3 train.py --snapshot imagenet --phi 4 --gpu 2 --snapshot-path ../../results/wbc2/det4 --random-transform --compute-val-loss --freeze-backbone --batch-size 4 --steps 1000 csv ../../datasets/wbc2_1024x1024/train.csv ../../datasets/wbc2_1024x1024/class.csv --val-annotations-path ../../datasets/wbc2_1024x1024/valid.csv

# Feb 28, 2021
python3 train.py --snapshot imagenet --phi 1 --gpu 0 --snapshot-path ../../results/det1 --random-transform --compute-val-loss --freeze-backbone --batch-size 32 --steps 1000 csv ../../datasets/wbc2_1024x1024/train.csv ../../datasets/wbc2_1024x1024/class.csv --val-annotations-path ../../datasets/wbc2_1024x1024/valid.csv