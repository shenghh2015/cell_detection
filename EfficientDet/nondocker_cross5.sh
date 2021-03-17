# Mar 17, 2021
python3 train_wbc_cv.py --snapshot imagenet --dataset wbc4_1024x1024 --cls 4 --phi 0 --gpu 2 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 1
python3 train_wbc_cv.py --snapshot imagenet --dataset wbc4_1024x1024 --cls 4 --phi 0 --gpu 2 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 2
python3 train_wbc_cv.py --snapshot imagenet --dataset wbc4_1024x1024 --cls 4 --phi 0 --gpu 2 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 3