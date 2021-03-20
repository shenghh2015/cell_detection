# Mar 17, 2021
# python3 train_wbc_cv.py --snapshot imagenet --dataset wbc4_1024x1024 --cls 4 --phi 0 --gpu 5 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 4
# python3 train_wbc_cv.py --snapshot imagenet --dataset wbc4_1024x1024 --cls 4 --phi 0 --gpu 5 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 5
# python3 train_wbc_cv.py --snapshot imagenet --dataset wbc4_1024x1024 --cls 5 --phi 0 --gpu 5 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 1

# Mar 20, 2021
python3 train_wbc_cv.py --snapshot imagenet --dataset wbc2_1024x1024 --cls 4 --phi 1 --gpu 4 --batch-size 4 --steps 100 --epoch 200 --valid False --cross 5
python3 train_wbc_cv.py --snapshot imagenet --dataset wbc2_1024x1024 --cls 5 --phi 1 --gpu 4 --batch-size 4 --steps 100 --epoch 200 --valid False --cross 1