# Mar 18, 2021
# python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 4 --phi 0 --gpu 0 --batch-size 8 --steps 100 --epoch 200 --valid False --cross 5
# python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 4 --phi 1 --gpu 2 --batch-size 4 --steps 100 --epoch 200 --valid False --cross 1
# python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 4 --phi 1 --gpu 2 --batch-size 4 --steps 100 --epoch 200 --valid False --cross 2
# python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc2_1024x1024 --cls 4 --phi 1 --gpu 2 --batch-size 4 --steps 100 --epoch 200 --valid False --cross 3

# Mar 21, 2021
python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc4_1024x1024 --cls 5 --phi 1 --gpu 2 --batch-size 4 --steps 100 --epoch 200 --valid False --cross 2
python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc4_1024x1024 --cls 5 --phi 1 --gpu 2 --batch-size 4 --steps 100 --epoch 200 --valid False --cross 4
python3 train_wbc_cv.py --docker --snapshot imagenet --dataset wbc4_1024x1024 --cls 5 --phi 1 --gpu 2 --batch-size 4 --steps 100 --epoch 200 --valid False --cross 5