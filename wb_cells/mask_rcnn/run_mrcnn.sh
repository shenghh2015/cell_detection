# Feb 24, 2021
python3 wbc.py train --dataset=./dataset/data --subset=train --weights=coco --gpu 0,2,3 --backbone 'resnet50' --imgs 4
python3 wbc.py train --dataset=./dataset/data --subset=train --weights=coco --gpu 0,3 --backbone 'resnet101' --imgs 2
python3 wbc.py train --dataset=./dataset/data --subset=train --weights=imagenet --gpu 3,5 --backbone 'resnet50' --imgs 6
python3 wbc.py train --dataset=./dataset/data --subset=train --weights=imagenet --gpu 0,1 --backbone 'resnet101' --imgs 2

# Feb 25, 2021
python3 wbc.py train --dataset=./dataset/data --subset=train --weights=coco --gpu 2,3 --backbone 'resnet50'
python3 wbc.py train --dataset=./dataset/data --subset=train --weights=coco --gpu 3,7 --backbone 'resnet101'

# Feb 28, 2021
python3 wbc2.py train --dataset=./dataset2/data --subset=train --weights=coco --gpu 0,2 --backbone 'resnet50'
python3 wbc2.py train --dataset=./dataset2/data --subset=train --weights=coco --gpu 0,1 --backbone 'resnet101'