# parser = argparse.ArgumentParser()
# parser.add_argument("--docker", type=str2bool, default = True)
# parser.add_argument("--gpu", type=str, default = '0')
# parser.add_argument("--net_type", type=str, default = 'AtUnet')  #Unet, Linknet, PSPNet, FPN
# parser.add_argument("--backbone", type=str, default = 'efficientnetb0')
# parser.add_argument("--feat_version", type=int, default = None)
# parser.add_argument("--epoch", type=int, default = 2)
# parser.add_argument("--dim", type=int, default = 512)
# parser.add_argument("--batch_size", type=int, default = 2)
# parser.add_argument("--dataset", type=str, default = 'wbc_1024x1024')
# parser.add_argument("--data_version", type=int, default = 0)
# parser.add_argument("--upsample", type=str, default = 'upsampling')
# parser.add_argument("--filters", type=int, default = 256)
# parser.add_argument("--rot", type=float, default = 0)
# parser.add_argument("--lr", type=float, default = 1e-3)
# parser.add_argument("--bk", type=float, default = 0.5)
# parser.add_argument("--focal_weight", type=float, default = 1)
# parser.add_argument("--pre_train", type=str2bool, default = True)
# parser.add_argument("--newest", type=str2bool, default = False)
# parser.add_argument("--train", type=int, default = None)
# parser.add_argument("--loss", type=str, default = 'focal+dice')
# parser.add_argument("--reduce_factor", type=float, default = 1.0)
# args = parser.parse_args()
# print(args)

python train_seg.py --docker True --gpu 0 --net_type Unet --backbone efficientnetb0 --epoch 2400 --dim 1024 --batch 2 --lr 1e-4 --loss focal --focal_weight 4
python train_seg.py --docker True --gpu 1 --net_type Unet --backbone efficientnetb1 --epoch 2400 --dim 928 --batch 2 --lr 1e-4 --loss focal --focal_weight 4