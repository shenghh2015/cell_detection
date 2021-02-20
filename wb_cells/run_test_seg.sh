# parser.add_argument("--gpu", type=str, default = '0')
# parser.add_argument("--model_file", type=str, default = 'models.txt')
# parser.add_argument("--model_index", type=int, default = 0)
# parser.add_argument("--save", type=str2bool, default = False)

python test_seg.py --gpu 2 --model_file './test_seg/models.txt' --index 0
python test_seg.py --gpu 2 --model_file './test_seg/models.txt' --index 1