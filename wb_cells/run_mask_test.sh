# python3 test_mask.py --gpu 2 --index 0 --save True
# python3 test_mask.py --gpu 2 --index 1 --save True
# python3 test_mask.py --gpu 2 --index 2 --save True

# python3 test_crop.py --gpu 2 --index 0 --save True
# python3 test_crop.py --gpu 2 --index 1 --save True
# python3 test_crop.py --gpu 2 --index 2 --save True

# python3 test_crop.py --gpu 2 --index 3 --save True
# python3 test_crop.py --gpu 2 --index 4 --save True
# python3 test_crop.py --gpu 2 --index 5 --save True
# python3 test_crop.py --gpu 2 --index 6 --save True

# python3 test_crop.py --gpu 2 --index 9 --save False
# python3 test_crop.py --gpu 2 --index 10 --save False
# python3 test_crop.py --gpu 2 --index 11 --save False
# python3 test_crop.py --gpu 2 --index 12 --save False

python3 test_crop.py --gpu 2 --index 0 --save False --model_file './test_mask/models2.txt'
python3 test_crop.py --gpu 2 --index 1 --save False --model_file './test_mask/models2.txt'
python3 test_crop.py --gpu 2 --index 2 --save False --model_file './test_mask/models2.txt'
python3 test_crop.py --gpu 2 --index 3 --save False --model_file './test_mask/models2.txt'
python3 test_crop.py --gpu 2 --index 4 --save False --model_file './test_mask/models2.txt'
python3 test_crop.py --gpu 2 --index 5 --save False --model_file './test_mask/models2.txt'
python3 test_crop.py --gpu 2 --index 6 --save False --model_file './test_mask/models2.txt'