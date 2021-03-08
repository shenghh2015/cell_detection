import os
import cv2
from skimage import io
from tifffile import imread, imsave
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
import sys
sys.path.append('../')
import segmentation_models as sm
from segmentation_models_v1 import Unet
from keras_applications.imagenet_utils import preprocess_input

from helper_function import precision, recall, f1_score, iou_calculate, generate_folder
from sklearn.metrics import confusion_matrix

sm.set_framework('tf.keras')
import glob
from natsort import natsorted
import pickle

def get_dataset(model_name):
		dataset = ''
		splits = model_name.split('-')
		for v, sp in enumerate(splits):
				if sp == 'set':
						dataset = splits[v + 1]
		return dataset
		

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# bounding box model dir
model_name = 'phi-1-set-wbc_1024x1024-wfpn-False-ep-400-stp-100-bz-10-snap-imagenet-cls-5'
# model_name = 'phi-1-set-wbc2_1024x1024-wfpn-False-ep-400-stp-100-bz-10-snap-imagenet-cls-5'
# model_name = 'phi-0-set-wbc2_1024x1024-wfpn-False-ep-400-stp-100-bz-8-snap-imagenet-cls-5'
# model_name = 'phi-1-set-wbc3_1024x1024-wfpn-False-ep-400-stp-100-bz-10-snap-imagenet-cls-5'
dataset = get_dataset(model_name)
bbox_root_dir = '/data/models/{}'.format(dataset)
bbox_dir = bbox_root_dir + '/' + model_name + '/' + 'pred_boxes'

# segmentation model dir
seg_model_name = 'mask-net-Unet-bone-efficientnetb7-pre-True-epoch-4800-batch-8-lr-0.0001-dim-384-train-None-rot-25.0-set-wbc_1024x1024-dv-0-loss-focal+dice-up-upsampling-filters-256-rf-1.0-bk-1.0-flw-4.0-fv-None-new-False-crop-True-cls-2'
# seg_model_name = 'mask-net-Unet-bone-efficientnetb7-pre-True-epoch-4800-batch-8-lr-0.0001-dim-384-train-None-rot-20.0-set-wbc_1024x1024-dv-0-loss-focal+dice-up-upsampling-filters-256-rf-1.0-bk-1.0-flw-4.0-fv-None-new-False-crop-True-cls-2'
#seg_model_name = 'mask-net-Unet-bone-efficientnetb7-pre-True-epoch-4800-batch-8-lr-0.0001-dim-384-train-None-rot-20.0-set-wbc2_1024x1024-dv-0-loss-focal+dice-up-upsampling-filters-256-rf-1.0-bk-1.0-flw-4.0-fv-None-new-False-crop-True-cls-2'
# seg_model_name = 'mask-net-Unet-bone-efficientnetb7-pre-True-epoch-4800-batch-8-lr-0.0001-dim-384-train-None-rot-20.0-set-wbc3_1024x1024-dv-0-loss-focal+dice-up-upsampling-filters-256-rf-1.0-bk-1.0-flw-4.0-fv-None-new-False-crop-True-cls-2'
seg_model_dir = '/data/wbc_models/{}'.format(seg_model_name)
model_file_path = seg_model_dir + '/ready_model.h5'

# load segmentation model
model = tf.keras.models.load_model(model_file_path)

## parse model name
splits = seg_model_name.split('-')

nb_filters = 256
upsample = 'upsampling'
feature_version = None
data_version = 0
crop = False
for v in range(len(splits)):
	if splits[v]=='set':
		if '1024x1024' in splits[v+1]:
			dataset = splits[v+1]
			val_dim1, val_dim2 = 1024, 1024
			test_dim1, test_dim2 = 1024, 1024
			dim1, dim2 = 1024, 1024
	elif splits[v] == 'crop':
			crop = True
			val_dim1, val_dim2 = 384, 384
			test_dim1, test_dim2 = 384, 384
			dim1, dim2 = 384, 384	
	elif splits[v] == 'net':
		net_arch = splits[v+1]
	elif splits[v] == 'bone':
		backbone = splits[v+1]
	elif splits[v] == 'dv':
		data_version = int(splits[v+1])
	elif splits[v] == 'up':
		upsample = splits[v+1]
	elif splits[v] == 'filters':
		nb_filters = int(splits[v+1])
	elif splits[v] == 'fv':
		if not splits[v+1] == 'None':
			feature_version = int(splits[v+1]); print('feature version:{}'.format(feature_version))	

def read_txt(txt_dir):
    lines = []
    with open(txt_dir, 'r+') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines
			
DATA_DIR = '/data/datasets/{}'.format(dataset)
images_dir = DATA_DIR+'/{}'.format('images') 
masks_dir = DATA_DIR+'/{}'.format('seg_maps')
preprocess_input = sm.get_preprocessing(backbone)

# crop_images_dir = DATA_DIR+'/crop/{}'.format('images')
# crop_fns = os.listdir(crop_images_dir)

# for cfn in crop_fns[:20]:
# 		crop_img = io.imread(crop_images_dir + '/{}'.format(cfn))
# 		crop_img = np.uint8((crop_img-crop_img.min())*255/(crop_img.max()-crop_img.min()))
# 		patch_input = preprocess_input(crop_img[np.newaxis,:])
# 		pr_mask = np.squeeze(model.predict(patch_input))
# 		pr_mask = (pr_mask > 0.5) * 1.0
# 		io.imsave('crop_img_{}'.format(cfn), patch_input.squeeze())
# 		io.imsave('crop_pr_{}'.format(cfn), pr_mask.squeeze())
# train_fns = read_txt(DATA_DIR+'/train_list.txt')
# valid_fns = read_txt(DATA_DIR+'/valid_list.txt')
# test_fns = read_txt(DATA_DIR+'/test_list.txt')

test_fns = os.listdir(bbox_dir)
# label_cache = {0:1, 1:3, 2:4, 3:5}
label_cache = {0:1, 1:2, 2:3, 3:4, 4:5}
pr_maps, gt_maps = [], []
save_dir = bbox_root_dir + '/' + model_name + '/post_process'
generate_folder(save_dir)
for fi, fn in enumerate(test_fns):
		#img_id = test_fns[1]
		img_id = test_fns[fi].replace('.pkl', '')
		image = io.imread(images_dir + '/{}.png'.format(img_id))
		gt_map = io.imread(masks_dir + '/{}.tif'.format(img_id))
		bboxes = pickle.load(open(bbox_dir + '/{}.pkl'.format(img_id),'rb'))
		gt_maps.append(gt_map)
		pr_map = np.zeros(image.shape[:2])

		boxes = bboxes['boxes']
		labels = bboxes['labels']
		scores = bboxes['scores']
		
		if not dataset == 'wbc_1024x1024': wx, hy, _ = image.shape 
		else: wx, hy = image.shape
		
		dim = 384//2

		# box_id = 0
		box_mask = np.zeros(image.shape[:2])
		for box_id in range(len(boxes)):
				x1, y1, x2, y2 = boxes[box_id]
				box_mask[int(y1): int(y2), int(x1): int(x2)] = 1
				cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
				x1 = cx - dim
				x2 = cx + dim
				y1 = cy - dim
				y2 = cy + dim
				if x1 < 0: 
						x2 = x2 - x1
						x1 = 0
				if x2 > wx: 
						x1 = x1 - (x2 - (wx - 1))
						x2 = wx - 1
				if y1 < 0: 
						y2 = y2 - y1
						y1 = 0
				if y2 > hy: 
						y1 = y1 - (y2 - (hy - 1))
						y2 = hy - 1
				print(x1, y1, x2, y2)
				if not dataset == 'wbc_1024x1024':
						patch_box = image[np.newaxis, y1:y2, x1:x2,:]
				else:
						patch_box = image[np.newaxis, y1:y2, x1:x2]
				patch_box = np.uint8((patch_box-patch_box.min())*255/(patch_box.max()-patch_box.min()))
				if dataset == 'wbc_1024x1024':
						patch_box = np.stack([patch_box, patch_box, patch_box], axis = -1)
				patch_input = preprocess_input(patch_box)
				pr_mask = np.squeeze(model.predict(patch_input))
				pr_mask = (pr_mask > 0.5) * 1.0
				pr_map[y1:y2, x1:x2] = pr_mask * (label_cache[labels[box_id]])
		box_mask = 1 - box_mask
		pr_map[np.where(box_mask > 0)] = 0
		io.imsave(save_dir + '/{}.png'.format(img_id), pr_map * 40)
		pr_maps.append(pr_map)
gt_maps = np.stack(gt_maps)
pr_maps = np.stack(pr_maps)
# gt_maps[np.where(gt_maps == 2)] = 0
y_true=gt_maps.flatten(); y_pred = pr_maps.flatten()
cf_mat = confusion_matrix(y_true, y_pred)
print(np.round(cf_mat, 3))
prec_scores = []; recall_scores = []; f1_scores = []; iou_scores=[]
for i in range(cf_mat.shape[0]):
	prec_scores.append(precision(i,cf_mat))
	recall_scores.append(recall(i,cf_mat))
	f1_scores.append(f1_score(i,cf_mat))
print('precision:{}'.format(np.round(prec_scores, 3)))
print('mean precision: {:.4f}\n'.format(np.mean(prec_scores)))
print('recall: {}'.format(np.round(recall_scores, 3)))
print('mean recall:{:.4f}\n'.format(np.mean(recall_scores)))
print('f1 score: {}'.format(np.round(f1_scores, 3)))
print('mean f1-score (pixel):{:.4f}\n'.format(np.mean(f1_scores)))

save_file_path = bbox_root_dir + '/' + model_name + '/post_seg.txt'
with open(save_file_path, 'w+') as f:
		f.write('detection model: {}\n'.format(model_name))
		f.write('segmentation model: {}\n'.format(seg_model_name))
		f.write('confusion matrix:\n')
		np.savetxt(f, cf_mat, fmt='%-7d')
		f.write('precision:\n')
		np.savetxt(f, np.array(prec_scores).reshape((1,-1)), fmt='%-.3f')
		f.write('mean precision: {:.3f}\n'.format(np.mean(prec_scores)))
		f.write('recall:\n')
		np.savetxt(f, np.array(recall_scores).reshape((1,-1)), fmt='%.3f')
		f.write('mean recall: {:.3f}\n'.format(np.mean(recall_scores)))
		f.write('f1_score:\n')
		np.savetxt(f, np.array(f1_scores).reshape((1,-1)), fmt='%.3f')
		f.write('mean f1_score: {:.3f}\n'.format(np.mean(f1_scores)))