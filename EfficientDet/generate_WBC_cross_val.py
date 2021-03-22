import cv2
import json
import numpy as np
import os
import time
import glob

from natsort import natsorted
import pickle
from cv_models import cv_models

from generators.csv_ import CSVGenerator
from eval.common import evaluate, evaluate_wbc

from model import efficientdet
from utils import preprocess_image, postprocess_boxes
# from utils.draw_boxes import draw_boxes

from helper_function import draw_pr_boxes, draw_gt_boxes

docker = True
dataset_dir = '/data/datasets' if docker else './datasets'

cls_label_map = {'neutrophils': 0, 'eosinophils': 1, 'lymphocytes': 2, 'monocytes': 3}
cls_label_map2 = {'neutrophils': 0, 'bands': 1, 'eosinophils': 2, 'lymphocytes': 3, 'monocytes': 4}

## cross validation 
def load_test_boxes(dataset, cross, cls):
		#     data_csv_file = dataset_dir + '/' + dataset + '/docker/test_{}c.csv'.format(cls)
    data_csv_file = dataset_dir + '/' + dataset + '/cv_sets/cv{}_test_{}c.csv'.format(cross, cls)
    gt_boxes = {}
    with open(data_csv_file, 'r+') as f:
        lines = f.readlines()
        for line in lines:
            splits = line.strip().split(',')
            img_path = splits[0]
            box = [int(splits[1]), int(splits[2]), int(splits[3]), int(splits[4])]
            label_map = cls_label_map if cls == 4 else cls_label_map2
            label = label_map[splits[5]]
            if not img_path in gt_boxes:
                gt_boxes[img_path] = ([box], [label])
            else:
                gt_boxes[img_path][0].append(box)
                gt_boxes[img_path][1].append(label)
    return gt_boxes

def evaluate_ap(model, phi, dataset, model_path = './', cross = 1, cls = 4, save = False):
		# dataset = 'wbc_1024x1024'
		val_annotations_path = dataset_dir + '/' + dataset + '/cv_sets/cv{}_test_{}c.csv'.format(cross, cls)
		classes_path = dataset_dir + '/' + dataset + '/docker/class_{}c.csv'.format(cls)
		validation_generator = CSVGenerator(
		val_annotations_path,
		classes_path,
		shuffle_groups=False,
		phi= phi)
		score_threshold = 0.3
		model_dir = os.path.dirname(model_path) if save else None
		average_precisions = evaluate_wbc(
				validation_generator,
				model,
				iou_threshold= 0.5,
				score_threshold= score_threshold,
				max_detections=10,
				model_dir = model_dir,
				visualize = False
		)
		total_instances = []
		precisions = []
		write_lines = []
		generator = validation_generator
		for label, (average_precision, num_annotations) in average_precisions.items():
				print('{:.0f} instances of class'.format(num_annotations),
							generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
				write_lines.append('{:.0f} instances of class '.format(num_annotations) + 
							generator.label_to_name(label) + ' with average precision: {:.4f}\n'.format(average_precision))
				total_instances.append(num_annotations)
				precisions.append(average_precision)
		print(np.mean(precisions))
		# save the summary if save is True
		if save:
			model_folder = os.path.join(model_root_dir, dataset, os.path.basename(os.path.dirname(model_path)))
			with open(model_folder + '/best_test_summary', 'w+') as f:
					f.write('Model path: {}\n'.format(model_path))
					for line in write_lines:
					    f.write(line)
					f.write('mAP: {:.4f}\n'.format(np.mean(precisions))) 
		return np.mean(precisions)

# dataset = 'wbc_1024x1024'
# gt_boxes = load_test_boxes(dataset)

model_root_dir = '/data/cv_models/'

def fetch_top_weights(model_name, top = 10):
    #model_name = 'phi-0-set-wbc_1024x1024-wfpn-False-ep-200-stp-100-bz-8'
    # fetch top epochs
    splits = model_name.split('-')
    dataset = 'wbc_1024x1024'
    for v, sp in enumerate(splits):
        if sp == 'set':
            dataset = splits[v + 1]
    model_folder = os.path.join(model_root_dir, dataset, model_name)
    log_file = model_folder + '/train_log.txt'
    mAPs = []
    with open(log_file, 'r+') as f:
        lines = f.readlines()
    for line in lines:
        if 'mAP: ' in line:
            mAPs.append(float(line.strip().split(': ')[-1]))
    mAP_index_map = {}
    for ind, mAP in enumerate(mAPs):
        mAP_index_map[mAP] = ind
    mAPs.sort()
    mAPs.reverse()
    print(mAPs[:top])
    top_epochs = list(set([mAP_index_map[mAP] + 1 for mAP in mAPs[:top]]))
    # fetch top weights
    weight_files = []
    for epoch in top_epochs:
        weight_files += glob.glob(model_folder + '/csv_{}_*'.format(epoch))
    print(weight_files)
    return weight_files



def main():
		os.environ['CUDA_VISIBLE_DEVICES'] = '0'
		dataset = 'wbc4_1024x1024'
		model_names = cv_models[dataset]; print(len(model_names))
		# model_names = [model_name for model_name in cv_models[dataset] if 'cls-5' in model_name]
		for model_name in model_names:
				# parse the model name to get parameters
				print(model_name)
				splits = model_name.split('-')
				dataset = 'wbc_1024x1024'
				phi = 0
				wfpn = False
				cls = 4
				cross = 1
				for v, sp in enumerate(splits):
						if sp == 'set':
								dataset = splits[v + 1]
						elif sp == 'phi':
								phi = int(splits[v+1])
						elif sp == 'wfpn':
								wfpn = splits[v+1].lower() == 'true'
						elif sp == 'cls':
								cls = int(splits[v+1])
						elif sp == 'cross':
								cross = int(splits[v+1])
				mAP_list = []
				top = 5
				weight_files = fetch_top_weights(model_name, top = top)
				image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
				image_size = image_sizes[phi]
				classes1 = {0:'neutrophils', 1:'eosinophils', 2:'lymphocytes', 3:'monocytes'}
				classes2 = {0:'neutrophils', 1:'bands', 2:'eosinophils', 3:'lymphocytes', 4:'monocytes'}
				classes = classes1 if cls == 4 else classes2
				num_classes = len(classes)
				score_threshold = 0.3
				colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]
				_, model = efficientdet(phi=phi,
																weighted_bifpn=wfpn,
																num_classes=num_classes,
																score_threshold=score_threshold)
				for model_path in weight_files:
						model.load_weights(model_path, by_name=True)
						mAP = evaluate_ap(model, phi, dataset, cross = cross, cls = cls)
						mAP_list.append(mAP)
				best_index = np.argmax(mAP_list); print('Best mAP: {:.3f}'.format(mAP_list[best_index]))
				model_path = weight_files[best_index]
				model_dir = os.path.dirname(model_path)
				model.load_weights(model_path, by_name=True)
				evaluate_ap(model, phi, dataset, model_path, cross = cross, cls = cls, save = True)
				gt_boxes = load_test_boxes(dataset, cross = cross, cls = cls)
				#for image_path in glob.glob('datasets/VOC2007/JPEGImages/*.png'):
				for image_path in gt_boxes:
						#print(image_path)
						image = cv2.imread(image_path)
						src_image = image.copy()
						gt_image = image.copy()
						# BGR -> RGB
						image = image[:, :, ::-1]
						h, w = image.shape[:2]
						image, scale = preprocess_image(image, image_size=image_size)
						# run network
						start = time.time()
						boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
						boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
						#print(time.time() - start)
						boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)

						# select indices which have a score above the threshold
						indices = np.where(scores[:] > score_threshold)[0]

						# select those detections
						boxes = boxes[indices]
						labels = labels[indices]
						scores = scores[indices]
						pkl_dic = {'boxes': boxes, 'labels': labels, 'scores': scores}
						pr_dir = model_dir + '/pred_boxes'
						if not os.path.exists(pr_dir):
								os.system('mkdir -p {}'.format(pr_dir))
						pickle.dump(pkl_dic, open(pr_dir + '/{}.pkl'.format(os.path.basename(image_path).replace('.png', '')), "wb"))
						# draw predicted bounding boxes
						draw_pr_boxes(src_image, boxes, scores, labels, colors, classes)
						# ground truth boxes
						gt_box, labels = gt_boxes[image_path]
						gt_box = np.array(gt_box); labels = np.array(labels)
						draw_gt_boxes(gt_image, gt_box, labels, colors, classes)
						# save prediction visual results
						#cv2.namedWindow('image', cv2.WINDOW_NORMAL)
						model_dir = os.path.dirname(model_path)
						result_dir = model_dir + '/visual_results'
						if not os.path.exists(result_dir):
								os.system('mkdir -p {}'.format(result_dir))
						cv2.imwrite(result_dir + '/pr_{}'.format(os.path.basename(image_path)), src_image)
						cv2.imwrite(result_dir + '/gt_{}'.format(os.path.basename(image_path)), gt_image)
						cv2.waitKey(0)

if __name__ == '__main__':
    main()