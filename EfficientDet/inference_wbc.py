import cv2
import json
import numpy as np
import os
import time
import glob

from generators.csv_ import CSVGenerator
from eval.common import evaluate

from model import efficientdet
from utils import preprocess_image, postprocess_boxes
# from utils.draw_boxes import draw_boxes

from helper_function import draw_pr_boxes, draw_gt_boxes

docker = True
dataset_dir = '/data/datasets' if docker else './datasets'

cls_label_map = {'neutrophils': 0, 'eosinophils': 1, 'lymphocytes': 2, 'monocytes': 3}

def load_test_boxes(dataset):
    data_csv_file = dataset_dir + '/' + dataset + '/docker/test_4c.csv'
    gt_boxes = {}
    with open(data_csv_file, 'r+') as f:
        lines = f.readlines()
        for line in lines:
            splits = line.strip().split(',')
            img_path = splits[0]
            box = [int(splits[1]), int(splits[2]), int(splits[3]), int(splits[4])]
            label = cls_label_map[splits[5]]
            if not img_path in gt_boxes:
                gt_boxes[img_path] = ([box], [label])
            else:
                gt_boxes[img_path][0].append(box)
                gt_boxes[img_path][1].append(label)
    return gt_boxes

# dataset = 'wbc_1024x1024'
# gt_boxes = load_test_boxes(dataset)

def main():
		os.environ['CUDA_VISIBLE_DEVICES'] = '2'
		# model_file = 'phi-1-set-wbc_1024x1024-wfpn-False-ep-50-stp-1000-bz-4/csv_02_0.4223_0.6017.h5'
		# model_file = 'phi-1-set-wbc2_1024x1024-wfpn-False-ep-50-stp-1000-bz-4/csv_02_0.3394_0.3840.h5'
		# model_file = 'phi-1-set-wbc2_1024x1024-wfpn-False-ep-50-stp-1000-bz-4/csv_02_0.3394_0.3840.h5'
		# model_file = 'phi-3-set-wbc_1024x1024-wfpn-False-ep-200-stp-1000-bz-4/csv_06_0.2215_0.9039.h5'
		# model_file = 'phi-3-set-wbc_1024x1024-wfpn-False-ep-200-stp-1000-bz-4/csv_07_0.2001_0.5420.h5'
		model_file = 'phi-4-set-wbc3_1024x1024-wfpn-False-ep-200-stp-1000-bz-2/csv_07_0.3233_0.4918.h5'
		model_name = os.path.dirname(model_file)
		splits = model_name.split('-')
		dataset = 'wbc_1024x1024'
		phi = 0
		wfpn = False
		for v, sp in enumerate(splits):
				if sp == 'set':
						dataset = splits[v + 1]
				elif sp == 'phi':
						phi = int(splits[v+1])
				elif sp == 'wfpn':
						wfpn = splits[v+1].lower() == 'true'
		# model_path = 'efficientdet-d1.h5'
		# model_path = '/data/results/wbc/det0/csv_08_0.0669_0.8021.h5'
		model_path = '/data/models/{}/{}'.format(dataset, model_file)
		image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
		image_size = image_sizes[phi]
		classes = {0:'neutrophils', 1:'eosinophils', 2:'lymphocytes', 3:'monocytes'}
		num_classes = len(classes)
		score_threshold = 0.3
		colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]
		_, model = efficientdet(phi=phi,
														weighted_bifpn=wfpn,
														num_classes=num_classes,
														score_threshold=score_threshold)
		model.load_weights(model_path, by_name=True)

		# evaluate the precisions
		evaluate_ap(model, phi, dataset)

		gt_boxes = load_test_boxes(dataset)
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

				# prediction
				draw_pr_boxes(src_image, boxes, scores, labels, colors, classes)
				# ground truth
				gt_box, labels = gt_boxes[image_path]
				gt_box = np.array(gt_box); labels = np.array(labels)
				draw_gt_boxes(gt_image, gt_box, labels, colors, classes)

				#cv2.namedWindow('image', cv2.WINDOW_NORMAL)
				model_dir = os.path.dirname(model_path)
				result_dir = model_dir + '/results'
				if not os.path.exists(result_dir):
						os.system('mkdir -p {}'.format(result_dir))
				cv2.imwrite(result_dir + '/pr_{}'.format(os.path.basename(image_path)), src_image)
				cv2.imwrite(result_dir + '/gt_{}'.format(os.path.basename(image_path)), gt_image)
				cv2.waitKey(0)

def evaluate_ap(model, phi, dataset):
		# dataset = 'wbc_1024x1024'
		val_annotations_path = dataset_dir + '/' + dataset + '/docker/test_4c.csv'
		classes_path = dataset_dir + '/' + dataset + '/docker/classes.csv'
		validation_generator = CSVGenerator(
		val_annotations_path,
		classes_path,
		shuffle_groups=False,
		phi= phi)
		score_threshold = 0.3
		average_precisions = evaluate(
				validation_generator,
				model,
				iou_threshold= 0.5,
				score_threshold= score_threshold,
				max_detections=10,
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

if __name__ == '__main__':
    main()
