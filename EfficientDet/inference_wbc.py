import cv2
import json
import numpy as np
import os
import time
import glob
from skimage import io

from model import efficientdet
from utils import preprocess_image, postprocess_boxes
from utils.draw_boxes import draw_boxes

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    phi = 0
    weighted_bifpn = False
    # model_path = 'efficientdet-d1.h5'
    model_path = '/data/results/wbc/det0/csv_08_0.0669_0.8021.h5'
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]
    # coco classes
    #classes = {value['id'] - 1: value['name'] for value in json.load(open('coco_90.json', 'r')).values()}
    classes = {0:'neutrophils', 1:'eosinophils', 2:'lymphocytes', 3:'monocytes'}
    num_classes = len(classes)
    score_threshold = 0.3
    colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]
    _, model = efficientdet(phi=phi,
                            weighted_bifpn=weighted_bifpn,
                            num_classes=num_classes,
                            score_threshold=score_threshold)
    model.load_weights(model_path, by_name=True)
		
    image_files = glob.glob('/data/datasets/wbc_1024x1024/images/*.png')
    
    #for image_path in glob.glob('datasets/VOC2007/JPEGImages/*.png'):
    for img_index, image_path in enumerate(image_files):
        print(image_path)
        image = cv2.imread(image_path)
        src_image = image.copy()
        # BGR -> RGB
        image = image[:, :, ::-1]
        h, w = image.shape[:2]

        image, scale = preprocess_image(image, image_size=image_size)
        # run network
        start = time.time()
        boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
        boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
        print(time.time() - start)
        boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)

        # select indices which have a score above the threshold
        indices = np.where(scores[:] > score_threshold)[0]

        # select those detections
        boxes = boxes[indices]
        labels = labels[indices]
        print(boxes)
        
        # prediction
        draw_boxes(src_image, boxes, scores, labels, colors, classes)
        # ground truth
        

        #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imwrite('./inference/{}'.format(os.path.basename(image_path)), src_image)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
