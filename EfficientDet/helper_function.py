import cv2


def draw_pr_boxes(image, boxes, scores, labels, colors, classes):
    for b, l, s in zip(boxes, labels, scores):
        class_id = int(l)
        class_name = classes[class_id]
    
        xmin, ymin, xmax, ymax = list(map(int, b))
        score = '{:.2f}'.format(s)
        color = colors[class_id]
        label = ':'.join([class_name, score])
    
        ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        # cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
        # cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.rectangle(image, (xmin, ymax), (xmin + baseline + ret[0], ymax + baseline + ret[1]), color, -1)
        cv2.putText(image, label, (xmin, ymax + baseline//2 + ret[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def draw_gt_boxes(image, boxes, labels, colors, classes):
    for b, l in zip(boxes, labels):
        class_id = int(l)
        class_name = classes[class_id]
    
        xmin, ymin, xmax, ymax = list(map(int, b))
        color = colors[class_id]
        label = ':'.join([class_name])
    
        ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        # cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
        # cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.rectangle(image, (xmin, ymax), (xmin + baseline + ret[0], ymax + baseline + ret[1]), color, -1)
        cv2.putText(image, label, (xmin, ymax + baseline//2 + ret[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
