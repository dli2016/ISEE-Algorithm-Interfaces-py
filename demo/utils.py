
# Common functions used in the demos.

import numpy as np

def convertBBoxesResFormat(instances):
    """
    Convert the detected bboxes to necessary format:
    [
      [x0, y0, x1, y1, score, label],
      ...,
      [x0, y0, x1, y1, score, label]
    ]
    """
    boxes  = instances.pred_boxes.tensor.numpy()
    num_boxes = boxes.shape[0]
    scores = instances.scores.view(num_boxes, -1).numpy()
    classes= instances.pred_classes.view(num_boxes, -1).numpy()
    # Combine them
    res = np.hstack((boxes, scores))
    res = np.hstack((res, classes))

    return res
