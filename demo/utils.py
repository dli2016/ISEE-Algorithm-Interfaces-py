
# Common functions used in the demos.
import cv2
import numpy as np
from densepose_estimation.densepose_rcnn_coco.dependencies import *

# Densepose visualization
DENSE_POSE_VIS = {
    "dp_contour": DensePoseResultsContourVisualizer,
    "dp_segm": DensePoseResultsFineSegmentationVisualizer,
    "dp_u": DensePoseResultsUVisualizer,
    "dp_v": DensePoseResultsVVisualizer,
    "bbox": ScoredBoundingBoxVisualizer,
}

# Write the specified densepose results to an image.
def showDenseposeResults(res, image, out_fpath, visualizations):
    """
    params:
      res: densepose results
      image: input image data
      out_fpath: the file path for saving the result image
      visualizations: the specified results to visualize
    """
    vis_specs = visualizations.split(',')
    visualizers = []
    extractors = []
    for vis_spec in vis_specs:
        vis = DENSE_POSE_VIS[vis_spec]()
        visualizers.append(vis)
        extractor = create_extractor(vis)
        extractors.append(extractor)
    visualizer = CompoundVisualizer(visualizers)
    extractor = CompoundExtractor(extractors)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
    data = extractor(res)
    image_vis = visualizer.visualize(image, data)
    cv2.imwrite(out_fpath, image_vis)
    return

# Convert bboxes format.
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
