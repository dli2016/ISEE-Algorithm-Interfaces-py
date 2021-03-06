"""
The demo show how to achive object detection with ISEEObjectDetection.
"""

import sys
import time
sys.path.append('./')
from common.isee_interface import ISEEVisAlgIntf
from detectron2.data.detection_utils import read_image

# Import the method that the user wanted to select.
def selectMethod(method_name):
    if method_name == 'Faster-RCNN':
        from object_detection.faster_rcnn_coco.ObjectDetection import ISEEObjectDetection
    predictor = ISEEObjectDetection()
    return predictor

if __name__ == '__main__':
    # Parameters
    params_dict = {
        'gpu_id': [-1],
        'model_path': [
            'detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl',
        ],
        'reserved': {
            'method': 'Faster-RCNN',
            'roi_threshold': 0.5
        }
    }
    # Configuration file path
    config_file = '../3parties/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
    # Input
    img_fpath = 'demo/data/input1.jpg'
    
    # Conduct detection
    method = 'Faster-RCNN'
    detector = selectMethod(method)
    # 1. Initialization
    err_no = detector.init(config_file, params_dict)
    if err_no < 0:
        err_type = ISEEVisAlgIntf.getErrType(err_no)
        print("ERROR: initialize the detector FAILED - {}".format(err_type))
        exit(err_no)
    else:
        print("INFO: initialize the detector SUCCESSFULLY!")
    # 2. Load image
    imgs_data = []
    img = read_image(img_fpath, format="BGR")
    imgs_data.append(img)
    # 3. Predict
    output = 'demo/data/'
    stamp1 = time.time()
    err_no = detector.process(imgs_data, output=output)
    stamp2 = time.time()
    if err_no < 0:
        err_type = ISEEVisAlgIntf.getErrType(err_no)
        print("ERROR: the detector predicts FAILED - {}".format(err_type))
    else:
        print("INFO: the detector predicts SUCCESSFULLY!")
    # 4. Get results
    bboxes = detector.getResults()
    print('INFO: prediction DONE, {} objects are detected and {:.4f} s is cost.'
      .format(bboxes[0].shape[0], stamp2 - stamp1))
