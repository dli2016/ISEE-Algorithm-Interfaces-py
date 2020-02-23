"""
The demo show how to achive object detection with ISEEObjectDetection.
"""

import sys
import time
sys.path.append('./')
from keypoints_detection.baseline_coco.KeypointsDetection import ISEEKeypointsDetection
from detectron2.data.detection_utils import read_image

if __name__ == '__main__':
    # Parameters
    params_dict = {
        'gpu_id': [-1],
        'model_path': [
            'detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl',
        ],
        'reserved': {
            'method': 'COCO-Baseline',
            'roi_threshold': 0.5
        }
    }
    
    # Configuration file path
    config_file = '../3parties/detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'
    # Input
    img_fpath = 'demo/data/input2.jpg'
    
    # Conduct detection
    detector = ISEEKeypointsDetection()
    # 1. Initialization
    err_no = detector.init(config_file, params_dict)
    if err_no < 0:
        err_type = ISEEObjectDetection.getErrType(err_no)
        print("ERROR: initialize the detector FAILED - {}".format(err_type))
        exit(err_no)
    else:
        print("INFO: initialize the detector SUCCESSFULLY!")
    # 2. Load image
    imgs_data = []
    img = read_image(img_fpath, format="BGR")
    imgs_data.append(img)
    # 3. Pridect
    output = 'demo/data/'
    stamp1 = time.time()
    err_no = detector.process(imgs_data, output=output)
    stamp2 = time.time()
    if err_no < 0:
        err_type = ISEEObjectDetection.getErrType(err_no)
        print("ERROR: the detector predicts FAILED - {}".format(err_type))
    else:
        print("INFO: the detector predicts SUCCESSFULLY!")
    # 4. Get results
    #bboxes = detector.getResults()
    #print('INFO: prediction DONE, {} objects are detected and {:.4f} s is cost.'
    #  .format(bboxes[0].shape[0], stamp2 - stamp1))
    #print(err)
    #ISEEObjectDetection.showCurrentDetectionMethod()

