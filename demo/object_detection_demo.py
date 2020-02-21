"""
The demo show how to achive object detection with ISEEObjectDetection.
"""

import sys
import time
sys.path.append('./')
from object_detection.ObjectDetection import ISEEObjectDetection
from detectron2.data.detection_utils import read_image

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
    # parameters bak:
    # Mask-RCNN
    # config_file = '../3parties/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
    # 'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'
    
    # Configuration file path
    config_file = '../3parties/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
    # Input
    img_fpath = 'demo/data/input1.jpg'
    
    # Conduct detection
    detector = ISEEObjectDetection()
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
    bboxes = detector.getResults()
    print('INFO: prediction DONE, {} objects are detected and {:.4f} s is cost.'
      .format(bboxes[0].shape[0], stamp2 - stamp1))
    #print(err)
    #ISEEObjectDetection.showCurrentDetectionMethod()

