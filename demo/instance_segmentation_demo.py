"""
The demo show how to achive instance segmentation with ISEEInstanceSegmentation.
"""

import sys
import time
sys.path.append('./')
from instance_segmentation.mask_rcnn_coco.InstanceSegmentation import ISEEInstanceSegmentation
from detectron2.data.detection_utils import read_image

if __name__ == '__main__':
    # Parameters
    params_dict = {
        'gpu_id': [-1],
        'model_path': [
            'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl',
        ],
        'reserved': {
            'method': 'Mask-RCNN',
            'roi_threshold': 0.5
        }
    }
    
    # Configuration file path
    config_file = '../3parties/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
    # Input
    img_fpath = 'demo/data/input2.jpg'
    
    # Conduct instance segmentation
    predictor = ISEEInstanceSegmentation()
    # 1. Initialization
    err_no = predictor.init(config_file, params_dict)
    if err_no < 0:
        err_type = ISEEInstanceSegmentation.getErrType(err_no)
        print("ERROR: initialize the predictor FAILED - {}".format(err_type))
        exit(err_no)
    else:
        print("INFO: initialize the predictor SUCCESSFULLY!")
    # 2. Load image
    imgs_data = []
    img = read_image(img_fpath, format="BGR")
    imgs_data.append(img)
    # 3. Predict
    output = 'demo/data/'
    stamp1 = time.time()
    err_no = predictor.process(imgs_data, output=output)
    stamp2 = time.time()
    if err_no < 0:
        err_type = ISEEInstanceSegmentation.getErrType(err_no)
        print("ERROR: instance segmentation is conducted FAILED - {}".format(err_type))
    else:
        print("INFO: instance segmentation is conducted SUCCESSFULLY!")
    # 4. Get results
    segment_res_list = predictor.getResults()
    print('INFO: prediction DONE, {} instance are segmented and {:.4f} s is cost.'
       .format(segment_res_list[0]['labels'].shape[0], stamp2 - stamp1))
