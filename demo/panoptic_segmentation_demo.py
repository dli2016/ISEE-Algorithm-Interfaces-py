"""
The demo show how to achive panoptic segmentation with ISEEInstanceSegmentation.
"""

import sys
import time
sys.path.append('./')
from panoptic_segmentation.panoptic_fpn_coco.PanopticSegmentation import ISEEPanopticSegmentation
from detectron2.data.detection_utils import read_image

if __name__ == '__main__':
    # Parameters
    params_dict = {
        'gpu_id': [-1],
        'model_path': [
            'detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl',
        ],
        'reserved': {
            'method': 'Mask-RCNN',
            'roi_threshold': 0.5,
            'panoptic_threshold': 0.5
        }
    }
    
    # Configuration file path
    config_file = '../3parties/detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml'
    # Input
    img_fpath = 'demo/data/input2.jpg'
    
    # Conduct panoptic segmentation
    predictor = ISEEPanopticSegmentation()
    # 1. Initialization
    err_no = predictor.init(config_file, params_dict)
    if err_no < 0:
        err_type = ISEEPanopticSegmentation.getErrType(err_no)
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
        err_type = ISEEPanopticSegmentation.getErrType(err_no)
        print("ERROR: panoptic segmentation is conducted FAILED - {}".format(err_type))
    else:
        print("INFO: panoptic segmentation is conducted SUCCESSFULLY!")
    # 4. Get results
    panoptic_res_list = predictor.getResults()
    print('INFO: prediction DONE, {} segments are got and {:.4f} s is cost.'
       .format(len(panoptic_res_list[0][1]), stamp2 - stamp1))
