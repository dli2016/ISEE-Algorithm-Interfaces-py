"""
The demo show how to achive densepose estimation with ISEEDenseposeEstimation.
"""

import sys
import time
sys.path.append('./')
from densepose_estimation.densepose_rcnn_coco.DenseposeEstimation import ISEEDenseposeEstimation
from detectron2.data.detection_utils import read_image

if __name__ == '__main__':
    # Parameters
    params_dict = {
        'gpu_id': [0],
        'model_path': [
            '../3dparties/detectron2/projects/DensePose/densepose_rcnn_R_50_FPN_s1x.pkl',
        ],
        'reserved': {
            'method': 'Densepose-RCNN',
            'roi_threshold': 0.8
        }
    }
    
    # Configuration file path
    config_file = '../3dparties/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml'
    # Input
    img_fpath = 'demo/data/input2.jpg'
    
    # Conduct prediction
    predictor = ISEEDenseposeEstimation()
    # 1. Initialization
    err_no = predictor.init(config_file, params_dict)
    if err_no < 0:
        err_type = ISEEDenseposeEstimation.getErrType(err_no)
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
    visualizations = 'dp_contour,bbox'
    stamp1 = time.time()
    err_no = predictor.process(imgs_data, output=output, visualizations=visualizations)
    stamp2 = time.time()
    if err_no < 0:
        err_type = ISEEDenseposeEstimation.getErrType(err_no)
        print("ERROR: the densepose estimation run FAILED - {}".format(err_type))
    else:
        print("INFO: the densepose estimation run SUCCESSFULLY!")
    # 4. Get results
    densepose_list = predictor.getResults()
    print('INFO: prediction DONE, the densepose of {} detected instances are estimated and {:.4f} s is cost.'
      .format(densepose_list[0]['scores'].shape[0], stamp2 - stamp1))

