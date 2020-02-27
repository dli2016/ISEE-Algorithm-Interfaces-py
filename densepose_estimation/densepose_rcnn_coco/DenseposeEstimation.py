"""
Brief     : The class Densepose Estimation. It is inherented from the abstract 
            class ISEEVisAlgIntf.
Version   : 0.1
Date      : 2020/02/27
Copyright : CRIPAC
"""

import os
from common.isee_interface import ISEEVisAlgIntf

# Import necessary packaages:
# Following is an EXAMPLE that achieves densepose estimation using detectron2
# (https://github.com/facebookresearch/detectron2)
import cv2
import numpy as np
import torch
from detectron2.config import get_cfg # configuration parameters.
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.boxes import BoxMode
from densepose_estimation.densepose_rcnn_coco.dependencies import *
from demo.utils import showDenseposeResults
# Import packages end.

class ISEEDenseposeEstimation(ISEEVisAlgIntf):

    # The variable can be modified by "config_file" or "params_dict" in init
    _ESTIMATION_METHOD = 'Densepose-RCNN'

    def __init__(self):
        super(ISEEDenseposeEstimation, self).__init__()

    def init(self, config_file, params_dict=None):
        """
        Initialize the densepose estimation model.
        
        config_file:
            The path of the configuration file that contains the necessary
            parameters to iniialize the prediction network (the formats, e.g. 
            YAML, Json and XML etc,, are recomended).
        params_dict:
            The necessary parameters to initialize the project. It is in the 
            type of dictionary as follows:
            {
              gpu_id: [-1], # the gpu id (a list of Integers), -1 means using CPU.
              model_path: ['/home/yourmodelpath', ..., ''], # a list of strings.
              reserved: {}  # other necessary parameters.
            }
            NOTE: If overlapped parameters are existed in the configuration file 
            and the variable of params_dict, the latter (params_dict) one will 
            be used.
        
        return: 
            error code: 0 for success; a negative number for the ERROR type.
        """
        # Input checking.
        if not os.path.exists(config_file):
            return self._isee_errors['no_such_file']

        # (user custom code START)
        # Load parameters. 
        # An EXAMPLE using detectron2.
        # set the method for densepose
        self._ESTIMATION_METHOD = params_dict['reserved']['method']
        # Device type
        device_type = params_dict['gpu_id'][0]
        if device_type < 0: # Only gpu is supported for densepose.
            device = 'cpu'
            return self._isee_errors['bad_device_id']
        else:
            device = 'cuda:{}'.format(params_dict['gpu_id'][0]) # Only one gpu mode is supported
        # Set parameters
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(config_file)
        cfg.MODEL.DEVICE = device
        cfg.MODEL.WEIGHTS = params_dict['model_path'][0]
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = params_dict['reserved']['roi_threshold']
        cfg.freeze()
        self._cfg = cfg # For showing the estimation results.
        # Create estimater
        self._estimater = DefaultPredictor(cfg)
        # (user custom code END)

        return self._isee_errors['success']

    def process(self, imgs_data, **kwargs):
        """
        Achieve densepose estimation with loaded model.

        imgs_data:
            A list images data to predict.
        **kwargs :
            The necessary parameters to implement inference combing 
            the results of other tasks.

        return:
            error code: 0 for success; a negative number for the ERROR type.
        """
        # Input checking
        if imgs_data is None or len(imgs_data) == 0:
            return self._isee_errors['null_data']
        estimater = self._estimater
        if estimater is None:
            return self._isee_errors['null_predictor']

        # Predict
        # (user custom code START)
        # An EXAMPLE using detectron2
        output_path = kwargs['output']
        visualizations = kwargs['visualizations']
        densepose_res = []
        cnt = 0
        for img in imgs_data:
            cnt += 1
            with torch.no_grad():
                out = estimater(img)['instances']
            densepose_res.append(out)
            # To write the estimation results to image.
            if output_path is not None:
                fname = 'densepose_res_{}.jpg'.format(cnt)
                fpath = os.path.join(output_path, fname)
                showDenseposeResults(out, img, fpath, visualizations)
        self._densepose_res = densepose_res
        # (user custom code END)

        return self._isee_errors['success']

    def getResults(self):
        """
        Convert the estimation results to necessary format.

        return:
          densepose_res_list: it is a list of dict that each element in the list
          is corresponding to the reslut of one image. The format of the list is
          as follows:
          [
            # image one
            {
              'scores': [],  # numpy array, a vector contians the confidence of each detected instance
              'bboxes': [[x0,y0,x1,y1], ..., [x0,y0,x1,y1]], # numpy array, the coordinates of each
                                                               detected bbox.
              'pred_densepose': [((3, H, W), iuv_png_data), ..., ((3, H, W), iuv_png_data)]
              # H is the height of detected bbox, W is its width. iuv_png_data is the densepose resutls
              # encoded into png data.
              # Note:
              # 1. The fuction of DensePoseResult.decode_png_data in the project DensePose can be used
              #    to decode iuv_png_data to get iuv data.
              # 2. The shape of the decoded iuv data is (3, H, W).
              # 3. iuv[0, :, :] is the patch index of image points, indicating which of the 24 surface 
              #    patches the point is on.
              # 4. iuv[1, :, :] is the U-coordinate value of image points.
              # 5. iuv[2, :, :] is the v-coordinate value of image points.
              # 6. More details in https://github.com/dli2016/detectron2/blob/master/projects/DensePose/doc/TOOL_APPLY_NET.md
            },
            ...,
            # Image I
            {
              'scores': [],
              'bboxes': [[x0,y0,x1,y1], ..., [x0,y0,x1,y1]],
              'pred_densepose': [((3, H, W), iuv_png_data), ..., ((3, H, W), iuv_png_data)]
            }
          ]
        """
        densepose_res = self._densepose_res
        densepose_res_list = []
        # (user custom code START)
        # Convert the data format to the necessary one.
        for output in densepose_res:
            instances = output.to('cpu')
            results = {}
            if instances.has("scores"):
                results['scores'] = instances.scores.to('cpu').numpy()
            if instances.has("pred_boxes"):
                results['bboxes'] = instances.pred_boxes.tensor.to('cpu').numpy()
                if instances.has("pred_densepose"):
                    boxes_XYWH = BoxMode.convert(
                        instances.pred_boxes.tensor.to('cpu'), 
                        BoxMode.XYXY_ABS, 
                        BoxMode.XYWH_ABS
                    )
                    results["pred_densepose"] = \
                        instances.get("pred_densepose").to_result(boxes_XYWH).results
            densepose_res_list.append(results)
        # (user custom code END)

        return densepose_res_list

    def release(self):
        """
        Release the used resources.
        """
        print('INFO: no use now')
        return self._isee_errors['success']
    
    @classmethod
    def showPredictionMethod(self):
        print("INFO: %s is used for densepose estimation!" % self._ESTIMATION_METHOD)
