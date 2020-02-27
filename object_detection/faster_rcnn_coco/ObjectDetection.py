"""
Brief     : The class for ObjectDetection. It is inherented from the abstract 
            class ISEEVisAlgIntf.
Version   : 0.1
Date      : 2020/02/12
Copyright : CRIPAC
"""

import os
from common.isee_interface import ISEEVisAlgIntf

# Import necessary packaages:
# Following is an EXAMPLE that achieves object detection using detectron2
# (https://github.com/facebookresearch/detectron2)
from detectron2.config import get_cfg # configuration parameters.
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from demo.utils import convertBBoxesResFormat
# Import packages end.

class ISEEObjectDetection(ISEEVisAlgIntf):

    # The variable can be modified by "config_file" or "params_dict" in init
    _DETECTION_METHOD = 'Faster-RCNN'

    def __init__(self):
        super(ISEEObjectDetection, self).__init__()

    def init(self, config_file, params_dict=None):
        """
        Initialize the detection model.
        
        config_file:
            The path of the configuration file that contains the necessary
            parameters to iniialize the detection network (the formats, e.g. 
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
        # Detection method
        self._DETECTION_METHOD = params_dict['reserved']['method']
        # Device type
        device_type = params_dict['gpu_id'][0]
        if device_type < 0:
            device = 'cpu'
        else:
            device = 'cuda:{}'.format(params_dict['gpu_id'][0]) # Only one gpu mode is supported
        # Set parameters
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.MODEL.DEVICE = device
        cfg.MODEL.WEIGHTS = params_dict['model_path'][0]
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = params_dict['reserved']['roi_threshold']
        cfg.freeze()
        # For showing the detection results.
        metadata = MetadataCatalog.get(
          cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self._metadata = metadata
        # Create detector
        self._detector = DefaultPredictor(cfg)
        # (user custom code END)

        return self._isee_errors['success']

    def process(self, imgs_data, **kwargs):
        """
        Achieve detection with loaded model.

        imgs_data:
            A list images data to detect.
        **kwargs :
            The necessary parameters to implement inference combing 
            the results of other tasks.

        return:
            error code: 0 for success; a negative number for the ERROR type.
        """
        # Input checking
        if imgs_data is None or len(imgs_data) == 0:
            return self._isee_errors['null_data']
        detector = self._detector
        if detector is None:
            return self._isee_errors['null_predictor']

        # Predict
        # (user custom code START)
        # An EXAMPLE using detectron2
        output_path = kwargs['output']
        metadata = self._metadata
        detect_res = []
        cnt = 0
        for img in imgs_data:
            cnt += 1
            out = detector(img)
            detect_res.append(out)
            # To write the detection results to an image.
            if output_path is not None:
                img = img[:, :, ::-1]
                vis = Visualizer(img, metadata)
                instances = out['instances'].to('cpu')
                vis_img = vis.draw_instance_predictions(predictions=instances)
                ifname = 'object_detection_res_{}.jpg'.format(cnt)
                ifpath = os.path.join(output_path, ifname)
                vis_img.save(ifpath)
        self._detection_res = detect_res
        # (user custom code END)

        return self._isee_errors['success']

    def getResults(self):
        """
        Convert the detection results to necessary format.
        The detection results must follow the format:
        [
          # image one
          [[x0, y0, x1, y1, score, label], # a bbox
           [x0, y0, x1, y1, score, label],
           ...,
           [x0, y0, x1, y1, score, lable]],
          ...,
          # image N
          [[x0, y0, x1, y1, score, label],
           [x0, y0, x1, y1, score, label],
           ...,
           [x0, y0, x1, y1, score, lable]]
        ]

        return:
            The detection results. None without calling the function of process.
        """
        detect_res = self._detection_res
        bboxes_list = []
        # (user custom code START)
        # Convert the data format to the necessary one.
        for output in detect_res:
            instances = output['instances'].to('cpu')
            bboxes = convertBBoxesResFormat(instances)
            bboxes_list.append(bboxes)
        # (user custom code END)

        return bboxes_list

    def release(self):
        """
        Release the used resources.
        """
        print('INFO: no use now')
        return self._isee_errors['success']
    
    @classmethod
    def showPredictionMethod(self):
        print("INFO: %s is used for detection!" % self._DETECTION_METHOD)
