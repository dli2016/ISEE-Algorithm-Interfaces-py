"""
Brief     : The class for panoptic segmentation. It is inherented from the 
            abstract class ISEEVisAlgIntf.
Version   : 0.1
Date      : 2020/02/24
Copyright : CRIPAC
"""

import os
from common.isee_interface import ISEEVisAlgIntf

# Import necessary packaages:
# Following is an EXAMPLE that achieves panoptic segmentation using detectron2
# (https://github.com/facebookresearch/detectron2)
from detectron2.config import get_cfg # configuration parameters.
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
# Import packages end.

class ISEEPanopticSegmentation(ISEEVisAlgIntf):

    # The variable can be modified by "config_file" or "params_dict" in init
    _SEGMENT_METHOD = 'Panoptic-FPN'

    def __init__(self):
        super(ISEEPanopticSegmentation, self).__init__()

    def init(self, config_file, params_dict=None):
        """
        Initialize the model for panoptic segmentation.
        
        config_file:
            The path of the configuration file that contains the necessary
            parameters to iniialize the segmentation network (the formats, 
            e.g. YAML, Json and XML etc., are recomended).
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
        # Prediction method
        self._SEGMENT_METHOD = params_dict['reserved']['method']
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
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = \
            params_dict['reserved']['panoptic_threshold']
        cfg.freeze()
        # For showing the segmentation results.
        metadata = MetadataCatalog.get(
          cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self._metadata = metadata
        # Create predictor
        self._predictor = DefaultPredictor(cfg)
        # (user custom code END)

        return self._isee_errors['success']

    def process(self, imgs_data, **kwargs):
        """
        Achieve instance segmentation with loaded model.

        imgs_data:
            A list of images data to segment.
        **kwargs :
            The necessary parameters to implement inference combing 
            the results of other tasks.

        return:
            error code: 0 for success; a negative number for the ERROR type.
        """
        # Input checking
        if imgs_data is None or len(imgs_data) == 0:
            return self._isee_errors['null_data']
        predictor = self._predictor
        if predictor is None:
            return self._isee_errors['null_predictor']

        # Predict
        # (user custom code START)
        # An EXAMPLE using detectron2
        output_path = kwargs['output']
        metadata = self._metadata
        segment_res = []
        cnt = 0
        for img in imgs_data:
            cnt += 1
            out = predictor(img)
            panoptic_seg, seg_info = out['panoptic_seg']
            segment_res.append(out)
            # To write the segmentation results to an image.
            if output_path is not None:
                img = img[:, :, ::-1]
                vis = Visualizer(img, metadata)
                vis_img = vis.draw_panoptic_seg_predictions(
                    panoptic_seg.to('cpu'), seg_info
                )
                ifname = 'panoptic_segment_res_{}.jpg'.format(cnt)
                ifpath = os.path.join(output_path, ifname)
                vis_img.save(ifpath)
        self._segment_res = segment_res
        # (user custom code END)

        return self._isee_errors['success']

    def getResults(self):
        """
        Convert the results of panoptic segementation to necessary format.

        return:
            The results of panoptic segmentation. None without calling the function of process.
            The results named 'panoptic_seg_list' is a list of tuple whose format as follows:
            # The format provided by detectron2 is followed.
            [
              # Image one
              (
                # Numpy array.
                segment_id_array,  # Its shape is (H, W). And each element in it is about the
                                   # segment id of the pixel. The segment id is corresponding
                                   # to the value in the list of dict.
                # A list of dict. (The length of the list is the number of segmentations)
                [
                  # Segmentation/Instance one
                  {
                    'id': segment_id,      # Segment id.
                    'isthing': True/False, # whether the segment is a thing or stuff.
                    'category_id': category_label, # Thing/Stuff class id.
                    'area': segment_area,  # the area of the semgment.
                    'instance_id': instance_id,    # this key is valid when 'isthing' == True.
                    'score': instance_confidence   # this key is valid when 'isthing' == True.
                  },
                  ...,
                  # Segmentation/Instance K
                  {
                    'id': segment_id,
                    'isthing': True/False,
                    'category_id': category_label,
                    'area': segment_area,
                    'instance_id': instance_id,
                    'score': instance_confidence
                  }
                ]
              )
              ...,
              # Image I
              (
                segment_id_array,
                [{}, {}, ..., {}]
              )
            ]
        """
        segment_res = self._segment_res
        panoptic_seg_list = []
        # (user custom code START)
        # Convert the data format to the necessary one.
        for output in segment_res:
            panoptic_seg, segments_info = output["panoptic_seg"]
            panoptic_seg_list.append((panoptic_seg.to('cpu').numpy(), segments_info))
        # (user custom code END)

        return panoptic_seg_list

    def release(self):
        """
        Release the used resources.
        """
        print('INFO: no use now')
        return self._isee_errors['success']
    
    @classmethod
    def showPredictionMethod(self):
        print("INFO: %s is used for panoptic segmentation!" % self._SEGMENT_METHOD)
