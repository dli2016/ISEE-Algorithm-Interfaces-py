"""
Brief     : The class for panoptic segmentation. It is inherented from the 
            abstract class ISEEVisAlgIntf.
Version   : 0.1
Date      : 2020/02/25
Copyright : CRIPAC
"""

import os
from common.isee_interface import ISEEVisAlgIntf

# Import necessary packaages:
# (User customs packages)
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
        # Set parameters
        # Create predictor
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
