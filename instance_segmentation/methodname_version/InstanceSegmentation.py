"""
Brief     : The class for instance segmentation. It is inherented from the 
            abstract class ISEEVisAlgIntf.
Version   : 0.1
Date      : 2020/02/24
Copyright : CRIPAC
"""

import os
from common.isee_interface import ISEEVisAlgIntf

# Import necessary packaages:
# (User custroms code)
# import ...
# Import packages end.

class ISEEInstanceSegmentation(ISEEVisAlgIntf):

    # The variable can be modified by "config_file" or "params_dict" in init
    _SEGMENT_METHOD = 'Mask-RCNN'

    def __init__(self):
        super(ISEEInstanceSegmentation, self).__init__()

    def init(self, config_file, params_dict=None):
        """
        Initialize the model for instance segmentation.
        
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
        Convert the results of instance segementation to necessary format.

        return:
            The results of instance segmentation. None without calling the function of process.
            The results named 'masks_list' is a list of dict whose format as follows:
            [
              # Image one
              {
                'scores': [], # a vector about the confidence as a numpy array.
                'labels': [], # a vector about the labels of the instances.
                'masks' : []  # its a numpy array with the shape (N, H, W), 
                                where N is the number of instances. And each
                                element is the label of the pixel with Ture/False.
              },
              ...,
              # Image I
              {
                'scores': [],
                'labels': [],
                'masks' : []
              }
            ]
        """
        segment_res = self._segment_res
        masks_list = []
        # (user custom code START)
        # Convert the data format to the necessary one.
        # (user custom code END)

        return masks_list

    def release(self):
        """
        Release the used resources.
        """
        print('INFO: no use now')
        return self._isee_errors['success']
    
    @classmethod
    def showPredictionMethod(self):
        print("INFO: %s is used for instance segmentation!" % self._SEGMENT_METHOD)
