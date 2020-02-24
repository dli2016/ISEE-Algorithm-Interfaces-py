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
# (user customs code START)
# import ...
# (User customs code END)
# Import packages end.

class ISEEObjectDetection(ISEEVisAlgIntf):

    # The variable can be modified by "config_file" or "params_dict" in init
    _DETECTION_METHOD = 'Methodname'


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

        # (user customs code START)
        # Set parameters. 
        # Create detector
        # (user customs code END)

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
        # (user custom code END)

        return self._isee_errors['success']

    def getResults(self):
        """
        Convert the detection results to necessary format.
        The detection results must follow the format:
        [
          [[x0, y0, x1, y1, score, label],
           [x0, y0, x1, y1, score, label],
           ...,
           [x0, y0, x1, y1, score, lable]],
          ...,
          [[x0, y0, x1, y1, score, label],
           [x0, y0, x1, y1, score, label],
           ...,
           [x0, y0, x1, y1, score, lable]]
        ]

        return:
            The dtection results. None without calling the function of process.
        """
        detect_res = self._detection_res
        bboxes_list = []
        # (user custom code START)
        # Convert the data format to the necessary one.
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
