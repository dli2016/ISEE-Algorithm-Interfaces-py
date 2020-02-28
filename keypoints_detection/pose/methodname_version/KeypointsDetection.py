"""
Brief     : The class for Pose Keypoints Detection. It is inherented from the 
            abstract class ISEEVisAlgIntf.
Version   : 0.2
Date      : 2020/02/23, 2020/02.28
Copyright : CRIPAC
"""

import os
from common.isee_interface import ISEEVisAlgIntf

# Import necessary packaages:
# (user customs code START)
# import ...
# (User customs code END)
# Import packages end.

class ISEEPoseKeypointsDetection(ISEEVisAlgIntf):

    # The variable can be modified by "config_file" or "params_dict" in init
    _DETECTION_METHOD = 'Methodname'


    def __init__(self):
        super(ISEEPoseKeypointsDetection, self).__init__()

    def init(self, config_file, params_dict=None):
        """
        Initialize the pose keypoints detection model.
        
        config_file:
            The path of the configuration file that contains the necessary
            parameters to iniialize the preidction network (the formats, e.g. 
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

        return:
           keypoints_list: The dtection results. Each element is corresponding
              to an input image. And They are orgnized as follows: 
              [
                # image one:
                [
                  # person one
                  [
                    [x, y, probability ( or visibility)] # a keypoint
                    ...,
                    [x, y, probability ( or visibility)]
                  ],
                  ...,
                  # person P
                  [
                    [x, y, probability ( or visibility)] # a keypoint
                    ...,
                    [x, y, probability ( or visibility)]
                  ]
                ],
                ...,
                # image N
                [
                  [
                    [x, y, probability ( or visibility)] # a keypoint
                    ...,
                    [x, y, probability ( or visibility)]
                  ],
                  ...,
                  [
                    [x, y, probability ( or visibility)] # a keypoint
                    ...,
                    [x, y, probability ( or visibility)]
                  ]
                ]
              ]

            None without calling the function of process.
        """
        detect_res = self._detection_res
        keypoints_list = []
        # (user custom code START)
        # Convert the data format to the necessary one.
        # (user custom code END)

        return keypoints_list

    def release(self):
        """
        Release the used resources.
        """
        print('INFO: no use now')
        return self._isee_errors['success']
    
    @classmethod
    def showPredictionMethod(self):
        print("INFO: {} is used for person keypoints detection!".
            format(self._DETECTION_METHOD))
