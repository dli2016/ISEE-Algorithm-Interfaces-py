"""
Brief    : The interface class for the modules of vision algorithm used in ISEE.
Verson   : 0.1
Date     : 2020.01.19
Copyright: CRIPAC
"""

import abs

class ISEEVisAlgIntf(metaclass=abs.ABCMeta):

    @classmethod
    def verson(self):
        return "Verson 0.1"

    @abs.abstractmethod
    def init(self, params_dict):
        """
        Load model.
        params:
          params_dict: the necessary parameters to initialize the project.
          It is in the type of dictionary as follows:
          {
            gpu_id: [-1], # the gpu id (a list of Integers), -1 means using CPU.
            model_path: ['/home/yourmodelpath', ..., ''], # a list of strings.
            reserved: {}  # other necessary parameters.
          }
        return:
          error code: 0 for success; a negative number for the ERROR type.
        """
        pass

    @abs.abstractmethod
    def process(self, imgs_data, **kwargs):
        """
        Inference through loaded model.
        params:
          imgs_data: a list images data to process.
          **kwargs : the necessary parameters to implement inference combing
                     the results of other tasks.
        return:
          error code: 0 for success; a negative number for the ERROR type.
        """
        pass

    @abs.abstractmethod
    def getResults(self):
        """
        Get the processing results.
        params:
        return:
          The processing results. None without calling the function of process.
        """
        pass

    @abs.abstractmethod
    def release(self):
        """
        Release the resources.
        """
        pass
