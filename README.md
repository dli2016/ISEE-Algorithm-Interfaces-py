## ISEE-Algorithm-Interfaces-py

This project defines the interfaces of the vision algorithms integrated in 
ISEE platform (python version).
The Intelligent Scene Exploration and Evaluation 
([ISEE](https://ieeexplore.ieee.org/document/8734005)) platform is a general
computer vision system for large-scale video parsing. The users can upload
their custom vision algorithms and design the execution plan flexibly. To
achieve this, we release the interfaces of different vision algorithms, so
that the user can run their algorithms on ISEE conveniently.

### Organization

* **common**: the file isee_interface.py in this folder define the abstract 
class ISEEVisAlgIntf that all the other vision algorithm classes should
inherent.
* **demo**: it contains all the main files to run the example algorithms that
achieve by inherenting the abstract class ISEEVisAlgIntf.
* **densepose_estimation**: the module for densepose estimation.
  + methedname_version: the root folder for the user uploaded algorithms.
  the user should name the folder through combining the methed name and
  version flag.
    - DenseposeEstimation.py: complete the ISEEDenseposeEstimation by
    implement the methods in ISEEVisAlgIntf.
    - dependencies: all the dependent codes or packages to achieve densepose
    estimation.
  + densepose_rcnn_coco: it provides an example to densepose estimation using
  the projects of [Detection2](https://github.com/facebookresearch/detectron2)
  and [DensePose](https://github.com/facebookresearch/detectron2/tree/master/projects/DensePose).
* **instance_segmentation**: the module for instance segmentation (its orgnization
  is similar with densepose_estimation).
* **keypoints_detection**: the module for keypoints detection (its orgnizaion is
  similar with densepose_estimation).
