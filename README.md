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
  + **methodname_version**: the root folder for the user uploaded algorithms.
  the user should name the folder through combining the method name and
  version flag.
    - **DenseposeEstimation.py**: complete the ISEEDenseposeEstimation by
    implement the methods in ISEEVisAlgIntf.
    - **dependencies**: all the dependent codes or packages to achieve densepose
    estimation.
  + **densepose_rcnn_coco**: it provides an example to densepose estimation using
  the projects of [Detection2](https://github.com/facebookresearch/detectron2)
  and [DensePose](https://github.com/facebookresearch/detectron2/tree/master/projects/DensePose).
* **instance_segmentation**: the module for instance segmentation (its orgnization
  is similar with densepose_estimation).
* **keypoints_detection**: the module for keypoints detection (its orgnization is
  similar with densepose_estimation).
* **object_detection**: the module for object detection (its orgnization is
  similar with densepose_estimation).
* **panoptic_segmentation**: the module for panoptic segmentation (its orgnization is
  similar with densepose_estimation).

### Remarks

* To run the custom algorithms on ISEE, only a cofiguration file (e.g., XML, JSON, YAML, etc.)
can be provided to initialize the method.
* The output formats of different algorthims have speficied in each module (more details can
be found in the README of each module). Please do not change the format which will influence 
the data management in ISEE.
* More vision algorithms will be introduced in the future.

### Acknowledgement 

* [Detectron2](https://github.com/facebookresearch/detectron2)
* [DensePose](https://github.com/facebookresearch/detectron2/tree/master/projects/DensePose)

### Citing ISEE

```BibTex
@article{li2019isee,
  title={ISEE: An Intelligent Scene Exploration and Evaluation Platform for Large-Scale Visual Surveillance},
  author={Li, Da and Zhang, Zhang and Yu, Kai and Huang, Kaiqi and Tan, Tieniu},
  journal={IEEE Transactions on Parallel and Distributed Systems},
  volume={30},
  number={12},
  pages={2743--2758},
  year={2019},
  publisher={IEEE}
}
```

