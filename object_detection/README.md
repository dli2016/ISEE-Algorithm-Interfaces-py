# How to add an algoritm for Object Detection

The text illustrates how to add custom object detection class to run on ISEE.

* Upload the project folder with the name "methedname_version", e.g. faster_rcnn_detectron, ssd_pytorch, yolo_v3 etc.
* The project folder should include the dependencies to complete the uploaded method.
* To achieve Object Detection, one should complete the content provided in "ObjectDetection.py":
  + The class of ISEEObjectDetection is inherented from the abstract class "ISEEVisAlgIntf"
      I.  Don't change the class name and functions name.
      II. Don't change the order of the parameters in the fucntions.
  + The return of the function "getResults" will influence the data management in ISEE, so the results should follow the format:


    ```
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
    ```

    - (x0, y0): The left-top of the detected bbox.
    - (x1, y1): The right-bottom of the detected bbox.
    - score: the confidence of the detected bbox.
    - label: the category label of the detected bbox.

* An example of object detection with faster rcnn that achieved by detectron is provided.
