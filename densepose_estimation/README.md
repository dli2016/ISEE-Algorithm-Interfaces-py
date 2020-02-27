# How to add an algoritm for Densepose Estimation

The text illustrates how to add custom densepose estimation class to run on ISEE.

* Upload the project folder with the name "methedname_version".
* The project folder should include the dependencies to complete the uploaded method.
* To achieve Densepose Estimation, one should complete the content provided in "DenseposeEstimation.py":
  + The class of ISEEDenseposeEstimation is inherented from the abstract class "ISEEVisAlgIntf"
    - Don't change the class name and functions name.
    - Don't change the order of the parameters in the fucntions.
  + The return of the function "getResults" will influence the data management in ISEE, so the results should follow the format:


    ```
    [
      # image one
      {
        'scores': [],  # numpy array, a vector contians the confidence of each detected instance
        'bboxes': [[x0,y0,x1,y1], ..., [x0,y0,x1,y1]], # numpy array, the coordinates of each
                                                         detected bbox.
        'pred_densepose': [((3, H, W), iuv_png_data), ..., ((3, H, W), iuv_png_data)]
        # H is the height of detected bbox, W is its width. iuv_png_data is the densepose resutls
        # encoded into png data.
      },
      ...,
      # Image I
      {
        'scores': [],
        'bboxes': [[x0,y0,x1,y1], ..., [x0,y0,x1,y1]],
        'pred_densepose': [((3, H, W), iuv_png_data), ..., ((3, H, W), iuv_png_data)]
      }
    ]
    ```

    Note:
    - The fuction of [DensePoseResult.decode_png_data()](https://github.com/facebookresearch/detectron2/blob/master/projects/DensePose/densepose/structures.py) in the project [DensePose]((https://github.com/facebookresearch/detectron2/blob/master/projects/DensePose) can be used to decode iuv_png_data to get iuv data.
    - The shape of the decoded iuv data is (3, H, W).
    - iuv[0, :, :] is the patch index of image points, indicating which of the 24 surface patches the point is on.
    - iuv[1, :, :] is the U-coordinate value of image points.
    - iuv[2, :, :] is the v-coordinate value of image points.
    - More details in the project of [DensePose](https://github.com/facebookresearch/detectron2/blob/master/projects/DensePose/doc/TOOL_APPLY_NET.md).

* An example of densepose estimation achieved by Detectron2 & Densepose is provided.
