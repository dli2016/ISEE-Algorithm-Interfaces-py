# How to add an algoritm for Panoptic Segmentation

The text illustrates how to achieve the panoptic segmentation class to run on ISEE.

* Upload the project folder with the name "methedname_version".
* The project folder should include the dependencies to complete the uploaded method.
* To achieve the Panoptic Segmentation, one should complete the content provided in "PanopticSegmentation.py":
  + The class of ISEEPanopticSegmentation is inherented from the abstract class "ISEEVisAlgIntf"
      I.  Don't change the class name and functions name.
      II. Don't change the order of the parameters in the fucntions.
  + The return of the function "getResults" will influence the data management in ISEE, so the results should follow the format:


    ```
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
    ```

* An example of panoptic segmentation achieved by detectron is provided.
