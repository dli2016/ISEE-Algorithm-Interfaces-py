# How to add an algoritm for Instance Segmentation

The text illustrates how to achieve the instance segmentation class to run on ISEE.

* Upload the project folder with the name "methedname_version".
* The project folder should include the dependencies to complete the uploaded method.
* To achieve the Instance Segmentation, one should complete the content provided in "InstanceSegmentation.py":
  + The class of ISEEInstanceSegmentation is inherented from the abstract class "ISEEVisAlgIntf"
      I.  Don't change the class name and functions name.
      II. Don't change the order of the parameters in the fucntions.
  + The return of the function "getResults" will influence the data management in ISEE, so the results should follow the format:


    ```
    # A list of dict whose length is the same with the number of input images.
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
    ```

* An example of instance segmentation achieved by detectron is provided.
