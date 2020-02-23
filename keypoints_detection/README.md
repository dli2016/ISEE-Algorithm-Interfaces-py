# How to add an algoritm for Keypoints Detection

The text illustrates how to add custom keypoints detection class to run on ISEE.

* Upload the project folder with the name "methedname_version".
* The project folder should include the dependencies to complete the uploaded method.
* To achieve Keypoints Detection, one should complete the content provided in "KeypointsDetection.py":
  + The class of ISEEKeypointsDetection is inherented from the abstract class "ISEEVisAlgIntf"
      I.  Don't change the class name and functions name.
      II. Don't change the order of the parameters in the fucntions.
  + The return of the function "getResults" will influence the data management in ISEE, so the results should follow the format:


    ```
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
    ```

* An example of person keypoints detection achieved by detectron is provided.
