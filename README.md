# Object Detection using Histogram of Oriented Gradients

This program is an implementation of object detection based on gradient features and sliding window classification

To facilitate edge detection, the program first processes an input by thresholding each pixel to create a binary image

A histogram of oriented gradients (HOG) is then generated from the binary image by over each 8x8 block of pixels in an image. The program bins the orientation into 9 equal sized bins between -pi/2 and pi/2. If the input image dimensions are not a multiple of 8, ***np.pad*** is used with the ***mode=edge*** option to pad the width and height up to the nearest integer multiple of 8.

To determine if a pixel is an edge, the program use a threshold of 10% of the maximum gradient magnitude in the image. Since each 8x8 block will contain a different number of edges, it will normalize the resulting histogram for each block to sum to 1 (i.e., ***np.sum(ohist,axis=2)*** should be 1 at every  location).

The program loops over the orientation bins. For each orientation bin it will identify those pixels in the image whose gradient magnitude is above the threshold and whose orientation falls in the given bin. To collect up pixels in each 8x8 spatial block, it uses the function ***ski.util.view_as_windows(...,(8,8),step=8)*** and ***np.count_nonzeros*** to count the number of edges in each block.

![image](https://user-images.githubusercontent.com/44386004/179329503-f9b640f6-c1f4-4e9b-bdf8-95130522720e.png)

Detection is then achieved by correlate the generated template with the a HOG of a feature map. Since the feature map and template are both three dimensional, each orientation is filtered separately and then summed up to get the final response.

When constructing the list of top detections, the program uses non-maxima suppression so that it doesn't return overlapping detections. It achieves this by sorting the responses in descending order of their score. Every time a detection is added to the list to return, the program checks to make sure that the location of this detection is not too close to any of the detections already in the output list. The overlap is estimated by computing the distance between a pair of detections and checking that the distance is greater than 70% of the width of the template.

![image](https://user-images.githubusercontent.com/44386004/179330257-28e23a49-31ac-4f45-a503-3f2645263381.png)

Finally basic learning is achieved by using multiple input example. The program learns a template from positive and negative examples. It takes a collection of cropped positive and negative examples of the object we are interested in detecting, extract the features for each, and generate a template by taking the average positive template minus the average negative template.

![image](https://user-images.githubusercontent.com/44386004/179330740-8a660311-8947-4f82-9dec-909e35e0dbcb.png)
![image](https://user-images.githubusercontent.com/44386004/179330779-cee9f5fc-20a3-43b6-9068-6dcff287cd91.png)


The detector implemented works well when the shapes of the objects are mostly invariant. Major changes in the angles of objects, and images that are visually busy proved challenging for the algoritm to navigate. Furthermore, it makes a big assumption about the size of the objects within the image, thus failing when the objects are presented larger or smaller than expected. The algorithm can probably be improved by implementing a way to use color to assist detection, since currently only the shapes are utilized.


