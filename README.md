# Object Detection using Histogram of Oriented Gradients

This program is an implementation of object detection based on gradient features and sliding window classification

To facilitate edge detection, the program first processes an input by thresholding each pixel to create a binary image

A histogram of oriented gradients (HOG) is then generated from the binary image by over each 8x8 block of pixels in an image. The program bins the orientation into 9 equal sized bins between -pi/2 and pi/2. If the input image dimensions are not a multiple of 8, ***np.pad*** is used with the ***mode=edge*** option to pad the width and height up to the nearest integer multiple of 8.

To determine if a pixel is an edge, the program use a threshold of 10% of the maximum gradient magnitude in the image. Since each 8x8 block will contain a different number of edges, it will normalize the resulting histogram for each block to sum to 1 (i.e., ***np.sum(ohist,axis=2)*** should be 1 at every  location).

The program loops over the orientation bins. For each orientation bin it will identify those pixels in the image whose gradient magnitude is above the threshold and whose orientation falls in the given bin. To collect up pixels in each 8x8 spatial block, it uses the function ***ski.util.view_as_windows(...,(8,8),step=8)*** and ***np.count_nonzeros*** to count the number of edges in each block.

![image](https://user-images.githubusercontent.com/44386004/179329503-f9b640f6-c1f4-4e9b-bdf8-95130522720e.png)
