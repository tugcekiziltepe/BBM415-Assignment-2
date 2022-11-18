# BBM415-Assignment-2
In this assignment, mean filter, Gaussian filter with a specified parameter sigma and Kuwahara filter[1] are implemented. Our aim is to implement these filters, compare them and show the results of the filters.

I implemented convolution kernel for every filters. A convolution kernel is a correlation kernel that has been rotated 180 degrees.

I did not use padding, instead the image size gets smaller. n x m image is reduced to (n - window_size + 1) x (m - window_size + 1)
